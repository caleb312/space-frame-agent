"""仅给EA的桁架和本次纠正格式故障回归；不调用供应商API。"""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
import numpy as np
from pydantic import ValidationError
from src.agent.runtime import FrameAgent,fingerprint,tools_schema
from src.solver.parameters import calculate_material_section
from tests.test_structure_routing import report_for
from tests.test_specialized_solver import model


def axial(EA=4800000,unit='kN',units='kN_m'):
    return dict(parameter_type='axial_rigidity',material_id=1,section_id=1,EA=EA,
                EA_unit=unit,units=units,source='image',evidence='题面只给EA')


def pending_report(kind='plane_truss'):
    dim=3 if kind=='space_truss' else 2
    nd=2 if kind=='plane_truss' else 3
    fix=[False]+[True]*(nd-1)
    raw=model(kind,[[0]*dim,[1]+[0]*(dim-1)],[[True]*nd,fix],[(2,[10]+[0]*(nd-1))])
    report=report_for(raw)
    for f in report['facts']:
        if f['path'] in ('/materials/0/E','/sections/0/A'):
            f.update(value=None,state='uncertain',source='image',evidence='题面只有EA，没有独立E/A')
    return report


class AxialCorrectionTests(unittest.TestCase):
    def test_python_equivalence_and_units(self):
        kn=calculate_material_section(axial())
        other=calculate_material_section(axial(4800000000,'N'))
        newton=calculate_material_section(axial(units='N_m'))
        self.assertEqual(kn['material']['E']*kn['section']['A'],4800000)
        self.assertEqual(kn['material'],other['material'])
        self.assertEqual(newton['material']['E'],4800000000)
        self.assertNotIn('G',kn['material'])
        self.assertNotIn('Iy',kn['section'])
        for change in ({'EA':0},{'EA':-1},{'EA_unit':'MPa'},{'A':1}):
            with self.assertRaises(ValidationError):calculate_material_section({**axial(),**change})

    def test_ea_patch_retains_loads_and_requires_confirmation(self):
        for kind in ('plane_truss','space_truss'):
            a=FrameAgent(None);draft=a.record(pending_report(kind))
            self.assertEqual(draft['status'],'needs_information')
            old_loads=deepcopy(draft['recognition']['loads'])
            updated=a.patch_record(dict(base_record_hash=fingerprint(draft['recognition']),parameter_inputs=[axial()]))
            self.assertEqual(updated['status'],'awaiting_confirmation',updated.get('blockers'))
            self.assertEqual(updated['recognition']['loads'],old_loads)
            self.assertEqual(len(updated['computed_fields']),2)
            self.assertIn('归一化',updated['summary'])
            self.assertEqual(a.solve()['status'],'needs_confirmation')
            a.confirm(fingerprint(a.model));out=a.solve()['result']
            self.assertAlmostEqual(out['nodes'][1]['native_displacement'][0],10/4800000)

    def test_direct_e_and_a_use_facts_without_circle(self):
        a=FrameAgent(None);draft=a.record(pending_report())
        patch=dict(base_record_hash=fingerprint(draft['recognition']),facts=[
            dict(path=path,value=value,state='user_given',source='user',evidence='用户给定')
            for path,value in [('/materials/0/E',200000000),('/sections/0/A',.01)]])
        out=a.patch_record(patch)
        self.assertEqual(out['status'],'awaiting_confirmation')
        self.assertEqual(out['parameter_calculations'],[])
        self.assertEqual(a.model['materials'][0]['E'],200000000)

    def test_frame_rejects_ea_only_and_duplicate_normalization_conflict(self):
        report=pending_report('plane_frame');report['parameter_inputs']=[axial()]
        out=FrameAgent(None).record(report)
        self.assertEqual(out['status'],'needs_information')
        self.assertTrue(any('只适用于' in b for b in out['blockers']))
        report=pending_report();report['parameter_inputs']=[axial(),axial(1000000)]
        out=FrameAgent(None).record(report)
        self.assertEqual(out['status'],'needs_information')
        self.assertTrue(any('输入冲突' in b for b in out['blockers']))

    def test_full_schema_and_actionable_invalid_detail(self):
        schema=tools_schema(True)[0]['function']['parameters']
        self.assertIn('Fact',schema['$defs'])
        self.assertIn('AxialRigidityInput',schema['$defs'])
        self.assertIn('MaterialSectionInput',schema['$defs'])
        a=FrameAgent(None);draft=a.record(pending_report())
        old=deepcopy(draft['recognition'])
        invalid=a.patch_record(dict(base_record_hash=fingerprint(old),parameter_inputs=[dict(material_id=1,E=4800000)]))
        self.assertEqual(invalid['status'],'invalid')
        self.assertIn('parameter_inputs',invalid['message'])
        self.assertEqual(a.draft['recognition'],old)

    def test_normalized_ea_saved_parameter_restore(self):
        report=pending_report();report['parameter_inputs']=[axial()]
        a=FrameAgent(None);out=a.record(report);saved=deepcopy(a.model)
        a.parameter_results=[]
        a.restore_saved_parameters(saved,out['parameter_calculations'])
        self.assertEqual(len(a.parameter_results),1)
        self.assertEqual(a.parameter_results[0]['axial_rigidity'],4800000)

    def test_correction_prompt_contains_native_rules_and_stops_safely(self):
        captured=[]
        class Client:
            config=SimpleNamespace(vision=False)
            def complete(self,messages,tools):
                captured.append(deepcopy(messages))
                return {'role':'assistant','content':None,'tool_calls':[dict(id='call'+str(len(captured)),
                    function=dict(name='patch_observations',arguments=json.dumps({'parameter_inputs':[{}]})))]}
        a=FrameAgent(Client());a.record(pending_report())
        result=a.chat('保留题给EA，节点2两向约束')
        self.assertIn('这不代表你的自然语言纠正不合规',result)
        self.assertIn('axial_rigidity',captured[0][0]['content'])
        self.assertIn('plane_truss',captured[0][0]['content'])
        self.assertIsNone(a.model)
        self.assertIsNone(a.result)

    def test_serialized_lists_are_decoded_then_validated(self):
        report=pending_report();report['parameter_inputs']=[axial()]
        report['questions']=['需要确认支座']
        a=FrameAgent(None);draft=a.record(report)
        updated=a.patch_record(dict(base_record_hash=fingerprint(draft['recognition']),
            questions='[]',parameter_inputs=json.dumps([axial()])))
        self.assertEqual(updated['status'],'awaiting_confirmation',updated.get('blockers'))
        self.assertEqual(updated['recognition']['questions'],[])
        self.assertEqual(updated['recognition']['parameter_inputs'],report['parameter_inputs'])
        self.assertEqual(updated['recognition']['loads'],draft['recognition']['loads'])
        self.assertIsNone(a.confirmed_hash)

    def test_empty_strings_do_not_clear_questions_or_parameters(self):
        report=pending_report();report['parameter_inputs']=[axial()]
        report['questions']=['未解决的几何问题']
        a=FrameAgent(None);draft=a.record(report)
        updated=a.patch_record(dict(base_record_hash=fingerprint(draft['recognition']),
            questions='',parameter_inputs=' ',facts=[dict(path='/nodes/1/restrained',value=[False,True],
                state='user_given',source='user',evidence='用户确认约束')]))
        self.assertEqual(updated['status'],'needs_information')
        self.assertEqual(updated['recognition']['questions'],report['questions'])
        self.assertEqual(updated['recognition']['parameter_inputs'],report['parameter_inputs'])
        self.assertEqual(updated['candidate_model']['materials'][0]['E'],4800000)

    def test_non_array_numbers_text_and_bad_nested_values_remain_invalid(self):
        for field,value in [('questions',False),('parameter_inputs',123),('questions','已确认'),
                            ('parameter_inputs','[{}]')]:
            a=FrameAgent(None);draft=a.record(pending_report())
            previous=deepcopy(draft['recognition'])
            out=a.patch_record(dict(base_record_hash=fingerprint(previous),**{field:value}))
            self.assertEqual(out['status'],'invalid')
            self.assertEqual(a.draft['recognition'],previous)

    def test_single_complete_parameter_object_and_plain_array_tool_schema(self):
        a=FrameAgent(None);draft=a.record(pending_report())
        out=a.patch_record(dict(base_record_hash=fingerprint(draft['recognition']),parameter_inputs=axial()))
        self.assertEqual(out['status'],'awaiting_confirmation')
        schema=tools_schema(True)[0]['function']['parameters']
        for key in ('questions','parameter_inputs','section_queries','facts'):
            self.assertEqual(schema['properties'][key]['type'],'array')
            self.assertNotIn('anyOf',schema['properties'][key])


if __name__=='__main__':unittest.main()
