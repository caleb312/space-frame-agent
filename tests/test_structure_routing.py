"""分类记录→确定性分派→用户确认→相应Solver；不调用供应商API。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
import numpy as np
from src.agent.runtime import FrameAgent,fingerprint,tools_schema
from src.agent.explanation import textbook_presentation,result_table
from tests.test_specialized_solver import model


def report_for(raw):
    facts=[{'path':'/'+key,'value':value,'state':'convention' if key=='convention' else 'user_given',
            'source':'project_convention' if key=='convention' else 'user','evidence':'明确给定'}
           for key,value in [('units',raw['units']),('convention','course_clockwise')]]
    for group in ('nodes','materials','sections','elements'):
        for i,item in enumerate(raw[group]):
            for key,value in item.items():
                if value is not None:facts.append({'path':f'/{group}/{i}/{key}','value':value,'state':'user_given','source':'user','evidence':'明确给定'})
    loads=[];kind=raw['structure_type']
    for i,load in enumerate(raw.get('loads',[])):
        v=[0.]*6
        active=[0,1,5] if kind=='plane_frame' else [0,1] if kind=='plane_truss' else [0,1,2]
        for at,val in zip(active,load['components']):v[at]=val
        force=v.copy();force[3:]=[0,0,0]
        moment=[0,0,0,*v[3:]]
        for typ,values in [('node_force',force),('node_moment',moment)]:
            if any(values):loads.append({'label':f'L{i}_{typ}','kind':typ,'target_id':load['node_id'],'components':values,
                'units':raw['units'],'coordinate_system':'global','position':None,'interval':None,
                'raw_description':'给定节点荷载','source':'user','evidence':'明确给定'})
    return {'structure_type':kind,'classification_evidence':'用户明确给定结构类型、维度和连接形式',
        'observations':['演示给定模型'],'facts':facts,'loads':loads,'loads_complete':True,'load_inventory_evidence':'已逐项登记',
        'questions':[],'requested_results':['displacements','reactions','end_forces']}


class RoutingTests(unittest.TestCase):
    def test_each_native_type_confirm_and_correct_entry(self):
        for kind in ('plane_truss','plane_frame','space_truss'):
            dimension=3 if kind=='space_truss' else 2;nd=2 if kind=='plane_truss' else 3
            fix=[False]*nd;fix[1]=kind!='plane_frame'
            if kind=='space_truss':fix[2]=True
            raw=model(kind,[[0]*dimension,[1]+[0]*(dimension-1)],[[True]*nd,fix],[(2,[1]+[0]*(nd-1))])
            agent=FrameAgent(None);draft=agent.record(report_for(raw))
            self.assertEqual(draft['status'],'awaiting_confirmation',draft.get('blockers'))
            self.assertEqual(agent.solve()['status'],'needs_confirmation')
            agent.confirm(fingerprint(agent.model))
            self.assertEqual(agent.dispatch('solve_space_frame',{})['status'],'invalid')
            result=agent.dispatch('solve_'+kind,{})
            self.assertEqual(result['status'],'solved')
            self.assertEqual(result['result']['structure_type'],kind)
            self.assertEqual(len(result['result']['native_dof_order']),nd)
            self.assertIn(kind,agent.model['structure_type'])

    def test_unknown_structure_blocks_solver(self):
        raw=model('plane_truss',[[0,0],[1,0]],[[True]*2,[False,True]],[(2,[1,0])])
        report=report_for(raw);report['structure_type']='unknown';report['classification_evidence']='无法判断连接形式'
        agent=FrameAgent(None);out=agent.record(report)
        self.assertEqual(out['status'],'needs_information')
        self.assertIsNone(agent.model)
        self.assertTrue(any('结构类型' in b for b in out['blockers']))
        patch={'base_record_hash':fingerprint(agent.draft['recognition']),'structure_type':'plane_truss',
               'classification_evidence':'用户确认二维全铰接桁架'}
        updated=agent.patch_record(patch)
        self.assertEqual(updated['status'],'awaiting_confirmation',updated.get('blockers'))
        self.assertEqual(agent.model['structure_type'],'plane_truss')

    def test_no_ignored_out_of_plane_force_or_truss_moment(self):
        raw=model('plane_truss',[[0,0],[1,0]],[[True]*2,[False,True]],[(2,[1,0])])
        for index in (2,5):
            report=report_for(raw);report['loads'][0]['components'][index]=2
            if index==5:report['loads'][0]['kind']='unknown'
            out=FrameAgent(None).record(report)
            self.assertEqual(out['status'],'needs_information')

    def test_plane_frame_endpoint_geometry_and_presentation(self):
        raw=model('plane_frame',[[0,0],[2,0]],[[True]*3,[False]*3],[(2,[0,-10,0])])
        report=report_for(raw);report['requested_results'].append('section_forces')
        report['section_queries']=[{'id':1,'element_id':100,'location':'j_end','side':'left','source':'user','evidence':'右端'}]
        agent=FrameAgent(None);out=agent.record(report)
        self.assertEqual(out['status'],'awaiting_confirmation',out.get('blockers'))
        self.assertEqual(agent.model['section_queries'][0]['position'],2)
        agent.confirm(fingerprint(agent.model));result=agent.solve()['result']
        display=textbook_presentation(result)
        self.assertLess(display['nodes'][1]['native_displacement'][2],0)
        self.assertEqual(display['nodes'][1]['native_displacement'][2],display['nodes'][1]['displacement'][5])
        self.assertEqual(display['elements'][0]['native_end_forces'][2],display['elements'][0]['end_i'][5])
        self.assertEqual(display['program_statistics']['per_node'][0]['largest_abs_rotation_component']['component'],'rz')
        self.assertIn('平面刚架',result_table(result))
        self.assertNotIn('| ux | uy | uz | rx |',result_table(result))

    def test_single_tool_call_classification_from_model_message(self):
        raw=model('plane_truss',[[0,0],[1,0]],[[True]*2,[False,True]],[(2,[1,0])])
        report=report_for(raw)
        class Client:
            def complete(self,messages,tools):
                return {'role':'assistant','content':None,'tool_calls':[{'id':'call1','type':'function',
                    'function':{'name':'record_observations','arguments':json.dumps({'report':report})}}]}
        agent=FrameAgent(Client());text=agent.run()
        self.assertIn('plane_truss',text)
        self.assertEqual(agent.model['structure_type'],'plane_truss')
        self.assertIsNone(agent.result)

    def test_tools_expose_distinct_solvers(self):
        names={t['function']['name'] for t in tools_schema()}
        self.assertTrue({'solve_plane_truss','solve_space_truss','solve_plane_frame','solve_space_frame'}<=names)
        report_schema=next(t for t in tools_schema() if t['function']['name']=='record_observations')['function']['parameters']['properties']['report']
        self.assertTrue({'structure_type','classification_evidence'}<=set(report_schema['required']))

    def test_truss_statistics_have_no_virtual_rotation_or_moment(self):
        raw=model('plane_truss',[[0,0],[2,0]],[[True]*2,[False,True]],[(2,[4,0])])
        from src.solver import solve_structure
        result=solve_structure(raw).as_dict()
        display=textbook_presentation(result)
        self.assertNotIn('largest_abs_rotation_component',display['program_statistics']['per_node'][0])
        self.assertNotIn('largest_abs_moment_component',display['program_statistics']['per_support'][0])
        self.assertEqual(result['convention'],'course_clockwise')

    def test_native_node_load_requires_coordinate_and_consistent_kind(self):
        raw=model('plane_frame',[[0,0],[2,0]],[[True]*3,[False]*3],[(2,[0,-10,0])])
        for change in ({'coordinate_system':None}, {'kind':'node_moment'},
                       {'components':[0,-10,0,0,0,2]}):
            report=report_for(raw)
            report['loads'][0].update(change)
            draft=FrameAgent(None).record(report)
            self.assertEqual(draft['status'],'needs_information')
            self.assertTrue(draft['blockers'])


if __name__=='__main__':unittest.main()
