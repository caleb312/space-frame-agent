"""网页人工验收暴露问题的离线回归；无供应商请求或密钥。"""
from copy import deepcopy
import json
from unittest.mock import patch
import unittest

from src.agent.recognition import RecognizedLoad
from src.agent.explanation import explain_result, explanation_presentation
from .test_parameters import agent, inputs, recognition_report, ROOT


class CorrectionTests(unittest.TestCase):
    def report(self, radius=.05):
        report = recognition_report()
        raw = inputs()
        raw['R'] = radius
        report['parameter_inputs'] = [raw]
        return report

    def test_unchanged_inputs_reuse_without_recalculating(self):
        a = agent()
        a.record(self.report())
        old = deepcopy(a.parameter_results)
        a.client.complete = lambda *args: {'role':'assistant','content':'等待记录'}
        a.chat('只改荷载，材料不变')
        with patch('src.agent.runtime.calculate_material_section', side_effect=AssertionError('不应重复计算')):
            out = a.record(self.report())
        self.assertEqual(out['status'], 'awaiting_confirmation')
        self.assertEqual(a.parameter_results, old)
        self.assertEqual(len(out['computed_fields']), 6)
        self.assertEqual(a.solve()['status'], 'needs_confirmation')

    def test_changed_radius_recalculates(self):
        a = agent()
        a.record(self.report())
        old_area = a.model['sections'][0]['A']
        out = a.record(self.report(.06))
        self.assertEqual(out['status'], 'awaiting_confirmation')
        self.assertGreater(a.model['sections'][0]['A'], old_area)
        self.assertEqual(out['parameter_calculations'][0]['input']['R'], .06)
        self.assertEqual(len(out['parameter_calculations']), 1)

    def test_undeclared_historical_parameters_stay_blocked(self):
        a = agent()
        a.record(self.report())
        a.current_parameter_ids.clear()
        self.assertEqual(a.record(recognition_report())['status'], 'needs_information')

    def load(self, kind, values):
        return RecognizedLoad.model_validate(dict(label='L', kind=kind, target_id=1,
            components=values, units='kN_m', position=1, interval=None,
            raw_description='明确分类的集中荷载', source='user', evidence='用户明确给定',
            coordinate_system='global'))

    def test_only_inactive_components_are_normalized(self):
        self.assertEqual(self.load('member_point_force',[None,10,0,None,None,None]).components,
                         [None,10,0,0,0,0])
        self.assertEqual(self.load('member_point_moment',[None,None,None,3,None,0]).components,
                         [0,0,0,3,None,0])
        self.assertEqual(self.load('unknown',[None]*6).components,[None]*6)
        # 非零冲突值保留，由现有编译器阻止，不偷偷抹掉附加力矩。
        self.assertEqual(self.load('member_point_force',[1,0,0,2,None,None]).components,[1,0,0,2,0,0])

    def test_explanation_precision_directions_and_source_unchanged(self):
        saved = json.loads((ROOT/'tests/fixtures/iv2_display_saved.json').read_text(encoding='utf-8'))
        result = saved['result']
        original = deepcopy(result)
        display = explanation_presentation(result)
        node6 = next(n for n in display['nodes'] if n['id']==6)
        self.assertEqual(node6['displacement'][0], .00903227)
        direction = next(n for n in display['translation_directions'] if n['node_id']==6)
        self.assertEqual(direction['global'], {'x':'+x','y':'+y','z':'-z'})
        a = agent()
        captured = []
        def complete(messages, tools, purpose):
            captured.append((messages,tools,purpose))
            return {'content':'节点6沿+x、+y、−z移动。'}
        a.client.complete = complete
        explain_result(a.client,result,question='节点6向哪移动？',concise=True)
        self.assertIsNone(captured[0][1])
        self.assertEqual(captured[0][2], 'explanation')
        self.assertIn('6位有效数字',captured[0][0][0]['content'])
        self.assertEqual(result, original)


if __name__ == '__main__':
    unittest.main()
