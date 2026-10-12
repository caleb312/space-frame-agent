"""物理顺/逆时针 -> Python课程分量 -> Solver -> 右手显示/箭头全链路。"""
from copy import deepcopy
import unittest
import numpy as np
from src.agent.runtime import FrameAgent,fingerprint
from src.visualization.native import build_native_payload
from tests.test_load_location_corrections import frame_report


def ready_report(direction='clockwise',magnitude=5):
    report=frame_report();report['questions']=[]
    next(f for f in report['facts'] if f['path']=='/nodes/1/restrained').update(value=[True]*3,state='user_given',source='user')
    report['loads'][0]['position']=1;report['loads'][3]['position']=1
    report['loads'][1].update(components=None,moment_direction=direction,moment_magnitude=magnitude,
        raw_description='节点4'+('顺时针' if direction=='clockwise' else '逆时针')+'5kN*m',source='user',evidence='用户明确转向及大小')
    return report


def independent_rh_node4():
    """本题仅节点4的三个未知量：独立手列右手坐标3x3刚度与固定端等效力。

    ①L=2的j端，②L=3的i端，③L=2的j端（轴线沿Y），各端支座固定。
    中点力给j端等效力矩+PL/8；②全跨q给i端qL²/12。
    """
    EA=200000000*.1;EI=200000000*.000005
    K=np.array([[EA/2+EA/3+12*EI/2**3,0,6*EI/2**2],
        [0,EA/2+12*EI/2**3+12*EI/3**3,-6*EI/2**2+6*EI/3**2],
        [6*EI/2**2,-6*EI/2**2+6*EI/3**2,4*EI/2+4*EI/3+4*EI/2]])
    P=np.array([5,-5-3,10*2/8-2*3**2/12+10*2/8-5])
    return np.linalg.solve(K,P)


class PlaneMomentDirectionTests(unittest.TestCase):
    def test_python_compiles_physical_direction(self):
        for direction,expected in [('clockwise',5),('counterclockwise',-5)]:
            a=FrameAgent(None);draft=a.record(ready_report(direction))
            self.assertEqual(draft['status'],'awaiting_confirmation',draft.get('blockers'))
            self.assertEqual(a.model['loads'][0]['components'],[0,0,expected])
            derived=next(f for f in draft['computed_fields'] if f['source']=='python_moment_sign')
            self.assertEqual(derived['value'][5],expected)
            self.assertIsNone(a.result)

    def test_legacy_description_sign_contradiction_blocks_solver(self):
        report=ready_report();report['loads'][1].update(moment_direction=None,moment_magnitude=None,components=[0,0,0,0,0,-5])
        a=FrameAgent(None);draft=a.record(report)
        self.assertEqual(draft['status'],'needs_information')
        self.assertTrue(any('顺时针' in b and '实际Mz=-5' in b for b in draft['blockers']))
        self.assertIsNone(a.model)
        self.assertEqual(a.solve()['status'],'needs_confirmation')

    def test_duplicate_components_conflict_with_physical_input(self):
        report=ready_report();report['loads'][1]['components']=[0,0,0,0,0,-5]
        draft=FrameAgent(None).record(report)
        self.assertEqual(draft['status'],'needs_information')
        self.assertTrue(any('物理转向与Mz分量冲突' in b for b in draft['blockers']))

    def test_missing_physical_direction_or_magnitude_does_not_guess(self):
        for field in ('moment_direction','moment_magnitude'):
            report=ready_report();report['loads'][1][field]=None
            draft=FrameAgent(None).record(report)
            self.assertEqual(draft['status'],'needs_information')

    def test_direction_patch_replaces_old_sign_and_preserves_other_loads(self):
        report=ready_report();report['loads'][1].update(moment_direction=None,moment_magnitude=None,components=[0,0,0,0,0,-5])
        a=FrameAgent(None);draft=a.record(report);old=deepcopy(draft['recognition'])
        updated=a.patch_record(dict(base_record_hash=fingerprint(old),load_changes=[dict(label='M4',
            moment_direction='clockwise',moment_magnitude=5,raw_description='节点4顺时针5kN*m',evidence='用户明确转向及大小')]))
        self.assertEqual(updated['status'],'awaiting_confirmation',updated.get('blockers'))
        for index in (0,2,3):self.assertEqual(updated['recognition']['loads'][index],old['loads'][index])
        self.assertEqual(a.model['loads'][0]['components'][2],5)
        self.assertIsNone(a.confirmed_hash)

    def test_point_moment_and_clockwise_arc(self):
        report=ready_report()
        report['loads'][1].update(kind='member_point_moment',target_id=1,position=1)
        a=FrameAgent(None);draft=a.record(report)
        self.assertEqual(draft['status'],'awaiting_confirmation')
        self.assertEqual(a.model['member_loads'][1]['components'],[0,0,5])
        payload=build_native_payload(a.model)
        glyph=next(g for g in payload['glyphs'] if g['kind']=='moment')
        center=np.array(glyph['position']);curve=np.array(glyph['paths'][0])-center
        self.assertLess(np.cross(curve[0],curve[1])[2],0)

    def test_corrected_solver_matches_independent_rh_equations(self):
        a=FrameAgent(None);a.record(ready_report());a.confirm(fingerprint(a.model))
        result=a.solve()['result'];raw=next(n for n in result['nodes'] if n['id']==4)['native_displacement']
        actual=np.array(raw);actual[2]*=-1
        np.testing.assert_allclose(actual,independent_rh_node4(),atol=1e-14,rtol=1e-10)
        self.assertGreater(raw[2],0) # 本题正确结果的节点转角也是顺时针
        payload=build_native_payload(a.model,result)
        actual_moment=next(l for l in payload['loads'] if l['kind']=='moment')
        self.assertEqual(actual_moment['global_components'],[0,0,-5])
        self.assertLess(abs(result['checks']['right_hand_moment_balance'][2]),1e-10)


if __name__=='__main__':unittest.main()
