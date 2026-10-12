"""集中荷载位置补丁、无关区间及真实区间缺失的回归。"""
from copy import deepcopy
import unittest
from pydantic import ValidationError
from src.agent.recognition import RecognizedLoad
from src.agent.runtime import FrameAgent,fingerprint,tools_schema
from tests.test_structure_routing import report_for


def load(kind='member_point_force',label='P1',target=1):
    return dict(label=label,kind=kind,target_id=target,components=[0,-10,0,0,0,0],units='kN_m',
                raw_description='给定集中力',source='image',evidence='图中箭头',coordinate_system='global')


def frame_report():
    raw=dict(structure_type='plane_frame',units='kN_m',convention='course_clockwise',
        nodes=[dict(id=i,coordinates=p,restrained=fix) for i,p,fix in [
            (1,[2,0],[True]*3),(2,[0,2],[True]*3),(3,[5,2],[True]*3),(4,[2,2],[False]*3)]],
        materials=[dict(id=1,E=200000000)],sections=[dict(id=1,A=.1,I=.000005)],
        elements=[dict(id=i,node_i=a,node_j=b,material_id=1,section_id=1) for i,a,b in [(1,2,4),(2,4,3),(3,1,4)]])
    report=report_for(raw)
    next(f for f in report['facts'] if f['path']=='/nodes/1/restrained').update(value=None,state='uncertain',source='image')
    p1=load();p2=load(label='P3',target=3);p2['components']=[10,0,0,0,0,0]
    q=load('member_distributed','q2',2);q.update(components=[0,-2,0,0,0,0],interval=[0,3],distribution='uniform')
    m=load('node_moment','M4',4);m['components']=[0,0,0,0,0,None]
    report.update(loads=[p1,m,q,p2],questions=['确认支座、作用位置和力偶转向'])
    return report


class LoadLocationTests(unittest.TestCase):
    def test_irrelevant_point_intervals_omitted_or_empty(self):
        for kind in ('member_point_force','member_point_moment','node_force','node_moment'):
            for missing in (True,False):
                item=load(kind)
                if not missing:item['interval']=[]
                if kind.startswith('member_'):item['position']=1
                if kind.endswith('moment'):item['components']=[0,0,0,0,0,5]
                recognized=RecognizedLoad.model_validate(item)
                self.assertIsNone(recognized.interval)

    def test_unknown_position_is_not_filled_as_zero(self):
        a=FrameAgent(None);draft=a.record(frame_report())
        self.assertEqual(draft['status'],'needs_information')
        self.assertIsNone(draft['recognition']['loads'][0]['position'])
        self.assertIsNone(a.model)

    def test_distributed_missing_interval_still_blocks(self):
        for missing in (True,False):
            report=frame_report()
            if missing:report['loads'][2].pop('interval')
            else:report['loads'][2]['interval']=[]
            a=FrameAgent(None);draft=a.record(report)
            self.assertEqual(draft['status'],'needs_information')
            self.assertIsNone(draft['recognition']['loads'][2]['interval'])
            self.assertIsNone(a.model)

    def test_nonempty_conflicting_interval_and_wrong_length_rejected(self):
        for kind,interval in [('member_point_force',[0,1]),('node_force',[0,1]),
                              ('member_distributed',[0]),('member_distributed',[0,1,2])]:
            item=load(kind);item['interval']=interval
            with self.assertRaises(ValidationError):RecognizedLoad.model_validate(item)

    def test_partial_locations_preserve_forces_and_unmodified_loads(self):
        a=FrameAgent(None);draft=a.record(frame_report());old=deepcopy(draft['recognition'])
        patch=dict(base_record_hash=fingerprint(old),questions=[],facts=[dict(path='/nodes/1/restrained',
            value=[True]*3,state='user_given',source='user',evidence='用户明确固定端')],load_changes=[
            dict(label=label,position=1,raw_description='距当前i端1m',evidence='用户明确位置') for label in ('P1','P3')])
        patch['load_changes'].append(dict(label='M4',components=[0,0,0,0,0,5],raw_description='顺时针5kN*m',evidence='用户明确顺时针'))
        updated=a.patch_record(patch)
        self.assertEqual(updated['status'],'awaiting_confirmation',updated.get('blockers'))
        for index in (0,3):
            current=updated['recognition']['loads'][index]
            for key in ('components','target_id','units','coordinate_system'):
                self.assertEqual(current[key],old['loads'][index][key])
            self.assertEqual(current['position'],1)
        self.assertEqual(updated['recognition']['loads'][2],old['loads'][2])
        self.assertEqual(a.model['loads'][0]['components'],[0,0,5])
        self.assertIsNone(a.result)
        self.assertEqual(a.solve()['status'],'needs_confirmation')

    def test_explicit_interval_patch_preserves_components(self):
        a=FrameAgent(None);draft=a.record(frame_report())
        out=a.patch_record(dict(base_record_hash=fingerprint(draft['recognition']),load_changes=[
            dict(label='q2',interval=[.5,2.5],raw_description='分布区间更新',evidence='用户给定')]))
        self.assertEqual(out['recognition']['loads'][2]['interval'],[.5,2.5])
        self.assertEqual(out['recognition']['loads'][2]['components'],draft['recognition']['loads'][2]['components'])

    def test_complete_point_replacements_accept_missing_and_empty_intervals(self):
        for empty in (True,False):
            a=FrameAgent(None);draft=a.record(frame_report())
            items=deepcopy(draft['recognition']['loads'])
            for item in (items[0],items[3]):
                item.update(position=1,source='user',evidence='用户明确作用位置')
                if empty:item['interval']=[]
                else:item.pop('interval')
            items[1].update(components=[0,0,0,0,0,5],source='user',evidence='用户明确顺时针')
            out=a.patch_record(dict(base_record_hash=fingerprint(draft['recognition']),loads=items,questions=[],
                facts=[dict(path='/nodes/1/restrained',value=[True]*3,state='user_given',source='user',evidence='用户给定')]))
            self.assertEqual(out['status'],'awaiting_confirmation',out.get('blockers'))

    def test_partial_schema_and_no_physical_change_rejected(self):
        schema=tools_schema(True)[0]['function']['parameters']['$defs']
        self.assertNotIn('components',schema['LoadChange']['required'])
        self.assertNotIn('interval',schema['RecognizedLoad']['required'])
        self.assertIn('position',schema['LoadChange']['properties'])
        a=FrameAgent(None);draft=a.record(frame_report())
        out=a.patch_record(dict(base_record_hash=fingerprint(draft['recognition']),load_changes=[
            dict(label='P1',raw_description='已修改',evidence='用户明确位置')]))
        self.assertEqual(out['status'],'invalid')


if __name__=='__main__':unittest.main()
