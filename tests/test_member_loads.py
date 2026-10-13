"""跨内荷载：独立静力/解析解、参考资料原题及旧验证适配器对照。"""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np
from pydantic import ValidationError
from src.solver import solve_space_frame
from src.solver.member_loads import LocalLoad, consistent_vector, point_vector
from src.agent.runtime import fingerprint
from .test_parameters import agent, inputs, recognition_report
from examples.native_textbook import native_textbook_model
from examples.textbook_reference import (equivalent_model, original_end_forces_right_hand,
                                         section_forces_in_appendix_convention)
from .helpers import cantilever_data

ROOT = Path(__file__).resolve().parents[1]


def beam(force=None, moment=None, interval=None):
    model = cantilever_data()
    model['loads'] = []
    model['member_loads'] = []
    if force is not None:
        model['member_loads'].append({'element_id': 100, 'kind': 'uniform_force' if interval else 'point_force',
            'coordinate_system': 'local', 'components': force,
            **({'interval': interval} if interval else {'position': 1.})})
    if moment is not None:
        model['member_loads'].append({'element_id': 100, 'kind': 'point_moment',
                                    'coordinate_system': 'local', 'components': moment, 'position': 1.})
    return model


def at(model, x, side='right', query_id=1):
    model['section_queries'] = [{'id': query_id, 'element_id': 100,
                                 'location': 'position', 'position': x, 'side': side}]
    return model


class MemberLoadTests(unittest.TestCase):
    def test_all_point_resultants(self):
        for x in (0., .4, 1., 2.):
            for component in range(6):
                vector = np.eye(6)[component]*7
                p = point_vector(2., x, force=vector[:3], moment=vector[3:])
                force = p[:3]+p[6:9]
                moment_rh = -p[3:6]-p[9:]+np.cross([2.,0,0],p[6:9])
                np.testing.assert_allclose(force,vector[:3],atol=1e-13)
                np.testing.assert_allclose(moment_rh,np.cross([x,0,0],vector[:3])-vector[3:],atol=1e-13)

    def test_uniform_resultants_and_virtual_work(self):
        load = LocalLoad(100,'uniform_force',np.array([3.,-4.,5.]),None,(.3,1.6))
        p = consistent_vector(2., load)
        force = p[:3]+p[6:9]
        moment = -p[3:6]-p[9:]+np.cross([2.,0,0],p[6:9])
        np.testing.assert_allclose(force,load.components*1.3,atol=1e-13)
        np.testing.assert_allclose(moment,np.cross([.95,0,0],force),atol=1e-13)
        # 独立多项式场：u=x, v=x², w=x³；课件 rz=-v'，ry=w'。
        displacement = np.array([0.,0.,0.,0.,0.,0.,2.,4.,8.,0.,12.,-4.])
        a,b = load.interval
        exact = 3*(b*b-a*a)/2 - 4*(b**3-a**3)/3 + 5*(b**4-a**4)/4
        self.assertAlmostEqual(float(p@displacement),exact,places=12)

    def test_fully_fixed_uniform_end_forces(self):
        model = beam([0.,-6.,0.], interval=[0.,2.])
        model['nodes'][1]['restrained'] = [True]*6
        result = solve_space_frame(model)
        # 参考资料固定端公式，q_y=-6、L=2：每端竖向反力6，课件力矩 i=-2、j=+2。
        expected = np.array([0.,6.,0.,0.,0.,-2.,0.,6.,0.,0.,0.,2.])
        np.testing.assert_allclose(result.h[0], expected, atol=1e-13)
        np.testing.assert_allclose(result.u,0,atol=0)

    def test_cantilever_uniform_analytic(self):
        model = at(beam([0.,-6.,0.], interval=[0.,2.]),1.)
        result = solve_space_frame(model)
        tip = result.u.reshape(-1,6)[1]
        # 独立梁微分方程：v(L)=qL⁴/(8EIz)，rz(L)=-qL³/(6EIz)。
        self.assertAlmostEqual(tip[1],-6*2**4/(8*8000),places=14)
        self.assertAlmostEqual(tip[5],6*2**3/(6*8000),places=14)
        np.testing.assert_allclose(result.reactions[:6],[0.,12.,0.,0.,0.,-12.],atol=1e-11)
        np.testing.assert_allclose(result.h[0,6:],0,atol=1e-11)
        # 右侧剩余向下荷载使局部截面 Mz 为负，不能套用另一组弯矩正号。
        np.testing.assert_allclose(result.sections[0]['forces'],[0.,6.,0.,0.,0.,-3.],atol=1e-11)

    def test_partial_uniform_analytic(self):
        # q 从 a 到 b：由自由端挠度的单位荷载积分得到。
        a,b,L,q,EI = .5,1.5,2.,4.,4000.
        result = solve_space_frame(at(beam([0.,0.,q],interval=[a,b]),1.))
        tip = result.u.reshape(-1,6)[1]
        primitive = lambda x: L*x**3/6-x**4/24
        expected = q*(primitive(b)-primitive(a))/EI
        self.assertAlmostEqual(tip[2],expected,places=14)
        self.assertAlmostEqual(tip[4],q*(b**3-a**3)/(6*EI),places=14)
        # 右侧向上荷载的物理 My 为负；参考资料 My 与该物理分量反号。
        np.testing.assert_allclose(result.sections[0]['forces'],[0.,0.,-2.,0.,.5,0.],atol=1e-11)

    def test_point_force_section_jump(self):
        model = beam([0.,5.,0.])
        model['section_queries'] = [
            {'id':1,'element_id':100,'location':'position','position':1.,'side':'left'},
            {'id':2,'element_id':100,'location':'position','position':1.,'side':'right'}]
        result = solve_space_frame(model)
        np.testing.assert_allclose(result.sections[0]['forces'],[0.,-5.,0.,0.,0.,0.],atol=1e-11)
        np.testing.assert_allclose(result.sections[1]['forces'],0,atol=1e-11)
        tip=result.u.reshape(-1,6)[1]
        self.assertAlmostEqual(tip[1],5*(3*2-1)/(6*8000),places=14)

    def test_point_moment_section_jump(self):
        model = beam(moment=[0.,0.,3.])
        model['section_queries'] = [
            {'id':1,'element_id':100,'location':'position','position':1.,'side':'left'},
            {'id':2,'element_id':100,'location':'position','position':1.,'side':'right'}]
        result=solve_space_frame(model)
        np.testing.assert_allclose(result.sections[0]['forces'],[0.,0.,0.,0.,0.,-3.],atol=1e-11)
        np.testing.assert_allclose(result.sections[1]['forces'],0,atol=1e-11)
        self.assertAlmostEqual(result.u[7],-3*(2-.5)/8000,places=14)
        self.assertAlmostEqual(result.u[11],3/8000,places=14)

    def test_axial_load_and_torsion(self):
        model=at(beam([7.,0.,0.],moment=[9.,0.,0.]),.5)
        result=solve_space_frame(model)
        self.assertAlmostEqual(result.u[6],7/(2e6),places=14)
        self.assertAlmostEqual(result.u[9],9/800,places=14)
        np.testing.assert_allclose(result.sections[0]['forces'],[7.,0.,0.,-9.,0.,0.],atol=1e-11)

    def test_global_and_local_loads_agree(self):
        data=beam([3.,4.,5.],moment=[6.,7.,8.],interval=[.2,1.7])
        local=solve_space_frame(data)
        data['nodes'][1]['coordinates']=[0.,2.,0.]
        data['elements'][0]['local_y']=[0.,0.,1.]
        rotated_local=solve_space_frame(data)
        q=rotated_local.elements[0].t[:3,:3]
        global_data=deepcopy(data)
        for load in global_data['member_loads']:
            load['coordinate_system']='global'
            load['components']=(q@load['components']).tolist()
        global_result=solve_space_frame(global_data)
        np.testing.assert_allclose(global_result.u,rotated_local.u,atol=1e-12)
        np.testing.assert_allclose(global_result.h,local.h,atol=1e-10)

    def test_midspan_is_computed(self):
        model=beam([1.,2.,3.])
        model['section_queries']=[{'id':7,'element_id':100,'location':'midspan','side':'left'}]
        self.assertEqual(solve_space_frame(model).sections[0]['position'],1.)

    def test_invalid_loads_and_queries(self):
        changes = [lambda d:d['member_loads'][0].update(element_id=999),
                   lambda d:d['member_loads'][0].update(position=-1),
                   lambda d:d['member_loads'][0].update(position=3),
                   lambda d:d['member_loads'][0].update(interval=[0.,1.]),
                   lambda d:d['member_loads'][0].update(components=[1.,2.]),
                   lambda d:d['member_loads'][0].update(coordinate_system='screen'),
                   lambda d:d['section_queries'][0].update(element_id=999),
                   lambda d:d['section_queries'][0].update(position=3),
                   lambda d:d['section_queries'][0].update(location='midspan'),
                   lambda d:d['section_queries'].append(deepcopy(d['section_queries'][0]))]
        for change in changes:
            model=at(beam([0.,1.,0.]),.5)
            change(model)
            with self.assertRaises(ValidationError): solve_space_frame(model)
        for interval in ([1.,1.],[-1.,1.],[1.,.5],[0.,3.]):
            with self.assertRaises(ValidationError): solve_space_frame(beam([0.,1.,0.],interval=interval))

    def test_native_textbook_cases(self):
        for name in ('iv1','iv2'):
            with self.subTest(case=name):
                raw=json.loads((ROOT/f'examples/textbook_{name}_original.json').read_text(encoding='utf-8'))
                result=solve_space_frame(native_textbook_model(raw))
                legacy_model,equivalent,_=equivalent_model(raw)
                legacy=solve_space_frame(legacy_model)
                np.testing.assert_allclose(result.u,legacy.u,atol=1e-12)
                np.testing.assert_allclose(result.reactions,legacy.reactions,atol=1e-9)
                rh=result.h*np.tile([1,1,1,-1,-1,-1],2)
                np.testing.assert_allclose(rh,original_end_forces_right_hand(legacy,equivalent),atol=1e-9)
                sections=np.array([query['forces'] for query in result.sections])
                np.testing.assert_allclose(sections,section_forces_in_appendix_convention(raw,legacy,equivalent),atol=1e-9)
                np.testing.assert_allclose(sections,raw['reference_section_forces'],atol=1e-3,rtol=0)
                displacements=result.u.reshape(-1,6)[raw['fixed_nodes']:].copy()
                displacements[:,3:]*=-1
                ref=np.array(raw.get('reference_displacements_corrected',raw['reference_displacements']))
                np.testing.assert_allclose(displacements[:,:3],ref[:,:3],atol=raw['displacement_atol'],rtol=0)
                np.testing.assert_allclose(displacements[:,3:],ref[:,3:],atol=raw['rotation_atol'],rtol=0)
                np.testing.assert_allclose(result.force_balance,0,atol=1e-8)
                np.testing.assert_allclose(result.moment_balance,0,atol=1e-8)

    def test_uniform_subinterval_superposition(self):
        model=at(beam([2.,-3.,4.],interval=[.2,1.7]),.7)
        full=solve_space_frame(model)
        split=deepcopy(model)
        split['member_loads']=[dict(model['member_loads'][0],interval=[.2,.8]),
                               dict(model['member_loads'][0],interval=[.8,1.7])]
        actual=solve_space_frame(split)
        np.testing.assert_allclose(actual.u,full.u,atol=1e-13)
        np.testing.assert_allclose(actual.h,full.h,atol=1e-11)
        np.testing.assert_allclose(actual.sections[0]['forces'],full.sections[0]['forces'],atol=1e-11)

    def test_element_equilibrium_with_original_loads(self):
        raw=json.loads((ROOT/'examples/textbook_iv2_original.json').read_text(encoding='utf-8'))
        model=native_textbook_model(raw)
        result=solve_space_frame(model)
        for index,eid in enumerate(result.arrays.element_ids):
            i,j=result.arrays.ihl[index]-1,result.arrays.ihr[index]-1
            xyz=np.column_stack((result.arrays.x,result.arrays.y,result.arrays.z))
            length=np.linalg.norm(xyz[j]-xyz[i])
            h=result.h[index]
            force=h[:3]+h[6:9]
            moment=-h[3:6]-h[9:]+np.cross([length,0,0],h[6:9])
            for load in model['member_loads']:
                if load['element_id']!=eid: continue
                f=np.array(load['components'])
                if load['kind']=='uniform_force':
                    a,b=load['interval']
                    force+=f*(b-a)
                    moment+=np.cross([(a+b)/2,0,0],f*(b-a))
                elif load['kind']=='point_force':
                    force+=f
                    moment+=np.cross([load['position'],0,0],f)
                else: moment-=f
            np.testing.assert_allclose(force,0,atol=1e-9)
            np.testing.assert_allclose(moment,0,atol=1e-9)

    def test_end_reversal_with_global_loads(self):
        model=at(beam([2.,-3.,4.],moment=[5.,6.,7.],interval=[.2,1.7]),.6)
        for load in model['member_loads']: load['coordinate_system']='global'
        result=solve_space_frame(model)
        reverse=deepcopy(model)
        reverse['elements'][0].update(node_i=90,node_j=10)
        for load in reverse['member_loads']:
            if load['kind']=='uniform_force':
                a,b=load['interval'];load['interval']=[2-b,2-a]
            else: load['position']=2-load['position']
        reverse['section_queries'][0]['position']=1.4
        other=solve_space_frame(reverse)
        np.testing.assert_allclose(other.u,result.u,atol=1e-13)
        np.testing.assert_allclose(other.reactions,result.reactions,atol=1e-10)
        q=result.elements[0].t[:3,:3]; qr=other.elements[0].t[:3,:3]
        np.testing.assert_allclose(qr@other.h[0,:3],q@result.h[0,6:9],atol=1e-10)
        np.testing.assert_allclose(qr@other.h[0,3:6],q@result.h[0,9:],atol=1e-10)
        s=np.array(result.sections[0]['forces']);sr=np.array(other.sections[0]['forces'])
        sign=np.array([1,-1,-1,1,-1,1])
        np.testing.assert_allclose(qr@(sr[:3]*sign[:3]),-q@(s[:3]*sign[:3]),atol=1e-10)
        np.testing.assert_allclose(qr@(sr[3:]*sign[3:]),-q@(s[3:]*sign[3:]),atol=1e-10)

    def test_all_moment_axes_interior(self):
        for axis in range(3):
            moment=np.eye(3)[axis]*3
            model=at(beam(moment=moment.tolist()),.5)
            result=solve_space_frame(model)
            expected=np.zeros(6);expected[3+axis]=3*[-1,1,-1][axis]
            np.testing.assert_allclose(result.sections[0]['forces'],expected,atol=1e-11)
            if axis==1:
                self.assertAlmostEqual(result.u[8],3*1.5/4000,places=14)
                self.assertAlmostEqual(result.u[10],3/4000,places=14)

    def test_native_unit_scaling(self):
        model=at(beam([2.,-3.,4.],moment=[5.,6.,7.],interval=[.2,1.7]),.6)
        base=solve_space_frame(model)
        model['units']='N_m'
        for mat in model['materials']:
            mat['E']*=1000;mat['G']*=1000
        for load in model['member_loads']:
            load['components']=[value*1000 for value in load['components']]
        other=solve_space_frame(model)
        np.testing.assert_allclose(other.u,base.u,atol=1e-13)
        np.testing.assert_allclose(other.h,1000*base.h,atol=1e-8)
        np.testing.assert_allclose(other.sections[0]['forces'],np.array(base.sections[0]['forces'])*1000,atol=1e-8)


class AgentMemberLoadTests(unittest.TestCase):
    def report(self):
        report=recognition_report()
        report['loads']=[{'label':'q','kind':'member_distributed','target_id':100,
                         'components':[0.,-1.,0.,0.,0.,0.],'units':'kN_m','position':None,
                         'interval':[0.,2.],'raw_description':'全跨均布力','source':'user','evidence':'用户给定',
                         'coordinate_system':'global','distribution':'uniform'}]
        report['requested_results']=['displacements','section_forces']
        report['section_queries']=[{'id':1,'element_id':100,'location':'midspan','side':'right',
                                   'source':'user','evidence':'用户要求跨中'}]
        return report

    def test_agent_compiles_and_solves_after_confirmation(self):
        assistant=agent()
        assistant.calculate_parameters(inputs())
        output=assistant.record(self.report())
        self.assertEqual(output['status'],'awaiting_confirmation')
        self.assertEqual(assistant.model['member_loads'][0]['kind'],'uniform_force')
        self.assertEqual(assistant.solve()['status'],'needs_confirmation')
        assistant.confirm(fingerprint(assistant.model))
        result=assistant.solve()
        self.assertEqual(result['status'],'solved')
        np.testing.assert_allclose(result['result']['sections'][0]['forces'],[0.,1.,0.,0.,0.,-.5],atol=1e-10)

    def test_unknown_load_frame_or_distribution_blocks(self):
        assistant=agent();assistant.calculate_parameters(inputs())
        for field,value in [('coordinate_system',None),('distribution',None),('distribution','other'),('interval',None)]:
            report=self.report();report['loads'][0][field]=value
            self.assertEqual(assistant.record(report)['status'],'needs_information')
            self.assertIsNone(assistant.model)

    def test_invalid_position_and_missing_section_targets(self):
        assistant=agent();assistant.calculate_parameters(inputs())
        report=self.report();report['loads'][0]['interval']=[0.,3.]
        self.assertEqual(assistant.record(report)['status'],'needs_information')
        report=self.report();report['section_queries']=[]
        self.assertTrue(any('需要指定截面' in v for v in assistant.record(report)['blockers']))
