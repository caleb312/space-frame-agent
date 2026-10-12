"""2014理论 + 独立解析解；真实原生自由度，不靠空间刚架输出自证。"""
from copy import deepcopy
import unittest
import numpy as np
from pydantic import ValidationError
from src.model import parse_model
from src.solver import solve_structure, solve_plane_truss, solve_plane_frame, solve_space_truss, SolverError
from src.solver.specialized import stif_plane_frame,stif_truss


def model(kind,coordinates,fixities,loads,connections=None):
    return {'structure_type':kind,'units':'kN_m','nodes':[
        {'id':10*(i+1),'coordinates':c,'restrained':fixities[i]} for i,c in enumerate(coordinates)],
        'materials':[{'id':1,'E':200000000.0}],'sections':[{'id':1,'A':.01,**({'I':.0001} if kind=='plane_frame' else {})}],
        'elements':[{'id':100+i,'node_i':10*a,'node_j':10*b,'material_id':1,'section_id':1}
                    for i,(a,b) in enumerate(connections or [(1,2)])],
        'loads':[{'node_id':10*i,'components':v} for i,v in loads]}


class SpecializedSolverTests(unittest.TestCase):
    def check_balance(self,result):
        for name in ('force_balance','right_hand_moment_balance'):
            np.testing.assert_allclose(result.as_dict()['checks'][name],0,atol=1e-10,rtol=0)

    def test_axial_bar_plane_and_reverse(self):
        m=model('plane_truss',[[0,0],[2,0]],[[True,True],[False,True]],[(2,[4,0])])
        r=solve_plane_truss(m);self.assertEqual(r.K.shape,(4,4))
        self.assertAlmostEqual(r.u[2],4e-6,places=14)
        self.assertAlmostEqual(r.as_dict()['elements'][0]['axial_force'],4)
        np.testing.assert_allclose(r.reactions,[-4,0,0,0],atol=1e-12)
        self.check_balance(r)
        m['elements'][0].update(node_i=20,node_j=10)
        rr=solve_structure(m);np.testing.assert_allclose(rr.u,r.u,atol=1e-15)
        self.assertAlmostEqual(rr.as_dict()['elements'][0]['axial_force'],4)

    def test_triangle_independent_equilibrium_and_compatibility(self):
        m=model('plane_truss',[[0,0],[2,0],[1,1]],[[True,True],[False,True],[False,False]],[(3,[0,-10])],[(1,2),(1,3),(2,3)])
        r=solve_structure(m);out=r.as_dict()
        np.testing.assert_allclose([e['axial_force'] for e in out['elements']],[5,-5*np.sqrt(2),-5*np.sqrt(2)],atol=1e-12)
        np.testing.assert_allclose(r.u,[0,0,10/2e6,0,5/2e6,-(10*np.sqrt(2)+5)/2e6],atol=1e-14)
        np.testing.assert_allclose(r.reactions[[1,3]],[5,5],atol=1e-12)
        self.check_balance(r)

    def test_space_three_bar_node(self):
        m=model('space_truss',[[1,0,0],[0,1,0],[0,0,1],[0,0,0]],[[True]*3]*3+[[False]*3],[(4,[3,-4,5])],[(1,4),(2,4),(3,4)])
        r=solve_space_truss(m);self.assertEqual(r.K.shape,(12,12))
        np.testing.assert_allclose(r.u[-3:],np.array([3,-4,5])/2e6,atol=1e-15)
        np.testing.assert_allclose([e['axial_force'] for e in r.as_dict()['elements']],[-3,4,-5],atol=1e-12)
        self.check_balance(r)

    def test_space_diagonal_nonzero_z_direction(self):
        m=model('space_truss',[[0,0,0],[1,1,1]],[[True]*3,[True,True,False]],[(2,[0,0,2])])
        r=solve_structure(m);self.assertAlmostEqual(r.u[-1],6*np.sqrt(3)/2e6,places=14)
        self.check_balance(r)

    def test_plane_frame_tip_force(self):
        m=model('plane_frame',[[0,0],[2,0]],[[True]*3,[False]*3],[(2,[0,-10,0])])
        r=solve_plane_frame(m);self.assertEqual(r.K.shape,(6,6))
        np.testing.assert_allclose(r.u[-3:],[0,-10*2**3/(3*20000),10*2**2/(2*20000)],atol=1e-14)
        np.testing.assert_allclose(r.reactions[:3],[0,10,-20],atol=1e-12)
        self.check_balance(r)

    def test_plane_frame_axial_and_clockwise_tip_moment(self):
        m=model('plane_frame',[[0,0],[2,0]],[[True]*3,[False]*3],[(2,[4,0,6])])
        r=solve_structure(m)
        np.testing.assert_allclose(r.u[-3:],[4e-6,-6*4/(2*20000),6*2/20000],atol=1e-14)
        self.check_balance(r)

    def test_plane_frame_uniform_load_and_section(self):
        m=model('plane_frame',[[0,0],[2,0]],[[True]*3,[False]*3],[])
        m['member_loads']=[{'element_id':100,'kind':'uniform_force','coordinate_system':'global','components':[0,-2,0],'interval':[0,2]}]
        m['section_queries']=[{'id':1,'element_id':100,'location':'position','position':1,'side':'right'}]
        r=solve_structure(m)
        np.testing.assert_allclose(r.u[-3:],[0,-2*2**4/(8*20000),2*2**3/(6*20000)],atol=1e-14)
        np.testing.assert_allclose(r.reactions[:3],[0,4,-4],atol=1e-12)
        self.assertAlmostEqual(abs(r.as_dict()['sections'][0]['forces'][5]),1)
        self.check_balance(r)

    def test_plane_frame_point_member_force(self):
        m=model('plane_frame',[[0,0],[2,0]],[[True]*3,[False]*3],[])
        m['member_loads']=[{'element_id':100,'kind':'point_force','coordinate_system':'local','components':[0,-10,0],'position':1}]
        r=solve_structure(m)
        np.testing.assert_allclose(r.u[-3:],[0,-10*(6-1)/(6*20000),10/(2*20000)],atol=1e-14)
        self.check_balance(r)

    def test_inclined_frame_local_load_and_reversed_ends(self):
        # 3-4-5方向的长2m悬臂，局部横向力。解析局部uy和顺时针rz。
        n=np.array([.6,.8]);ey=np.array([-.8,.6])
        m=model('plane_frame',[[0,0],(2*n).tolist()],[[True]*3,[False]*3],[])
        m['member_loads']=[{'element_id':100,'kind':'point_force',
            'coordinate_system':'local','components':[0,-10,0],'position':2}]
        r=solve_structure(m)
        np.testing.assert_allclose(r.u[-3:],np.r_[ey*(-10*8/(3*20000)),.001],atol=1e-14)
        self.check_balance(r)
        # 改为同一个全局力并反向编号，不改变物理问题及节点位移。
        m['elements'][0].update(node_i=20,node_j=10)
        m['member_loads'][0].update(coordinate_system='global',
            components=[*(ey*-10),0],position=0)
        rr=solve_structure(m)
        np.testing.assert_allclose(rr.u,r.u,atol=1e-14)
        self.check_balance(rr)

    def test_frame_member_moment_and_partial_uniform(self):
        m=model('plane_frame',[[0,0],[2,0]],[[True]*3,[False]*3],[])
        m['member_loads']=[{'element_id':100,'kind':'point_moment',
            'coordinate_system':'global','components':[0,0,6],'position':1}]
        r=solve_structure(m)
        # M施于a=1，弯矩仅在[0,a]：u(L)=-M*a*(L-a/2)/EI。
        np.testing.assert_allclose(r.u[-3:],[0,-6*1.5/20000,6/20000],atol=1e-14)
        self.check_balance(r)
        m['member_loads']=[{'element_id':100,'kind':'uniform_force',
            'coordinate_system':'global','components':[0,-2,0],'interval':[.5,1.5]}]
        r=solve_structure(m)
        # 单位集中力影响函数积分：q*s²*(3L-s)/(6EI)与q*s²/(2EI)。
        a,b=.5,1.5;L=2;EI=20000;q=-2
        expected_y=q*((L*b**3-b**4/4)-(L*a**3-a**4/4))/(6*EI)
        expected_theta=-q*(b**3-a**3)/(6*EI)
        np.testing.assert_allclose(r.u[-3:],[0,expected_y,expected_theta],atol=1e-14)
        self.check_balance(r)

    def test_matrices_symmetric_and_dimensions(self):
        for matrix in (stif_plane_frame(100,2,3),stif_truss(100,3,np.array([.6,.8])),stif_truss(100,3,np.array([1.,0.,0.]))):
            np.testing.assert_allclose(matrix,matrix.T,atol=1e-14)
            self.assertIn(matrix.shape,[(4,4),(6,6)])

    def test_invalid_native_input_and_solver_mismatch(self):
        m=model('plane_truss',[[0,0],[1,0]],[[True]*2,[False,True]],[(2,[1,0])])
        wrong=deepcopy(m);wrong['loads'][0]['components']=[1,0,2]
        with self.assertRaises(ValidationError):parse_model(wrong)
        wrong=deepcopy(m);wrong['elements'][0]['node_j']=99
        with self.assertRaises(ValidationError):parse_model(wrong)
        wrong=deepcopy(m);wrong['nodes'][1]['coordinates']=[0,0]
        with self.assertRaises(ValidationError):parse_model(wrong)
        wrong=deepcopy(m);wrong['member_loads']=[{'element_id':100,'kind':'point_force','coordinate_system':'global','components':[0,1,0],'position':.5}]
        with self.assertRaises(ValidationError):parse_model(wrong)
        with self.assertRaises(SolverError):solve_plane_frame(m)

    def test_mechanism_diagnosed_without_virtual_constraints(self):
        m=model('plane_truss',[[0,0],[1,0]],[[True]*2,[False]*2],[(2,[1,0])])
        with self.assertRaisesRegex(SolverError,'节点20/uy'):solve_structure(m)


if __name__=='__main__':unittest.main()
