import copy
import unittest
import numpy as np

from src.model import StructuralModel
from src.solver import SolverError, solve_space_frame
from src.solver.assembly import digitize, dor, i0j0
from .helpers import cantilever_data, cantilever_reference, serial_tip_flexibility


class AssemblyTests(unittest.TestCase):
    def test_arrays_and_course_numbering(self):
        data = cantilever_data()
        data['nodes'].reverse()
        arrays = digitize(StructuralModel.model_validate(data))
        self.assertEqual((arrays.n,arrays.nc,arrays.m,arrays.nn),(2,1,1,6))
        self.assertEqual(arrays.node_ids,[10,90])
        np.testing.assert_array_equal(arrays.ihl,[1])
        np.testing.assert_array_equal(arrays.ihr,[2])
        i0,j0 = i0j0(arrays,1)
        np.testing.assert_array_equal(i0,[-1]*6)
        np.testing.assert_array_equal(j0,np.arange(6))
        K,r,dp,free,elements = dor(arrays)
        self.assertEqual(K.shape,(12,12))
        self.assertEqual(r.shape,(6,6))
        np.testing.assert_allclose(K,elements[0].global_matrix)
        np.testing.assert_allclose(r,elements[0].ce)
        np.testing.assert_allclose(r,K[np.ix_(free,free)])
        np.testing.assert_array_equal(dp,data['loads'][0]['components'])

    def test_partial_constraints(self):
        data = cantilever_data()
        data['nodes'][1]['restrained'] = [False,False,True,True,True,False]
        arrays = digitize(StructuralModel.model_validate(data))
        _,j0 = i0j0(arrays,1)
        np.testing.assert_array_equal(j0,[0,1,-1,-1,-1,2])
        self.assertEqual(arrays.nn,3)
        K,r,_,free,_ = dor(arrays)
        np.testing.assert_allclose(r,K[np.ix_(free,free)])


class AnalysisTests(unittest.TestCase):
    def assert_balanced(self,result):
        np.testing.assert_allclose(result.force_balance,0,atol=1e-8)
        np.testing.assert_allclose(result.moment_balance,0,atol=1e-8)
        np.testing.assert_allclose(result.free_residual,0,atol=1e-8)
        self.assertLess(result.backward_error,1e-12)

    def test_cantilever_each_load_and_combined(self):
        cases = list(np.eye(6)) + [np.array([3.,-5.,7.,2.,-4.,6.])]
        for load in cases:
            with self.subTest(load=load.tolist()):
                data = cantilever_data()
                data['loads'][0]['components'] = load.tolist()
                result = solve_space_frame(data)
                u, reaction = cantilever_reference(load)
                np.testing.assert_allclose(result.u[6:],u,atol=1e-13,rtol=1e-11)
                np.testing.assert_allclose(result.reactions[:6],reaction,atol=1e-9)
                np.testing.assert_allclose(result.h[0,:6],reaction,atol=1e-9)
                np.testing.assert_allclose(result.h[0,6:],load,atol=1e-9)
                self.assert_balanced(result)

    def test_oblique_cantilever(self):
        q,_ = np.linalg.qr(np.array([[1.,2.,3.],[4.,1.,2.],[2.,3.,1.]]))
        if np.linalg.det(q)<0:
            q[:,2] *= -1
        data = cantilever_data()
        data['nodes'][1]['coordinates'] = (2*q[:,0]).tolist()
        data['elements'][0]['local_y'] = q[:,1].tolist()
        local_load = np.array(data['loads'][0]['components'])
        data['loads'][0]['components'] = np.r_[q@local_load[:3],q@local_load[3:]].tolist()
        result = solve_space_frame(data)
        ul,rl = cantilever_reference(local_load)
        np.testing.assert_allclose(result.u[6:],np.r_[q@ul[:3],q@ul[3:]],rtol=1e-10,atol=1e-13)
        np.testing.assert_allclose(result.reactions[:6],np.r_[q@rl[:3],q@rl[3:]],atol=1e-9)
        np.testing.assert_allclose(result.h[0,6:],local_load,atol=1e-9)
        self.assert_balanced(result)

    def test_serial_spatial_bent_frame_independent_force_method(self):
        data = cantilever_data()
        data['nodes'].append({'id': 300, 'coordinates': [2.,1.,1.]})
        data['elements'].append({'id': 200, 'node_i': 90, 'node_j': 300,
                                 'material_id':1,'section_id':1,'local_y':[0.,1.,0.]})
        data['loads'][0]['node_id'] = 300
        model = StructuralModel.model_validate(data)
        reference = serial_tip_flexibility(model) @ np.array(data['loads'][0]['components'])
        result = solve_space_frame(model)
        np.testing.assert_allclose(result.u[-6:],reference,rtol=1e-10,atol=1e-12)
        # 独立静力截面平衡，检查每杆两端的全部 12 项内力。
        tip = np.array(model.nodes[-1].coordinates)
        nodes = {node.id: np.array(node.coordinates) for node in model.nodes}
        load = np.array(data['loads'][0]['components'])
        for index, element in enumerate(model.elements):
            xi, xj = nodes[element.node_i], nodes[element.node_j]
            ex = (xj-xi)/np.linalg.norm(xj-xi)
            ey = np.array(element.local_y)-ex*np.dot(ex,element.local_y)
            ey /= np.linalg.norm(ey)
            q = np.column_stack((ex,ey,np.cross(ex,ey)))
            fi, fj = -load[:3], load[:3]
            mi = np.cross(tip-xi,load[:3])-load[3:]
            mj = load[3:]-np.cross(tip-xj,load[:3])
            expected = np.r_[q.T@fi,q.T@mi,q.T@fj,q.T@mj]
            np.testing.assert_allclose(result.h[index],expected,atol=1e-8,rtol=1e-10)
        self.assert_balanced(result)

    def test_redundant_tip_support_force_method(self):
        data = cantilever_data()
        data['nodes'].append({'id':300,'coordinates':[2.,1.,1.],
                              'restrained':[False,True,False,False,False,False]})
        data['elements'].append({'id':200,'node_i':90,'node_j':300,'material_id':1,
                                 'section_id':1,'local_y':[0.,1.,0.]})
        data['loads'][0]['node_id']=300
        model = StructuralModel.model_validate(data)
        c = serial_tip_flexibility(model)
        load = np.array(data['loads'][0]['components'])
        redundant = -(c@load)[1]/c[1,1]
        combined = load.copy()
        combined[1] += redundant
        expected = c@combined
        result = solve_space_frame(model)
        np.testing.assert_allclose(result.u[-6:],expected,rtol=1e-9,atol=1e-12)
        self.assertAlmostEqual(result.reactions[-5],redundant,places=8)
        self.assert_balanced(result)

    def test_planar_reduction(self):
        data = cantilever_data()
        load = np.array([3.,-5.,0.,0.,0.,6.])
        data['nodes'][1]['restrained'] = [False,False,True,True,True,False]
        data['loads'][0]['components'] = load.tolist()
        result = solve_space_frame(data)
        u,r = cantilever_reference(load)
        np.testing.assert_allclose(result.u[6:],u,atol=1e-13)
        np.testing.assert_allclose(result.reactions[:6],r,atol=1e-9)
        self.assert_balanced(result)

    def test_split_member(self):
        data = cantilever_data()
        data['nodes'].insert(1,{'id': 50,'coordinates':[1.,0.,0.]})
        data['elements'][0]['node_j']=50
        data['elements'].append({'id': 200,'node_i':50,'node_j':90,'material_id':1,
                                'section_id':1,'local_y':[0.,1.,0.]})
        result = solve_space_frame(data)
        expected,_ = cantilever_reference()
        np.testing.assert_allclose(result.u[-6:],expected,atol=1e-12)
        self.assert_balanced(result)

    def test_reversed_element_and_hash(self):
        data = cantilever_data()
        baseline = solve_space_frame(data)
        data['elements'][0].update(node_i=90,node_j=10)
        reversed_result = solve_space_frame(data)
        np.testing.assert_allclose(reversed_result.u,baseline.u,atol=1e-13)
        np.testing.assert_allclose(reversed_result.reactions,baseline.reactions,atol=1e-9)
        self.assertNotEqual(reversed_result.model_hash,baseline.model_hash)
        # 端点反转后 ex/ez 翻转，局部力对应符号必须转换。
        signs = np.array([-1,1,-1,-1,1,-1])
        np.testing.assert_allclose(reversed_result.h[0,:6],signs*baseline.h[0,6:],atol=1e-9)
        self.assert_balanced(reversed_result)

    def test_support_load_subtracted_and_repeated_loads_sum(self):
        data = cantilever_data()
        data['loads'].append(copy.deepcopy(data['loads'][0]))
        support = np.array([2.,3.,4.,5.,6.,7.])
        data['loads'].append({'node_id':10,'components':support.tolist()})
        result = solve_space_frame(data)
        u,r = cantilever_reference()
        np.testing.assert_allclose(result.u[6:],2*u,atol=1e-13)
        np.testing.assert_allclose(result.reactions[:6],2*r-support,atol=1e-9)
        self.assert_balanced(result)

    def test_all_fixed(self):
        data = cantilever_data()
        data['nodes'][1]['restrained']=[True]*6
        result = solve_space_frame(data)
        np.testing.assert_array_equal(result.u,np.zeros(12))
        np.testing.assert_array_equal(result.reactions,-result.arrays.p)
        np.testing.assert_array_equal(result.h,np.zeros((1,12)))
        self.assert_balanced(result)

    def test_unrestrained_mechanism_diagnostic(self):
        data = cantilever_data()
        data['nodes'][0]['restrained']=[False]*6
        with self.assertRaisesRegex(SolverError,'节点'):
            solve_space_frame(data)

    def test_zero_load_and_serialization(self):
        data = cantilever_data()
        data['loads']=[]
        result = solve_space_frame(data)
        np.testing.assert_array_equal(result.u,np.zeros(12))
        output = result.as_dict(diagnostics=True)
        self.assertEqual(output['nodes'][0]['id'],10)
        self.assertEqual(output['units']['rotation'],'rad')
        self.assertEqual(len(output['element_matrices'][0]['ca']),6)
        self.assertEqual(len(output['model_hash']),64)
        self.assert_balanced(result)
