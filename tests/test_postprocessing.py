"""杆内变形的独立解析解与可视化数据来源检查。"""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np
from src.solver import solve_space_frame
from src.solver.postprocessing import sample_frame_results
from src.visualization import build_frame_payload
from .test_member_loads import beam
from .helpers import cantilever_data


class PostprocessingTests(unittest.TestCase):
    def test_fixed_uniform_beam_bends_inside(self):
        model=beam([0.,-6.,0.],interval=[0.,2.])
        model['nodes'][1]['restrained']=[True]*6
        result=solve_space_frame(model).as_dict()
        samples=sample_frame_results(model,result,5)[0]['samples']
        for sample in samples:
            x=sample['position']
            expected=-6*x*x*(2-x)**2/(24*8000)
            self.assertAlmostEqual(sample['displacement'][1],expected,places=14)
        np.testing.assert_allclose(samples[0]['displacement'],0,atol=1e-14)
        np.testing.assert_allclose(samples[-1]['displacement'],0,atol=1e-14)
        self.assertLess(min(row['displacement'][1] for row in samples),0)

    def test_cantilever_uniform_curve(self):
        model=beam([0.,-6.,0.],interval=[0.,2.])
        result=solve_space_frame(model).as_dict()
        for sample in sample_frame_results(model,result,7)[0]['samples']:
            x=sample['position']
            expected=-6*x*x*(6*2**2-4*2*x+x*x)/(24*8000)
            slope=-6*x*(3*2**2-3*2*x+x*x)/(6*8000)
            self.assertAlmostEqual(sample['displacement'][1],expected,places=14)
            self.assertAlmostEqual(sample['rotation_course'][2],-slope,places=14)

    def test_point_force_and_moment_curves(self):
        model=beam([0.,5.,0.],moment=[0.,0.,3.])
        result=solve_space_frame(model).as_dict()
        for sample in sample_frame_results(model,result,9)[0]['samples']:
            x=sample['position']
            if x<=1:
                expected=5*x*x*(3-x)/(6*8000)-3*x*x/(2*8000)
            else:
                expected=5*(3*x-1)/(6*8000)-3*(x-.5)/8000
            self.assertAlmostEqual(sample['displacement'][1],expected,places=14)

    def test_axial_and_torsion(self):
        model=beam([7.,0.,0.],moment=[9.,0.,0.])
        result=solve_space_frame(model).as_dict()
        for sample in sample_frame_results(model,result,7)[0]['samples']:
            x=min(sample['position'],1.)
            self.assertAlmostEqual(sample['displacement'][0],7*x/2e6,places=14)
            self.assertAlmostEqual(sample['rotation_course'][0],9*x/800,places=14)

    def test_oblique_and_closed_frame_endpoints(self):
        root=Path(__file__).resolve().parents[1]
        for name in ('bent_frame_3d.json','closed_frame_3d.json','textbook_iv1_native.json','textbook_iv2_native.json'):
            with self.subTest(case=name):
                model=json.loads((root/'examples'/name).read_text(encoding='utf-8'))
                result=solve_space_frame(model).as_dict()
                answers={n['id']:n for n in result['nodes']}
                for curve in sample_frame_results(model,result,7):
                    np.testing.assert_allclose(curve['samples'][0]['displacement'],answers[curve['node_i']]['displacement'][:3],atol=1e-11)
                    np.testing.assert_allclose(curve['samples'][-1]['displacement'],answers[curve['node_j']]['displacement'][:3],atol=1e-11)
                    np.testing.assert_allclose(curve['samples'][-1]['rotation_course'],answers[curve['node_j']]['displacement'][3:],atol=1e-11)

    def test_force_jump_is_retained(self):
        model=beam([0.,5.,0.])
        result=solve_space_frame(model).as_dict()
        rows=[s for s in sample_frame_results(model,result,3)[0]['samples'] if s['position']==1.]
        self.assertEqual([r['side'] for r in rows],['left','right'])
        self.assertAlmostEqual(rows[1]['forces'][1]-rows[0]['forces'][1],5)

    def test_hash_and_endpoint_mismatch_rejected(self):
        model=cantilever_data();result=solve_space_frame(model).as_dict()
        different=deepcopy(model);different['nodes'][1]['coordinates'][0]=3.
        with self.assertRaises(ValueError):sample_frame_results(different,result)
        bad=deepcopy(result);bad['elements'][0]['end_i'][0]+=1e6
        with self.assertRaises(ValueError):sample_frame_results(model,bad)
        for count in (2,302,True):
            with self.assertRaises(ValueError):sample_frame_results(model,result,count)

    def test_payload_preserves_results_and_supports_preview(self):
        model=cantilever_data();result=solve_space_frame(model).as_dict();before=deepcopy(result)
        data=build_frame_payload(model,result)
        self.assertEqual(result,before)
        self.assertEqual(data['model_hash'],result['model_hash'])
        self.assertEqual([n['id'] for n in data['nodes']],[10,90])
        self.assertTrue(data['solved'])
        self.assertEqual(data['nodes'][1]['displacement'][3:],[-v for v in result['nodes'][1]['displacement'][3:]])
        preview=build_frame_payload(model)
        self.assertFalse(preview['solved'])
        self.assertTrue(all(n['displacement'] is None for n in preview['nodes']))
        self.assertTrue(all(e['curve'] is None for e in preview['elements']))
        self.assertTrue(build_frame_payload(model,solve_space_frame(model))['solved'])
