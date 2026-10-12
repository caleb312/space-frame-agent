import copy
import json
from pathlib import Path
import unittest

import numpy as np

from src.model import StructuralModel
from src.solver import solve_space_frame
from .energy_reference import energy_frame_reference

PATH = Path(__file__).resolve().parents[1] / 'examples/closed_frame_3d.json'


class ClosedFrameTests(unittest.TestCase):
    def read_model(self):
        return json.loads(PATH.read_text(encoding='utf-8'))

    def compare_reference(self, data):
        model = StructuralModel.model_validate(data)
        K, P, U, R, H = energy_frame_reference(model)
        result = solve_space_frame(model)
        mapping = {node.id: i for i,node in enumerate(model.nodes)}
        permutation = np.concatenate([np.arange(6*mapping[node_id],6*mapping[node_id]+6)
                                       for node_id in result.arrays.node_ids])
        np.testing.assert_allclose(result.K, K[np.ix_(permutation,permutation)],rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(result.arrays.p,P[permutation],atol=1e-14)
        np.testing.assert_allclose(result.u,U[permutation],rtol=1e-9,atol=1e-12)
        np.testing.assert_allclose(result.reactions,R[permutation],rtol=1e-9,atol=1e-8)
        np.testing.assert_allclose(result.h,H,rtol=1e-9,atol=1e-8)
        np.testing.assert_allclose(result.force_balance,0.,atol=1e-8)
        np.testing.assert_allclose(result.moment_balance,0.,atol=1e-8)
        return result

    def test_closed_frame_energy_reference(self):
        result = self.compare_reference(self.read_model())
        self.assertEqual((result.arrays.n,result.arrays.nc,result.arrays.m),(8,4,10))
        self.assertEqual(result.arrays.nn,24)

    def test_relabel_reorder_and_reverse_members(self):
        original = self.read_model()
        baseline = solve_space_frame(original)
        data = copy.deepcopy(original)
        renamed = {node['id']:node['id']*17-250 for node in data['nodes']}
        for node in data['nodes']:
            node['id'] = renamed[node['id']]
        for load in data['loads']:
            load['node_id'] = renamed[load['node_id']]
        for e in data['elements']:
            e['node_i'],e['node_j'] = renamed[e['node_j']],renamed[e['node_i']]
        data['nodes'] = data['nodes'][::-1]
        data['elements'] = data['elements'][::-1]
        result = self.compare_reference(data)
        base_nodes = {node_id:baseline.u[6*i:6*i+6] for i,node_id in enumerate(baseline.arrays.node_ids)}
        reverse_names = {value:key for key,value in renamed.items()}
        for i,node_id in enumerate(result.arrays.node_ids):
            np.testing.assert_allclose(result.u[6*i:6*i+6],base_nodes[reverse_names[node_id]],atol=1e-12)

    def test_global_rotation_and_translation(self):
        original = self.read_model()
        baseline = solve_space_frame(original)
        data = copy.deepcopy(original)
        # 显式非平凡正交旋转，兼顾各方向分量。
        angle = .7
        q1 = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        q2 = np.array([[1,0,0],[0,np.cos(.4),-np.sin(.4)],[0,np.sin(.4),np.cos(.4)]])
        q = q2@q1
        for node in data['nodes']:
            node['coordinates'] = (q@node['coordinates']+np.array([10.,-3.,2.])).tolist()
        for e in data['elements']:
            e['local_y'] = (q@e['local_y']).tolist()
        for load in data['loads']:
            p = np.array(load['components'])
            load['components'] = np.r_[q@p[:3],q@p[3:]].tolist()
        result = self.compare_reference(data)
        for i in range(result.arrays.n):
            u,r = baseline.u[6*i:6*i+6],baseline.reactions[6*i:6*i+6]
            np.testing.assert_allclose(result.u[6*i:6*i+6],np.r_[q@u[:3],q@u[3:]],rtol=1e-9,atol=1e-12)
            np.testing.assert_allclose(result.reactions[6*i:6*i+6],np.r_[q@r[:3],q@r[3:]],rtol=1e-9,atol=1e-8)
        np.testing.assert_allclose(result.h,baseline.h,rtol=1e-9,atol=1e-8)

    def test_newton_vs_kilonewton_units(self):
        original = self.read_model()
        baseline = solve_space_frame(original)
        data = copy.deepcopy(original)
        data['units'] = 'N_m'
        for mat in data['materials']:
            mat['E'] *= 1000
            mat['G'] *= 1000
        for load in data['loads']:
            load['components'] = (np.array(load['components'])*1000).tolist()
        result = self.compare_reference(data)
        np.testing.assert_allclose(result.u,baseline.u,atol=1e-12)
        np.testing.assert_allclose(result.reactions,1000*baseline.reactions,atol=1e-7)
        np.testing.assert_allclose(result.h,1000*baseline.h,atol=1e-7)
        self.assertEqual(result.as_dict()['units']['force'],'N')
