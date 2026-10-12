"""原生模型复用既有查看器；原始Solver结果不改动。"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.solver import solve_structure
from src.visualization.native import build_native_payload,write_native_html
from tests.test_specialized_solver import model

ROOT=Path(__file__).resolve().parents[1]


class NativeVisualizationTests(unittest.TestCase):
    def test_truss_real_fields_and_space_load_glyph(self):
        for kind in ('plane_truss','space_truss'):
            dim=2 if kind=='plane_truss' else 3
            raw=model(kind,[[0]*dim,[2]+[0]*(dim-1)],[[True]*dim,[False]+[True]*(dim-1)],[(2,[4]+[0]*(dim-1))])
            result=solve_structure(raw).as_dict();before=deepcopy(result)
            payload=build_native_payload(raw,result)
            self.assertEqual(payload['native_dof_indices'],list(range(dim)))
            self.assertNotIn('Mz',payload['fields']);self.assertNotIn('rz',payload['fields'])
            self.assertEqual(len(payload['glyphs']),1)
            self.assertTrue(all(len(p)==3 for path in payload['glyphs'][0]['paths'] for p in path))
            samples=payload['elements'][0]['curve']['samples']
            self.assertAlmostEqual(samples[len(samples)//2]['displacement'][0],result['nodes'][1]['displacement'][0]/2)
            self.assertEqual(result,before)

    def test_plane_frame_curvature_and_actual_dofs(self):
        raw=model('plane_frame',[[0,0],[2,0]],[[True]*3,[False]*3],[(2,[0,-10,0])])
        result=solve_structure(raw).as_dict();payload=build_native_payload(raw,result)
        self.assertEqual(payload['native_dof_indices'],[0,1,5])
        self.assertEqual(set(payload['fields']),{'u_norm','ux','uy','rz','N','Qy','Mz'})
        samples=payload['elements'][0]['curve']['samples'];mid=samples[len(samples)//2]
        EI=raw['materials'][0]['E']*raw['sections'][0]['I'];x=1.;L=2.
        self.assertAlmostEqual(mid['displacement'][1],-10*x*x*(3*L-x)/(6*EI))
        self.assertNotAlmostEqual(mid['displacement'][1],result['nodes'][1]['displacement'][1]/2)
        self.assertLess(payload['elements'][0]['curve']['endpoint_error_m'],1e-12)

    def test_uniform_load_curve_even_when_both_nodes_fixed(self):
        raw=model('plane_frame',[[0,0],[2,0]],[[True]*3,[True]*3],[])
        raw['member_loads']=[dict(element_id=100,kind='uniform_force',coordinate_system='global',components=[0,-10,0],interval=[0,2])]
        payload=build_native_payload(raw,solve_structure(raw).as_dict())
        mid=payload['elements'][0]['curve']['samples'][32]
        EI=raw['materials'][0]['E']*raw['sections'][0]['I']
        self.assertAlmostEqual(mid['displacement'][1],-10*2**4/(384*EI))
        self.assertGreater(payload['max_sample_displacement_m'],0)
        self.assertEqual(payload['max_node_displacement_m'],0)
        self.assertEqual(payload['loads'][0]['interval'],[0,2])

    def test_point_force_jump_and_local_load_arrows(self):
        raw=model('plane_frame',[[0,0],[0,2]],[[True]*3,[False]*3],[])
        raw['member_loads']=[dict(element_id=100,kind='point_force',coordinate_system='local',components=[0,-10,0],position=1)]
        payload=build_native_payload(raw,solve_structure(raw).as_dict())
        rows=[s for s in payload['elements'][0]['curve']['samples'] if s['position']==1]
        self.assertEqual([s['side'] for s in rows],['left','right'])
        self.assertAlmostEqual(abs(rows[1]['forces'][1]-rows[0]['forces'][1]),10)
        self.assertEqual(payload['loads'][0]['global_components'],[10,0,0])

    def test_preview_has_controls_but_no_invented_results(self):
        raw=model('plane_truss',[[0,0],[2,0]],[[True]*2,[False,True]],[])
        payload=build_native_payload(raw)
        self.assertFalse(payload['solved'])
        self.assertIsNone(payload['elements'][0]['curve'])
        self.assertIsNone(payload['nodes'][0]['displacement'])
        self.assertTrue(all(not f['available'] for f in payload['fields'].values()))

    def test_hash_mismatch_rejected(self):
        raw=model('plane_frame',[[0,0],[2,0]],[[True]*3,[False]*3],[(2,[0,-10,0])])
        result=solve_structure(raw).as_dict();result['model_hash']='incorrect'
        with self.assertRaises(ValueError):build_native_payload(raw,result)

    def test_html_reuses_original_controls_and_detail_panel(self):
        raw=model('plane_frame',[[0,0],[2,0]],[[True]*3,[False]*3],[(2,[0,-10,0])])
        (ROOT/'work').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as folder:
            path=Path(folder)/'frame.html'
            write_native_html(raw,solve_structure(raw).as_dict(),path)
            text=path.read_text(encoding='utf-8')
        for marker in ('id="field"','id="scaleSlider"','id="scale"','data-camera="xy"',
                       'data-camera="iso"','id="detail"','data-tab="nodes"','data-tab="ends"','data-tab="loads"'):
            self.assertIn(marker,text)
        self.assertNotIn('__DATA__',text)


if __name__=='__main__':unittest.main()
