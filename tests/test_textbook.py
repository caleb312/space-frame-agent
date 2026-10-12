import json
from pathlib import Path
import unittest
import numpy as np

from src.solver import solve_space_frame
from src.solver.geometry import ch
from src.solver.stiffness import local_stiffness
from src.solver.textbook_axes import textbook_basis
from examples.textbook_reference import (equivalent_model, displacements_in_appendix_convention,
                                         textbook_nodal_load, section_forces_in_appendix_convention)
from src.model import StructuralModel
from pydantic import ValidationError
from .energy_reference import energy_frame_reference


class TextbookTests(unittest.TestCase):
    def test_chapter5_local_matrix_equations_5_22_to_5_24(self):
        # 书 p124，按变量 EA/L=10, GJ/L=2.5, EIy=3,EIz=4,L=2 独立逐项抄录。
        rda = np.array([[10,0,0,0,0,0],[0,6,0,0,0,-6],[0,0,4.5,0,4.5,0],
                        [0,0,0,2.5,0,0],[0,0,4.5,0,6,0],[0,-6,0,0,0,8]])
        rdb = np.array([[-10,0,0,0,0,0],[0,-6,0,0,0,-6],[0,0,-4.5,0,4.5,0],
                        [0,0,0,-2.5,0,0],[0,0,-4.5,0,3,0],[0,6,0,0,0,4]])
        rde = np.array([[10,0,0,0,0,0],[0,6,0,0,0,6],[0,0,4.5,0,-4.5,0],
                        [0,0,0,2.5,0,0],[0,0,-4.5,0,6,0],[0,6,0,0,0,8]])
        k = local_stiffness(2,20,3,4,5)
        np.testing.assert_allclose(k,np.block([[rda,rdb],[rdb.T,rde]]))

    def test_axes_chapter_and_appendix(self):
        ex = np.array([1.,2.,3.])/np.sqrt(14)
        h = np.hypot(ex[0],ex[1])
        chapter = textbook_basis([0,0,0],[1,2,3])
        ey = [ex[2]*ex[0]/h,ex[2]*ex[1]/h,-h]
        np.testing.assert_allclose(chapter[:,1],ey)
        appendix = textbook_basis([0,0,0],[1,2,3],style='appendix4')
        np.testing.assert_allclose(appendix[:,1:],-chapter[:,1:])
        for direction in ([0,0,1],[0,0,-1]):
            for style in ('chapter5','appendix4'):
                q = textbook_basis([0,0,0],direction,.3,style)
                np.testing.assert_allclose(q.T@q,np.eye(3),atol=1e-14)
                self.assertAlmostEqual(np.linalg.det(q),1)
        rolled = textbook_basis([0,0,0],[1,2,3],.4,'chapter5')
        np.testing.assert_allclose(rolled[:,1],np.cos(.4)*chapter[:,1]+np.sin(.4)*chapter[:,2])
        angle=.4
        a,b,c=ex
        # 附录p183式IV-3独立逐项计算，而非调用相同旋转算法。
        expected=np.array([[a,(b*np.sin(angle)-a*c*np.cos(angle))/h,
                             (b*np.cos(angle)+a*c*np.sin(angle))/h],
                           [b,(-a*np.sin(angle)-b*c*np.cos(angle))/h,
                             (-a*np.cos(angle)+b*c*np.sin(angle))/h],
                           [c,h*np.cos(angle),-h*np.sin(angle)]])
        np.testing.assert_allclose(textbook_basis([0,0,0],[1,2,3],angle,'appendix4'),expected)
        for sign in (1,-1):
            q=textbook_basis([0,0,0],[0,0,sign],angle,'appendix4')
            expected=np.array([[0,np.sin(angle),-sign*np.cos(angle)],
                               [0,np.cos(angle),sign*np.sin(angle)],[sign,0,0]])
            np.testing.assert_allclose(q,expected,atol=1e-14)
        with self.assertRaises(ValueError):
            textbook_basis([0,0,0],[0,0,0])

    def test_appendix_sign_change_is_congruence(self):
        # 书正文 p124 与附录 p194 的耦合项反号：DkD，旋转分量全部反号。
        k_course = local_stiffness(2,20,3,4,5)
        sign = np.tile([1,1,1,-1,-1,-1],2)
        k_appendix = sign[:,None]*k_course*sign[None,:]
        self.assertEqual(k_appendix[1,5],6)
        self.assertEqual(k_appendix[2,4],-4.5)
        self.assertEqual(k_appendix[1,11],6)
        self.assertEqual(k_appendix[4,8],4.5)

    def test_original_iv1_nodal_displacements(self):
        root = Path(__file__).resolve().parents[1]
        raw = json.loads((root/'examples/textbook_iv1_original.json').read_text(encoding='utf-8'))
        data,_,_ = equivalent_model(raw)
        saved = json.loads((root/'examples/textbook_iv1_equivalent_nodes.json').read_text(encoding='utf-8'))
        self.assertEqual(data,saved)
        result = solve_space_frame(data)
        actual = displacements_in_appendix_convention(result)[2:]
        reference = np.array(raw['reference_displacements'])
        np.testing.assert_allclose(actual[:,:3],reference[:,:3],atol=raw['displacement_atol'],rtol=0)
        np.testing.assert_allclose(actual[:,3:],reference[:,3:],atol=raw['rotation_atol'],rtol=0)
        np.testing.assert_allclose(result.force_balance,0,atol=1e-8)
        np.testing.assert_allclose(result.moment_balance,0,atol=1e-8)

    def test_equivalent_load_resultants(self):
        # 独立整体静力核对：右手分量下等效力和力矩应等于原始作用。
        length=6.
        for plane in (0,1):
            for kind in (1,2):
                load={'kind':kind,'plane':plane,'value':4.,'position':3.}
                p = textbook_nodal_load(length,load)
                force = p[:3]+p[6:9]
                moment = p[3:6]+p[9:12]+np.cross([length,0,0],p[6:9])
                total = 4. if kind==1 else 12.
                position = 3. if kind==1 else 1.5
                expected_f=np.zeros(3)
                expected_f[1+plane]=total
                np.testing.assert_allclose(force,expected_f,atol=1e-13)
                np.testing.assert_allclose(moment,np.cross([position,0,0],expected_f),atol=1e-13)

    def test_iv1_original_section_forces(self):
        root = Path(__file__).resolve().parents[1]
        raw=json.loads((root/'examples/textbook_iv1_original.json').read_text(encoding='utf-8'))
        model,equivalent,_=equivalent_model(raw)
        actual=section_forces_in_appendix_convention(raw,solve_space_frame(model),equivalent)
        # 对照原书 float 源程序的表值；单独注明容限，不替换参考答案。
        np.testing.assert_allclose(actual,raw['reference_section_forces'],atol=1e-3,rtol=0)

    def test_iv2_displacements_and_section_forces_with_explicit_erratum(self):
        root = Path(__file__).resolve().parents[1]
        raw=json.loads((root/'examples/textbook_iv2_original.json').read_text(encoding='utf-8'))
        data,equivalent,_=equivalent_model(raw)
        result=solve_space_frame(data)
        actual=displacements_in_appendix_convention(result)[3:]
        expected=np.array(raw['reference_displacements_corrected'])
        np.testing.assert_allclose(actual[:,:3],expected[:,:3],atol=raw['displacement_atol'],rtol=0)
        np.testing.assert_allclose(actual[:,3:],expected[:,3:],atol=raw['rotation_atol'],rtol=0)
        sections=section_forces_in_appendix_convention(raw,result,equivalent)
        np.testing.assert_allclose(sections,raw['reference_section_forces'],atol=1e-3,rtol=0)

    def test_iv2_printed_typo_independently_detected(self):
        root = Path(__file__).resolve().parents[1]
        raw=json.loads((root/'examples/textbook_iv2_original.json').read_text(encoding='utf-8'))
        data,_,_=equivalent_model(raw)
        model=StructuralModel.model_validate(data)
        K,P,U,_,_=energy_frame_reference(model)
        printed=np.zeros_like(U)
        original=np.array(raw['reference_displacements'])
        printed[18:]=original.ravel()
        printed.reshape(-1,6)[:,3:]*=-1
        # 原表的 40/50 大位移破坏自由方程，独立参考不会复现它们。
        self.assertGreater(np.max(np.abs((K@printed-P)[18:])),1e5)
        independent=U.reshape(-1,6)[3:].copy()
        independent[:,3:]*=-1
        expected=np.array(raw['reference_displacements_corrected'])
        np.testing.assert_allclose(independent,expected,atol=5e-7,rtol=0)
        self.assertEqual(original[0,0],40.0025641)
        self.assertEqual(original[1,0],50.0025679)

    def test_benchmark_member_loads_not_silently_accepted_by_solver(self):
        root = Path(__file__).resolve().parents[1]
        raw=json.loads((root/'examples/textbook_iv1_original.json').read_text(encoding='utf-8'))
        data,_,_=equivalent_model(raw)
        data['member_loads']=raw['member_loads']
        with self.assertRaises(ValidationError):
            solve_space_frame(data)
