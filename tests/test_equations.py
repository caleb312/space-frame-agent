import unittest
import json
from pathlib import Path
import numpy as np

from src.solver.equations import SolverError, ldlt, slveq


class EquationTests(unittest.TestCase):
    def test_course_four_bar_reference(self):
        # C06 p15、p16、p20；使用已记录的 δy3 负号勘误。
        path = Path(__file__).resolve().parents[1] / 'examples/course_four_bar_reference.json'
        reference = json.loads(path.read_text(encoding='utf-8'))
        r = np.array(reference['reduced_stiffness'])
        p = np.array(reference['load'])
        factors = ldlt(r)
        actual = slveq(factors,p)
        expected = np.array(reference['displacement'])
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(r@actual,p,atol=1e-12)

    def test_factorization_and_numpy_reference(self):
        rng = np.random.default_rng(10)
        a = rng.normal(size=(8,8))
        r = a.T@a + np.eye(8)
        p = rng.normal(size=8)
        old = r.copy()
        f = ldlt(r)
        reconstructed = f.scale[:,None] * (f.L@np.diag(f.D)@f.L.T) * f.scale[None,:]
        np.testing.assert_allclose(reconstructed,r,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(slveq(f,p),np.linalg.solve(r,p),rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(old,r)

    def test_mixed_scales(self):
        s = np.array([1e-8,1e8])
        r = s[:,None]*np.array([[2.,.2],[.2,3.]])*s[None,:]
        expected = np.array([2e8,-3e-8])
        np.testing.assert_allclose(slveq(ldlt(r),r@expected),expected,rtol=1e-13)

    def test_failure_cases(self):
        for r in [np.ones((2,2)), np.diag([1.,0.]), np.array([[1.,2.],[2.,1.]]),
                  np.array([[1.,0.],[1.,1.]]), np.array([[np.nan]])]:
            with self.subTest(r=r.tolist()), self.assertRaises(SolverError):
                ldlt(r)
        with self.assertRaises(SolverError):
            slveq(ldlt(np.eye(2)),[1])

    def test_no_free_dofs(self):
        f = ldlt(np.empty((0,0)))
        self.assertEqual(slveq(f,np.array([])).shape,(0,))
