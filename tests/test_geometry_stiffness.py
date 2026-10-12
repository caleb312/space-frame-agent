import unittest
import numpy as np

from src.solver.geometry import ch, force_transform, mtmult, trnsps
from src.solver.stiffness import local_stiffness, stif


class GeometryTests(unittest.TestCase):
    def test_oblique_and_vertical_axes(self):
        for end, hint in [([1, 2, 3], [0, 1, 0]), ([0, 0, 5], [0, 1, 0]), ([-2, 0, 0], [0, 1, 0])]:
            with self.subTest(end=end):
                clxy, q = ch([0, 0, 0], end, hint)
                self.assertAlmostEqual(clxy[0], np.linalg.norm(end))
                np.testing.assert_allclose(q[:, 0], np.array(end)/clxy[0])
                np.testing.assert_allclose(q.T@q, np.eye(3), atol=1e-14)
                self.assertAlmostEqual(np.linalg.det(q), 1)
                t, t12 = force_transform(q)
                self.assertEqual(t.shape, (6, 6))
                np.testing.assert_allclose(t12.T@t12, np.eye(12), atol=1e-14)

    def test_work_invariant(self):
        _, q = ch([0, 0, 0], [1, 2, 3], [0, 1, 0])
        _, t = force_transform(q)
        rng = np.random.default_rng(5)
        U, S = rng.normal(size=(2, 12))
        self.assertAlmostEqual(np.dot(U, t@S), np.dot(t.T@U, S))

    def test_geometry_failure(self):
        for args in [([0]*3, [0]*3, [0, 1, 0]), ([0]*3, [1, 0, 0], [1, 0, 0]),
                     ([0]*3, [1, 0, 0], [0]*3), ([0]*2, [1, 0, 0], [0, 1, 0])]:
            with self.assertRaises(ValueError):
                ch(*args)
        with self.assertRaises(ValueError):
            force_transform(np.diag([1, 1, -1]))

    def test_matrix_helpers(self):
        a, b = np.arange(6.).reshape(2, 3), np.arange(12.).reshape(3, 4)
        old = a.copy()
        np.testing.assert_array_equal(trnsps(a), a.T)
        np.testing.assert_array_equal(a, old)
        np.testing.assert_array_equal(mtmult(a, b), a@b)
        with self.assertRaises(ValueError):
            mtmult(b, a)


class StiffnessTests(unittest.TestCase):
    def test_course_planar_matrix(self):
        # C09 p21/p33～34，L=2, EA=20, EJ=4，顺时针为正。
        expected = np.array([[10,0,0,-10,0,0], [0,6,-6,0,-6,-6],
                             [0,-6,8,0,6,4], [-10,0,0,10,0,0],
                             [0,-6,6,0,6,6], [0,-6,4,0,6,8]], dtype=float)
        k = local_stiffness(2, 20, 3, 4, 5)
        indexes = [0, 1, 5, 6, 7, 11]
        np.testing.assert_allclose(k[np.ix_(indexes, indexes)], expected)

    def test_symmetry_and_six_rigid_modes(self):
        length = 2.3
        k = local_stiffness(length, 20, 3, 4, 5)
        self.assertEqual(k.shape, (12, 12))
        np.testing.assert_allclose(k, k.T)
        modes = np.zeros((12, 6))
        modes[:3, :3] = np.eye(3)
        modes[6:9, :3] = np.eye(3)
        modes[3:6, 3:] = np.eye(3)
        modes[9:12, 3:] = np.eye(3)
        # θ_RH=−θ_course；j 端刚体平移为 −θ_course×(L,0,0)。
        for axis in range(3):
            modes[6:9, 3+axis] = -np.cross(np.eye(3)[axis], [length, 0, 0])
        np.testing.assert_allclose(k@modes, 0, atol=1e-13)
        eigenvalues = np.linalg.eigvalsh(k)
        self.assertEqual(np.count_nonzero(eigenvalues > 1e-9), 6)
        self.assertGreater(eigenvalues.min(), -1e-12)

    def test_four_blocks_and_energy(self):
        length, q = ch([0,0,0], [1,2,3], [0,1,0])
        e = stif(length[0], q, 20, 3, 4, 5)
        for name in ('rda','rdb','rdd','rde','ca','cb','cd','ce'):
            self.assertEqual(getattr(e,name).shape, (6,6))
        np.testing.assert_allclose(e.cb.T, e.cd, atol=1e-14)
        np.testing.assert_allclose(e.global_matrix, e.t12@e.local@e.t12.T, atol=1e-14)
        U = np.arange(12.) / 10
        self.assertAlmostEqual(U@e.global_matrix@U, (e.t12.T@U)@e.local@(e.t12.T@U))

    def test_invalid_rigidity(self):
        with self.assertRaises(ValueError):
            local_stiffness(2,20,3,4,0)
