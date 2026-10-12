"""保留原书答案与已定位的条件错配；不把印刷差异改成Solver正确值。"""
from copy import deepcopy
import unittest
import numpy as np
from examples.audit_textbook_iii1 import audit,original_model,compare,free_equations,BOOK_U


class TextbookIII1AuditTests(unittest.TestCase):
    def test_original_conditions_independent_equations(self):
        report=audit()
        np.testing.assert_allclose(report['computed_displacement_course'],report['independent_displacement_course'],atol=1e-14,rtol=1e-10)
        self.assertEqual(report['original_model']['sections'][0]['A'],.1)
        self.assertEqual(report['original_model']['sections'][0]['I'],.000005)

    def test_answer_reproduced_only_by_next_example_sections(self):
        report=audit()
        self.assertTrue(report['displacement_matches_printed_7_decimal_places'])
        self.assertTrue(report['all_sections_match_float32_and_print_rounding_tolerance'])
        self.assertGreater(report['stated_vs_book_max_force_error_kN'],.05)
        self.assertLess(report['alternate_max_force_error_kN'],5e-7)
        self.assertLess(report['alternate_max_moment_error_kNm'],5e-8)

    def test_printed_displacement_and_float32_cannot_explain_difference(self):
        report=audit()
        residual=np.abs(report['printed_displacement_residual'])
        bound=np.array(report['printed_displacement_rounding_residual_bound'])
        self.assertTrue(np.all(residual>bound))
        self.assertLess(np.max(np.abs(report['float32_difference'])),1e-10)

    def test_element2_reversal_changes_local_moment_not_global_displacement(self):
        book_model=original_model();current=deepcopy(book_model)
        current['elements'][1].update(node_i=4,node_j=3)
        a,ua,fa=compare(book_model);b,ub,fb=compare(current)
        np.testing.assert_allclose(ua,ub,atol=1e-14,rtol=1e-10)
        np.testing.assert_allclose(fa[1,:2],fb[1,:2],atol=1e-10)
        self.assertAlmostEqual(fa[1,2],-fb[1,2])


if __name__=='__main__':unittest.main()
