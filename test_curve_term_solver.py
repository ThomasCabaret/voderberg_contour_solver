import unittest
from types import SimpleNamespace

import curve_term_solver as solver


def occurrence(variable, inverse=False):
    return SimpleNamespace(variable=variable, inverse=inverse)


def self_mapping(*, target_orientation, mirror_sign):
    return {
        "schema_version": "terminal-contact-mapping-v1",
        "segment_count": 1,
        "mappings": [{
            "copy_index": 0,
            "mirror_sign": mirror_sign,
            "segment_pairs": [{
                "source_position": 0,
                "source_orientation": 1,
                "target_position": 0,
                "target_orientation": target_orientation,
            }],
        }],
    }


class CurveTermSolverTests(unittest.TestCase):
    def solve(self, target_orientation, mirror_sign):
        return solver.solve_curve_terms(
            curve_variables=("V0",),
            occurrences=(occurrence("V0"),),
            terminal_mapping=self_mapping(
                target_orientation=target_orientation,
                mirror_sign=mirror_sign,
            ),
        )

    def test_same_direction_reflection_is_formal_straight_value(self):
        result = self.solve(target_orientation=1, mirror_sign=-1)
        term = result.terms["V0"]
        self.assertEqual(term["kind"], "straight")
        self.assertEqual(term["text"], "Straight(lambda0)")
        self.assertTrue(result.length_classes["lambda0"]["positive"])

    def test_endpoint_swapping_reflection_is_mirror_of_inverse_half(self):
        result = self.solve(target_orientation=-1, mirror_sign=-1)
        term = result.terms["V0"]
        self.assertEqual(term["kind"], "mirror_symmetric_join")
        self.assertIn("Mirror(Inverse(C0_half))", term["text"])
        self.assertEqual(result.internal_angle_parameters, ("curve_angle0",))

    def test_direct_reversal_is_only_marked_as_already_word_solved(self):
        result = self.solve(target_orientation=-1, mirror_sign=1)
        term = result.terms["V0"]
        self.assertEqual(term["kind"], "word_solver_resolved_central_symmetry")
        self.assertFalse(term["adds_new_symbolic_degrees_of_freedom"])

    def test_specialization_can_be_disabled_without_changing_generic_words(self):
        result = solver.solve_curve_terms(
            curve_variables=("V0",),
            occurrences=(occurrence("V0"),),
            terminal_mapping=None,
            enabled=False,
        )
        self.assertFalse(result.enabled)
        self.assertEqual(result.terms["V0"]["kind"], "curve_parameter")
        self.assertFalse(result.relation_analysis.enabled)

    def test_identity_adds_no_wrapper(self):
        result = self.solve(target_orientation=1, mirror_sign=1)
        self.assertEqual(result.terms["V0"]["kind"], "curve_parameter")

    def test_formal_solution_has_no_numeric_geometry(self):
        payload = self.solve(target_orientation=1, mirror_sign=-1).to_dict()
        self.assertFalse(payload["numeric_geometry_embedded"])
        self.assertNotIn("numeric_template_compiler", payload)


if __name__ == "__main__":
    unittest.main()
