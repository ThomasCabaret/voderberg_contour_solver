import unittest
from types import SimpleNamespace

import curve_template_constraints as constraints


def occurrence(variable, inverse=False):
    return SimpleNamespace(variable=variable, inverse=inverse)


def mapping(*pairs, mirror_sign=1):
    return {
        "schema_version": "terminal-contact-mapping-v1",
        "segment_count": max(
            max(pair[0], pair[2]) for pair in pairs
        ) + 1,
        "mappings": [
            {
                "copy_index": 0,
                "mirror_sign": mirror_sign,
                "segment_pairs": [
                    {
                        "source_position": source_position,
                        "source_orientation": source_orientation,
                        "target_position": target_position,
                        "target_orientation": target_orientation,
                    }
                    for (
                        source_position,
                        source_orientation,
                        target_position,
                        target_orientation,
                    ) in pairs
                ],
            }
        ],
    }


class CurveTemplateConstraintTests(unittest.TestCase):
    def test_reflected_same_directed_occurrence_collapses_to_straight(self):
        plan = constraints.build_plan(
            curve_variables=("V0",),
            occurrences=(occurrence("V0"),),
            terminal_mapping=mapping((0, 1, 0, 1), mirror_sign=-1),
            kappa_assignments={"V0": "K0"},
            intermediate_points=4,
        )
        component = plan.component_for("V0")
        self.assertEqual(component.mode, "straight")
        self.assertEqual(component.effective_edge_count, 1)
        self.assertEqual(component.free_length_count, 1)
        self.assertEqual(component.free_turn_count, 0)
        self.assertEqual(plan.kappa_reduction.assignments["K0"], "0")

    def test_direct_reversal_keeps_nontrivial_half_turn_template(self):
        plan = constraints.build_plan(
            curve_variables=("V0",),
            occurrences=(occurrence("V0"),),
            terminal_mapping=mapping((0, 1, 0, -1), mirror_sign=1),
            kappa_assignments={"V0": "K0"},
            intermediate_points=4,
        )
        component = plan.component_for("V0")
        self.assertEqual(component.mode, "half_turn")
        self.assertEqual(component.effective_edge_count, 5)
        self.assertEqual(plan.kappa_reduction.assignments["K0"], "0")
        # Three free palindrome lengths and two anti-palindrome turns.
        decoded = constraints.decode_templates(
            plan,
            {"V0": "K0"},
            (1.0, 2.0, 3.0, 0.2, -0.4),
            kappa_values={},
        )
        self.assertEqual(decoded.lengths["V0"], (1.0, 2.0, 3.0, 2.0, 1.0))
        self.assertEqual(decoded.turns["V0"], (0.2, -0.4, 0.4, -0.2))

    def test_reflected_reversal_keeps_endpoint_swapping_mirror_shape(self):
        plan = constraints.build_plan(
            curve_variables=("V0",),
            occurrences=(occurrence("V0"),),
            terminal_mapping=mapping((0, 1, 0, -1), mirror_sign=-1),
            kappa_assignments={"V0": "K0"},
            intermediate_points=3,
        )
        component = plan.component_for("V0")
        self.assertEqual(component.mode, "endpoint_swapping_reflection")
        self.assertEqual(component.effective_edge_count, 4)
        self.assertEqual(plan.kappa_parameter_names, ("K0",))
        decoded = constraints.decode_templates(
            plan,
            {"V0": "K0"},
            (1.0, 2.0, 0.3, 0.7),
            kappa_values={"K0": 0.7},
        )
        self.assertEqual(decoded.lengths["V0"], (1.0, 2.0, 2.0, 1.0))
        self.assertAlmostEqual(decoded.turns["V0"][0], 0.3)
        self.assertAlmostEqual(decoded.turns["V0"][1], 0.1)
        self.assertAlmostEqual(decoded.turns["V0"][2], 0.3)

    def test_relation_between_distinct_variables_shares_transformed_template(self):
        plan = constraints.build_plan(
            curve_variables=("V0", "V1"),
            occurrences=(occurrence("V0"), occurrence("V1", inverse=True)),
            terminal_mapping=mapping((0, 1, 1, 1), mirror_sign=-1),
            kappa_assignments={"V0": "K0", "V1": "K0"},
            intermediate_points=2,
        )
        self.assertEqual(len(plan.components), 1)
        component = plan.components[0]
        self.assertEqual(component.transforms["V1"].label, "mirror_reverse")
        decoded = constraints.decode_templates(
            plan,
            {"V0": "K0", "V1": "K0"},
            (1.0, 2.0, 3.0, 0.25, 0.4),
            kappa_values={"K0": 0.4},
        )
        self.assertEqual(decoded.lengths["V1"], tuple(reversed(decoded.lengths["V0"])))
        # mirror+reverse reverses turn order without changing signs.
        self.assertEqual(decoded.turns["V1"], tuple(reversed(decoded.turns["V0"])))

    def test_reported_false_positive_profile_forces_both_variables_straight(self):
        occurrences = (
            occurrence("V0", True),
            occurrence("V1"),
            occurrence("V1", True),
            occurrence("V0"),
            occurrence("V0", True),
            occurrence("V1"),
            occurrence("V1", True),
        )
        terminal_mapping = {
            "schema_version": "terminal-contact-mapping-v1",
            "segment_count": 7,
            "mappings": [
                {
                    "copy_index": 0,
                    "mirror_sign": 1,
                    "segment_pairs": [
                        {"source_position": 0, "source_orientation": 1, "target_position": 3, "target_orientation": -1},
                        {"source_position": 1, "source_orientation": 1, "target_position": 2, "target_orientation": -1},
                        {"source_position": 2, "source_orientation": 1, "target_position": 1, "target_orientation": -1},
                        {"source_position": 3, "source_orientation": 1, "target_position": 0, "target_orientation": -1},
                    ],
                },
                {
                    "copy_index": 1,
                    "mirror_sign": -1,
                    "segment_pairs": [
                        {"source_position": 4, "source_orientation": 1, "target_position": 0, "target_orientation": 1},
                        {"source_position": 5, "source_orientation": 1, "target_position": 1, "target_orientation": 1},
                        {"source_position": 6, "source_orientation": 1, "target_position": 2, "target_orientation": 1},
                    ],
                },
            ],
        }
        plan = constraints.build_plan(
            curve_variables=("V0", "V1"),
            occurrences=occurrences,
            terminal_mapping=terminal_mapping,
            kappa_assignments={"V0": "K0", "V1": "K1"},
            intermediate_points=3,
        )
        self.assertEqual(plan.component_for("V0").mode, "straight")
        self.assertEqual(plan.component_for("V1").mode, "straight")
        self.assertEqual(plan.component_for("V0").effective_edge_count, 1)
        self.assertEqual(plan.component_for("V1").effective_edge_count, 1)

    def test_missing_terminal_mapping_is_rejected_when_enabled(self):
        with self.assertRaises(constraints.TemplateConstraintError):
            constraints.build_plan(
                curve_variables=("V0",),
                occurrences=(occurrence("V0"),),
                terminal_mapping=None,
                kappa_assignments={"V0": "0"},
                intermediate_points=1,
                enabled=True,
            )


if __name__ == "__main__":
    unittest.main()
