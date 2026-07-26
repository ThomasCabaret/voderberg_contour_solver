import unittest
from types import SimpleNamespace

import curve_relation_algebra as algebra


def occurrence(variable, inverse=False):
    return SimpleNamespace(variable=variable, inverse=inverse)


def mapping(*pairs, mirror_sign):
    return {
        "schema_version": "terminal-contact-mapping-v1",
        "segment_count": max(max(pair[0], pair[2]) for pair in pairs) + 1,
        "mappings": [{
            "copy_index": 0,
            "mirror_sign": mirror_sign,
            "segment_pairs": [{
                "source_position": source_position,
                "source_orientation": source_orientation,
                "target_position": target_position,
                "target_orientation": target_orientation,
            } for source_position, source_orientation, target_position, target_orientation in pairs],
        }],
    }


class CurveRelationAlgebraTests(unittest.TestCase):
    def test_same_direction_reflected_self_relation_is_straight(self):
        result = algebra.analyze_curve_relations(
            curve_variables=("V0",),
            occurrences=(occurrence("V0"),),
            terminal_mapping=mapping((0, 1, 0, 1), mirror_sign=-1),
        )
        self.assertEqual(result.component_for("V0").mode, "straight")

    def test_reflected_reversal_is_endpoint_swapping_symmetry(self):
        result = algebra.analyze_curve_relations(
            curve_variables=("V0",),
            occurrences=(occurrence("V0"),),
            terminal_mapping=mapping((0, 1, 0, -1), mirror_sign=-1),
        )
        self.assertEqual(
            result.component_for("V0").mode,
            "endpoint_swapping_reflection",
        )

    def test_distinct_variables_record_mirror_inverse_function(self):
        result = algebra.analyze_curve_relations(
            curve_variables=("V0", "V1"),
            occurrences=(occurrence("V0"), occurrence("V1", inverse=True)),
            terminal_mapping=mapping((0, 1, 1, 1), mirror_sign=-1),
        )
        component = result.component_for("V0")
        self.assertEqual(component.transforms["V1"].label, "mirror_reverse")


if __name__ == "__main__":
    unittest.main()
