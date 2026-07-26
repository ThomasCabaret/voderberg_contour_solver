import unittest

import symbolic_enumerator as base
import voderberg_type_classifier as classifier
from test_voderberg_profile import VoderbergProfileTests


class VoderbergTypeClassifierTests(unittest.TestCase):
    def test_classic_voderberg_placement_is_type1(self):
        case = VoderbergProfileTests.find_voderberg_case()
        result = classifier.classify_placement(case)
        self.assertEqual(result.compatible_types, (classifier.TYPE_1,))
        self.assertEqual(result.witnesses[0].principal_contact, "A")
        self.assertEqual(result.witnesses[0].secondary_contact, "B")

    def test_glide_reflection_topology_is_type2(self):
        case = next(
            case
            for case in base.enumerate_placement_cases()
            if case.case_id == 218
        )
        result = classifier.classify_placement(case)
        self.assertEqual(result.compatible_types, (classifier.TYPE_2,))
        self.assertEqual(result.witnesses[0].principal_contact, "B")
        self.assertEqual(result.witnesses[0].secondary_contact, "A")

    def test_selection_modes_are_distinct(self):
        type1 = classifier.VoderbergTypeClassification(
            compatible_types=(classifier.TYPE_1,), witnesses=()
        )
        neither = classifier.VoderbergTypeClassification(
            compatible_types=(), witnesses=()
        )
        self.assertTrue(classifier.matches_selection(neither, "all"))
        self.assertTrue(classifier.matches_selection(type1, "type1"))
        self.assertFalse(classifier.matches_selection(type1, "type2"))
        self.assertTrue(classifier.matches_selection(type1, "type1+type2"))
        self.assertFalse(classifier.matches_selection(neither, "type1+type2"))

    def test_geometry_record_requires_annotation_for_filtered_mode(self):
        with self.assertRaisesRegex(ValueError, "no voderberg_type annotation"):
            classifier.record_matches_selection({}, "type2")
        self.assertTrue(classifier.record_matches_selection({}, "all"))


if __name__ == "__main__":
    unittest.main()
