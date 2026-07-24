import unittest

import contact_side_filter as sides
import symbolic_enumerator as base


class ContactSideFilterTests(unittest.TestCase):
    def test_reverse_contact_requires_direct_copy(self):
        self.assertEqual(sides.required_mirror_sign(base.REVERSE), sides.DIRECT)

    def test_forward_contact_requires_reflected_copy(self):
        self.assertEqual(sides.required_mirror_sign(base.FORWARD), sides.REFLECTED)

    def test_inferred_parity_always_places_copy_on_opposite_side(self):
        for case in base.enumerate_placement_cases():
            analysis = sides.analyze_contact_sides(case, allow_reflections=True)
            self.assertTrue(analysis.feasible)
            for constraint in analysis.constraints:
                self.assertEqual(constraint.copy_interior_side, sides.RIGHT)

    def test_direct_only_mode_rejects_a_required_reflection(self):
        case = next(
            case
            for case in base.enumerate_placement_cases()
            if case.a_direction == base.FORWARD
        )
        analysis = sides.analyze_contact_sides(case, allow_reflections=False)
        self.assertFalse(analysis.feasible)
        self.assertTrue(analysis.requires_reflection)

    def test_voderberg_contacts_require_two_direct_copies(self):
        expected_loci = {
            "A_start": "P1",
            "A_end": "P0",
            "B_start": "P1",
            "B_end": "A",
        }
        case = next(
            case
            for case in base.enumerate_placement_cases()
            if case.marker_locus_map() == expected_loci
            and case.a_interior_blocks == (("B_end",),)
            and case.b_interior_blocks == ()
            and case.a_direction == base.REVERSE
            and case.b_direction == base.REVERSE
        )
        analysis = sides.analyze_contact_sides(case)
        self.assertEqual(analysis.parity_label(), "DD")
        self.assertFalse(analysis.requires_reflection)


if __name__ == "__main__":
    unittest.main()
