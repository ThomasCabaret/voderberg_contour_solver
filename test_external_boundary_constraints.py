import unittest
from fractions import Fraction

import external_boundary_constraints as external
import symbolic_enumerator as base


class ExternalBoundaryPrototypeTests(unittest.TestCase):
    @staticmethod
    def voderberg_case_and_state():
        expected_loci = {
            "A_start": "P1",
            "A_end": "P0",
            "B_start": "P1",
            "B_end": "A",
        }
        case = next(
            case
            for case in base.enumerate_placement_cases()
            if (
                case.marker_locus_map() == expected_loci
                and case.a_interior_blocks == (("B_end",),)
                and case.b_interior_blocks == ()
                and case.a_direction == base.REVERSE
                and case.b_direction == base.REVERSE
            )
        )
        state = next(
            state
            for state, derivation in base.enumerate_terminal_states(case)
            if derivation
            == (
                "equal_length",
                "left_strictly_shorter",
                "involutive_palindrome",
            )
        )
        return case, state

    def test_voderberg_external_free_factors(self):
        case, state = self.voderberg_case_and_state()
        system = external.build_joint_boundary_system(case, state)
        words = dict(system.outer_boundary.initial_words)
        self.assertEqual(base.word_to_text(words[external.A_COPY]), "B0")
        self.assertEqual(base.word_to_text(words[external.B_COPY]), "B0 A0")
        self.assertEqual(system.outer_boundary.points[0].physical_point, external.OUTER_P0)
        self.assertEqual(system.outer_boundary.points[-1].physical_point, external.OUTER_P0)

    def test_voderberg_outer_rotation_is_gluing_identity(self):
        case, state = self.voderberg_case_and_state()
        system = external.build_joint_boundary_system(case, state)
        inner, outer = system.rotation_equations
        expected = inner.lhs.scale(3).add(
            external.AngleForm(pi_constant=Fraction(-4))
        )
        self.assertEqual(outer.lhs, expected)
        self.assertTrue(system.rotation_analysis.feasible)

    def test_reflected_contact_produces_conjugated_chords(self):
        for case in base.enumerate_placement_cases():
            if not case.requires_reflection:
                continue
            terminal = next(
                base.enumerate_terminal_states(case, max_depth=3, max_states=30),
                None,
            )
            if terminal is None:
                continue
            state, _derivation = terminal
            system = external.build_joint_boundary_system(case, state)
            outer_coefficients = system.translation_equations[1].coefficients
            self.assertTrue(
                any(item.chord.conjugated for item in outer_coefficients)
            )
            return
        self.fail("No reflected terminal case was found for the smoke test")


if __name__ == "__main__":
    unittest.main()
