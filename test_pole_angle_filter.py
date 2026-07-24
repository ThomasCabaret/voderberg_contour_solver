import unittest
from fractions import Fraction

import pole_angle_filter as pole
import symbolic_enumerator as base
import turning_filter as turning


class PoleAngleFilterTests(unittest.TestCase):
    @staticmethod
    def find_voderberg_case() -> base.PlacementCase:
        expected_loci = {
            "A_start": "P1",
            "A_end": "P0",
            "B_start": "P1",
            "B_end": "A",
        }
        matches = [
            case
            for case in base.enumerate_placement_cases()
            if case.marker_locus_map() == expected_loci
            and case.a_interior_blocks == (("B_end",),)
            and case.b_interior_blocks == ()
            and case.a_direction == base.REVERSE
            and case.b_direction == base.REVERSE
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"Expected one Voderberg placement, found {len(matches)}"
            )
        return matches[0]

    @classmethod
    def voderberg_two_parameter_state(cls):
        case = cls.find_voderberg_case()
        state = next(
            state
            for state, _derivation in base.enumerate_terminal_states(
                case, max_depth=10, max_states=1000
            )
            if base.word_to_text(state.environment_map()["A0"]) == "V0"
            and base.word_to_text(state.environment_map()["A1"])
            == "V1 V1^-1 V0^-1"
            and base.word_to_text(state.environment_map()["B0"])
            == "V0 V1 V1^-1"
        )
        return case, state

    def test_pole_contacts_use_the_correct_three_prototype_points(self):
        case = self.find_voderberg_case()
        contacts = pole.pole_contact_points(case)
        self.assertEqual(contacts["P0"], ("P0", "P1", "S[B_end]"))
        self.assertEqual(contacts["P1"], ("P1", "P0", "P1"))

    def test_single_shared_turn_class_cannot_fit_both_poles(self):
        self.assertEqual(
            pole.joint_capacity({"Theta0": 1}, {"Theta0": 1}),
            Fraction(1),
        )
        self.assertFalse(
            pole.feasibility_from_coefficients(
                {"Theta0": 1},
                {"Theta0": 1},
            )
        )

    def test_opposite_pole_requirements_are_jointly_impossible(self):
        self.assertEqual(
            pole.joint_capacity({"Theta0": 2}, {"Theta0": -2}),
            Fraction(0),
        )
        self.assertFalse(
            pole.feasibility_from_coefficients(
                {"Theta0": 2},
                {"Theta0": -2},
            )
        )

    def test_independent_twofold_turns_can_fit_both_poles(self):
        self.assertEqual(
            pole.joint_capacity({"Theta0": 2}, {"Theta1": 2}),
            Fraction(2),
        )
        self.assertTrue(
            pole.feasibility_from_coefficients(
                {"Theta0": 2},
                {"Theta1": 2},
            )
        )

    def test_voderberg_two_parameter_profile_passes_pole_filter(self):
        case, state = self.voderberg_two_parameter_state()
        angle_solution = turning.complete_angle_solution(case, state)
        analysis = pole.analyze_pole_angles(
            case,
            state,
            angle_solution=angle_solution,
        )

        self.assertTrue(analysis.feasible)
        self.assertEqual(analysis.joint_capacity_pi_units, Fraction(3))
        self.assertEqual(
            analysis.constraints[0].contact_points,
            ("P0", "P1", "S[B_end]"),
        )
        self.assertEqual(
            analysis.constraints[1].contact_points,
            ("P1", "P0", "P1"),
        )


if __name__ == "__main__":
    unittest.main()
