import unittest

import symbolic_enumerator as base
import turning_filter as turning


class TurningFilterTests(unittest.TestCase):
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
    def voderberg_state(cls):
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

    def test_inverse_curve_occurrences_cancel_total_turn(self):
        self.assertTrue(
            turning.feasibility_from_coefficients(
                segment_coefficients={"X": 1},
                point_coefficients={},
            )
        )
        self.assertFalse(
            turning.feasibility_from_coefficients(
                segment_coefficients={"X": 0},
                point_coefficients={},
            )
        )

    def test_two_regular_point_classes_are_not_enough_for_one_full_turn(self):
        # The attainable interval is (-2*pi, 2*pi), so +2*pi is not attained
        # without an excluded pi U-turn at the boundary.
        self.assertFalse(
            turning.feasibility_from_coefficients(
                segment_coefficients={},
                point_coefficients={"Theta0": 1, "Theta1": 1},
            )
        )

    def test_three_point_classes_can_supply_one_full_turn(self):
        self.assertTrue(
            turning.feasibility_from_coefficients(
                segment_coefficients={},
                point_coefficients={
                    "Theta0": 1,
                    "Theta1": 1,
                    "Theta2": 1,
                },
            )
        )

    def test_voderberg_profile_has_one_free_curve_turn(self):
        case, state = self.voderberg_state()
        analysis = turning.analyze_total_turn(case, state)

        self.assertEqual(analysis.segment_map(), {"V0": 1})
        self.assertTrue(analysis.has_unbounded_segment_freedom)
        self.assertTrue(analysis.feasible)
        self.assertIn("Kappa[V0]", analysis.equation_text())
        self.assertIn("= 2*pi", analysis.equation_text())

    def test_complete_angle_solution_includes_unconstrained_poles(self):
        case, state = self.voderberg_state()
        solution = turning.complete_angle_solution(case, state)
        assignments = solution.assignment_map()
        self.assertIn("P0", assignments)
        self.assertIn("P1", assignments)

    def test_real_terminal_profile_can_be_discarded(self):
        case = base.find_case(2)
        state, _derivation = next(
            base.enumerate_terminal_states(case, max_depth=10, max_states=1000)
        )
        analysis = turning.analyze_total_turn(case, state)
        self.assertFalse(analysis.feasible)
        self.assertEqual(analysis.segment_map(), {})
        self.assertEqual(sum(abs(value) for value in analysis.point_map().values()), 1)
        self.assertTrue(analysis.equation_text().endswith("= 2*pi"))


if __name__ == "__main__":
    unittest.main()
