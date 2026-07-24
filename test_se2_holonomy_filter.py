import unittest

import angle_constraints as angles
import se2_holonomy_filter as se2
import symbolic_enumerator as base
import turning_filter as turning


class SE2HolonomyFilterTests(unittest.TestCase):
    @staticmethod
    def _phase(**coefficients):
        return se2.LinearAngleForm.from_mapping(coefficients)

    @staticmethod
    def _coefficient(variable, groups):
        return se2.CurveTranslationCoefficient(
            variable=variable,
            phases=tuple(
                se2.PhaseMultiplicity(phase=phase, multiplicity=multiplicity)
                for phase, multiplicity in groups
            ),
        )

    def test_linear_angle_form_combines_integer_coefficients(self):
        form = self._phase(A=1).add_term("A", -1).add_term("B", 2)
        self.assertEqual(form.coefficients, (("B", 2),))

    def test_single_formal_chord_direction_is_an_exact_obstruction(self):
        coefficient = self._coefficient("X", [(self._phase(), 3)])
        multiplicities = tuple(item.multiplicity for item in coefficient.phases)
        self.assertEqual(multiplicities, (3,))

    def test_two_unequal_vector_groups_cannot_cancel(self):
        coefficient = self._coefficient(
            "X",
            [(self._phase(A=1), 2), (self._phase(B=1), 1)],
        )
        self.assertNotEqual(
            coefficient.phases[0].multiplicity,
            coefficient.phases[1].multiplicity,
        )

    def test_voderberg_profile_has_no_exact_translation_obstruction(self):
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
        state = next(
            state
            for state, _ in base.enumerate_terminal_states(case, max_depth=10, max_states=1000)
            if base.word_to_text(state.environment_map()["A0"]) == "V0"
            and base.word_to_text(state.environment_map()["A1"]) == "V1 V1^-1 V0^-1"
        )
        angle_solution = turning.complete_angle_solution(case, state)
        translation = se2.analyze_translation_closure(case, state, angle_solution)
        self.assertFalse(translation.exact_obstruction)
        self.assertEqual({item.variable for item in translation.coefficients}, {"V0", "V1"})

    def test_real_one_parameter_profile_is_rejected_when_all_phases_match(self):
        found = None
        for case in base.enumerate_placement_cases():
            for state, _ in base.enumerate_terminal_states(case, max_depth=4, max_states=50):
                solution = turning.complete_angle_solution(case, state)
                analysis = se2.analyze_translation_closure(case, state, solution)
                if analysis.exact_obstruction and len(analysis.coefficients) == 1:
                    found = analysis
                    break
            if found is not None:
                break
        self.assertIsNotNone(found)


if __name__ == "__main__":
    unittest.main()
