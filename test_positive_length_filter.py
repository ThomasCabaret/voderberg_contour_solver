import unittest

import positive_length_filter as length_filter
import symbolic_enumerator as base


def literal(name: str, inverse: bool = False) -> base.Literal:
    return base.Literal(name, inverse)


class PositiveLengthFilterTests(unittest.TestCase):
    def test_rejects_variable_forced_to_empty_length(self):
        equations = (
            base.Equation((literal("A"), literal("B")), (literal("A"),)),
        )
        result = length_filter.analyze_equations(equations)
        self.assertFalse(result.feasible)
        self.assertIsNone(result.positive_integer_witness)

    def test_rejects_cross_equation_length_contradiction(self):
        equations = (
            base.Equation(
                (literal("A"), literal("B")),
                (literal("C", True), literal("B", True)),
            ),
            base.Equation(
                (literal("C"),),
                (literal("C", True), literal("B", True)),
            ),
        )
        result = length_filter.analyze_equations(equations)
        self.assertFalse(result.feasible)

    def test_accepts_and_returns_positive_integer_witness(self):
        equations = (
            base.Equation((literal("A"), literal("B")), (literal("C"),)),
            base.Equation((literal("C"),), (literal("B"), literal("A"))),
        )
        result = length_filter.analyze_equations(equations)
        self.assertTrue(result.feasible)
        witness = dict(result.positive_integer_witness or ())
        self.assertTrue(witness)
        self.assertTrue(all(value > 0 for value in witness.values()))
        for row in result.balance_matrix:
            self.assertEqual(
                sum(coefficient * witness[variable] for coefficient, variable in zip(row, result.variable_order)),
                0,
            )

    def test_generated_case_counts(self):
        cases = list(base.enumerate_placement_cases())
        infeasible = [case for case in cases if not length_filter.analyze_case(case).feasible]
        initially_inconsistent = [case for case in cases if base.initial_solver_state(case) is None]
        additional = [case for case in infeasible if base.initial_solver_state(case) is not None]
        self.assertEqual(len(cases), 2816)
        self.assertEqual(len(infeasible), 2184)
        self.assertEqual(len(initially_inconsistent), 544)
        self.assertEqual(len(additional), 1640)
        self.assertEqual(len(cases) - len(infeasible), 632)


if __name__ == "__main__":
    unittest.main()
