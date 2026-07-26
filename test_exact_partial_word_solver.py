import unittest

import exact_partial_word_solver as exact
import family_representative_expansion as expansion
import parametric_expressions as expr
import symbolic_enumerator as base


class ExactPartialWordSolverTests(unittest.TestCase):
    def solve(self, case_id, **overrides):
        options = dict(
            max_nodes=500,
            max_edges=3000,
            max_families=10000,
            representative_exponent_value=1,
        )
        options.update(overrides)
        return exact.solve_case(base.find_case(case_id), **options)

    def test_complete_finite_case_is_exact_and_expands_without_a_policy(self):
        result = self.solve(34)
        self.assertEqual(result.status, exact.EXACT_FINITE)
        self.assertTrue(result.graph_complete)
        self.assertTrue(result.families)
        case = base.find_case(34)
        for family in result.families:
            self.assertFalse(family.parametric)
            expanded = expansion.expand_family(
                dict(family.environment),
                dict(family.exponent_minimums),
                policy=expansion.ExpansionPolicy(kind=expansion.POLICY_NONE),
            )
            self.assertEqual(len(expanded), 1)
            self.assertTrue(
                base.terminal_state_satisfies_case(case, expanded[0].state)
            )
            self.assertNotIn("representative_environment", family.to_dict())

    def test_simple_power_family_is_retained_exactly(self):
        result = self.solve(95)
        self.assertEqual(result.status, exact.EXACT_POWER)
        self.assertTrue(any(family.exponent_minimums for family in result.families))
        self.assertTrue(any("^n" in expr.to_text(family.a_expression) for family in result.families))

    def test_commutation_family_uses_two_dynamic_exponents(self):
        result = self.solve(352)
        self.assertEqual(result.status, exact.EXACT_NESTED_POWER)
        self.assertEqual(len(result.families), 1)
        minimums = dict(result.families[0].exponent_minimums)
        self.assertEqual(sorted(minimums.values()), [1, 1])


    def test_power_and_nested_power_formulas_validate_beyond_exponent_one(self):
        for case_id in (95, 569):
            case = base.find_case(case_id)
            result = self.solve(case_id)
            for family in result.families:
                environment = dict(family.environment)
                minimums = dict(family.exponent_minimums)
                for requested in (0, 1, 2, 3):
                    state, _assignment = __import__(
                        "family_representative_expansion"
                    ).expand_environment(
                        environment, minimums, requested_value=requested
                    )
                    self.assertTrue(
                        base.terminal_state_satisfies_case(case, state),
                        (case_id, family.family_id, requested),
                    )

    def test_complete_but_unsupported_graph_is_parked_without_representative(self):
        result = self.solve(204)
        self.assertEqual(result.status, exact.EXACT_GRAPH_UNSUPPORTED)
        self.assertTrue(result.graph_complete)
        self.assertFalse(result.families)
        self.assertTrue(result.unsupported_families)
        self.assertTrue(result.unsupported_reasons)

    def test_unsupported_branch_does_not_erase_supported_branches(self):
        result = self.solve(1320)
        self.assertEqual(
            result.status,
            exact.EXACT_MIXED_SUPPORTED_AND_UNSUPPORTED,
        )
        self.assertTrue(result.families)
        self.assertTrue(result.unsupported_families)
        self.assertFalse(result.complete_family_language_compiled)

    def test_finite_specialization_is_removed_from_primitive_output(self):
        finite = exact.ExactFormalFamily(
            family_id=0,
            kind=exact.EXACT_FINITE,
            environment=(("X", expr.concat(expr.atom("W"), expr.atom("W"))),),
            a_expression=expr.atom("W"),
            b_expression=expr.atom("W"),
            exponent_minimums=(),
            trace=(),
        )
        parametric = exact.ExactFormalFamily(
            family_id=1,
            kind=exact.EXACT_POWER,
            environment=(("X", expr.Repeat(expr.atom("W"), "n", minimum=0)),),
            a_expression=expr.atom("W"),
            b_expression=expr.atom("W"),
            exponent_minimums=(("n", 0),),
            trace=(),
        )
        retained, suppressed = exact._remove_parametric_specializations(
            [finite, parametric]
        )
        self.assertEqual(suppressed, 1)
        self.assertEqual(len(retained), 1)
        self.assertTrue(retained[0].parametric)


    def test_graph_limit_is_unresolved_not_false_unsat(self):
        result = self.solve(95, max_nodes=1, max_edges=1)
        self.assertEqual(result.status, exact.UNRESOLVED_GRAPH_LIMIT)
        self.assertFalse(result.graph_complete)
        self.assertFalse(result.families)


if __name__ == "__main__":
    unittest.main()
