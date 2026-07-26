import unittest

import family_representative_expansion as expansion
import parametric_expressions as expr
import symbolic_enumerator as base


class FamilyRepresentativeExpansionTests(unittest.TestCase):
    def test_none_policy_keeps_parametric_family_symbolic(self):
        environment = {"X": expr.Repeat(expr.atom("W"), "n", minimum=0)}
        expanded = expansion.expand_family(
            environment,
            {"n": 0},
            policy=expansion.ExpansionPolicy(kind=expansion.POLICY_NONE),
        )
        self.assertEqual(expanded, ())

    def test_none_policy_still_materializes_genuinely_finite_family(self):
        expanded = expansion.expand_family(
            {"X": expr.concat(expr.atom("U"), expr.atom("V"))},
            {},
            policy=expansion.ExpansionPolicy(kind=expansion.POLICY_NONE),
        )
        self.assertEqual(len(expanded), 1)
        self.assertEqual(
            base.word_to_text(dict(expanded[0].state.environment)["X"]),
            "U V",
        )

    def test_range_policy_enumerates_nested_parameter_assignments(self):
        environment = {
            "X": expr.Repeat(
                expr.Repeat(expr.atom("W"), "m", minimum=0),
                "n",
                minimum=1,
            )
        }
        expanded = expansion.expand_family(
            environment,
            {"m": 0, "n": 1},
            policy=expansion.ExpansionPolicy(
                kind=expansion.POLICY_RANGE,
                maximum_exponent=2,
                max_specializations=10,
            ),
        )
        self.assertEqual(
            [item.assignment_map() for item in expanded],
            [
                {"m": 0, "n": 1},
                {"m": 0, "n": 2},
                {"m": 1, "n": 1},
                {"m": 1, "n": 2},
                {"m": 2, "n": 1},
                {"m": 2, "n": 2},
            ],
        )

    def test_requested_one_respects_positive_minimum(self):
        environment = {
            "X": expr.Repeat(expr.atom("W"), "n", minimum=2),
            "Y": expr.Repeat(expr.atom("Z"), "k", minimum=0),
        }
        state, assignment = expansion.expand_environment(
            environment,
            {"n": 2, "k": 0},
            requested_value=1,
        )
        self.assertEqual(assignment, {"n": 2, "k": 1})
        values = dict(state.environment)
        self.assertEqual(base.word_to_text(values["X"]), "W W")
        self.assertEqual(base.word_to_text(values["Y"]), "Z")

    def test_expansion_is_independent_from_exact_ast(self):
        original = expr.Repeat(expr.atom("W"), "n", minimum=0)
        state, _ = expansion.expand_environment({"X": original}, {"n": 0}, requested_value=1)
        self.assertEqual(expr.to_text(original), "W^n")
        self.assertEqual(base.word_to_text(dict(state.environment)["X"]), "W")

    def test_nested_length_cap_does_not_expand_large_word(self):
        expression = expr.Repeat(
            expr.Repeat(expr.atom("W"), "m", minimum=0),
            "n",
            minimum=0,
        )
        self.assertEqual(
            expr.expanded_length(expression, {"m": 1000, "n": 1000}, cap=20),
            21,
        )


if __name__ == "__main__":
    unittest.main()
