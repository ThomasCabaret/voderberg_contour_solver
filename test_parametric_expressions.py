import unittest

import parametric_expressions as expr
import symbolic_enumerator as base


class ParametricExpressionTests(unittest.TestCase):
    def test_nested_repeat_expands_with_independent_exponents(self):
        expression = expr.concat(
            expr.atom("A"),
            expr.Repeat(
                expr.concat(expr.atom("B"), expr.Repeat(expr.atom("C"), "n", minimum=1)),
                "k",
                minimum=0,
            ),
        )
        expanded = expr.expand(expression, {"n": 2, "k": 2})
        self.assertEqual(base.word_to_text(expanded), "A B C C B C C")
        self.assertEqual(expr.repeat_nesting_depth(expression), 2)
        self.assertEqual(expr.exponent_parameters(expression), ("k", "n"))

    def test_inverse_reverses_concat_and_preserves_repeat_parameter(self):
        expression = expr.Repeat(
            expr.concat(expr.atom("A"), expr.atom("B", inverse=True)),
            "n",
            minimum=1,
        )
        inverse = expr.inverse(expression)
        self.assertEqual(expr.to_text(inverse), "(B A^-1)^n")
        self.assertEqual(expr.expand(inverse, {"n": 1}), (
            base.Literal("B"),
            base.Literal("A", True),
        ))

    def test_unmapped_inverse_atom_is_not_double_inverted(self):
        value = expr.atom("X", inverse=True)
        self.assertEqual(expr.substitute(value, {}), value)


if __name__ == "__main__":
    unittest.main()
