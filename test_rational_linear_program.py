import unittest
from fractions import Fraction

from rational_linear_program import maximize_free_variables


class RationalLinearProgramTests(unittest.TestCase):
    def test_bounded_optimum(self):
        # max x subject to -1 <= x <= 2
        result = maximize_free_variables(
            [([1], 2), ([-1], 1)],
            [1],
        )
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.optimum, Fraction(2))
        self.assertEqual(result.solution, (Fraction(2),))

    def test_infeasible(self):
        result = maximize_free_variables(
            [([1], 0), ([-1], -1)],
            [0],
        )
        self.assertEqual(result.status, "infeasible")

    def test_redundant_equalities_encoded_as_inequalities(self):
        # x+y=2, |x|,|y| <= 1-delta, maximize delta.
        # The only solution is x=y=1, hence delta=0.
        inequalities = [
            ([1, 1, 0], 2),
            ([-1, -1, 0], -2),
            ([3, 3, 0], 6),
            ([-3, -3, 0], -6),
            ([1, 0, 1], 1),
            ([-1, 0, 1], 1),
            ([0, 1, 1], 1),
            ([0, -1, 1], 1),
            ([0, 0, -1], 0),
            ([0, 0, 1], 1),
        ]
        result = maximize_free_variables(inequalities, [0, 0, 1])
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.optimum, Fraction(0))


if __name__ == "__main__":
    unittest.main()
