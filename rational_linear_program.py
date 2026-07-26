#!/usr/bin/env python3
"""Small exact rational simplex solver.

The solver maximizes ``c.x`` subject to ``A.x <= b`` and ``x >= 0``.  It is a
Fraction-based transcription of the standard two-phase tableau algorithm.  It
is intended for the tiny exact feasibility problems used by the contour
filters, not as a general large-scale optimization package.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class LinearProgramResult:
    status: str  # optimal, infeasible, unbounded
    optimum: Optional[Fraction]
    solution: Tuple[Fraction, ...]


class RationalSimplex:
    def __init__(
        self,
        A: Sequence[Sequence[int | Fraction]],
        b: Sequence[int | Fraction],
        c: Sequence[int | Fraction],
    ) -> None:
        if len(A) != len(b):
            raise ValueError("A and b row counts differ")
        self.m = len(b)
        self.n = len(c)
        if any(len(row) != self.n for row in A):
            raise ValueError("A row width differs from c")

        self.B = [self.n + index for index in range(self.m)]
        self.N = [index for index in range(self.n)] + [-1]
        self.D = [
            [Fraction(0) for _ in range(self.n + 2)]
            for _ in range(self.m + 2)
        ]
        for i in range(self.m):
            for j in range(self.n):
                self.D[i][j] = Fraction(A[i][j])
            self.D[i][self.n] = Fraction(-1)
            self.D[i][self.n + 1] = Fraction(b[i])
        for j in range(self.n):
            self.D[self.m][j] = -Fraction(c[j])
        self.D[self.m + 1][self.n] = Fraction(1)

    def _pivot(self, row: int, column: int) -> None:
        inverse = Fraction(1, 1) / self.D[row][column]
        for i in range(self.m + 2):
            if i == row:
                continue
            for j in range(self.n + 2):
                if j == column:
                    continue
                self.D[i][j] -= (
                    self.D[row][j] * self.D[i][column] * inverse
                )
        for j in range(self.n + 2):
            if j != column:
                self.D[row][j] *= inverse
        for i in range(self.m + 2):
            if i != row:
                self.D[i][column] *= -inverse
        self.D[row][column] = inverse
        self.B[row], self.N[column] = self.N[column], self.B[row]

    def _simplex(self, phase: int) -> bool:
        objective_row = self.m + 1 if phase == 1 else self.m
        while True:
            candidates = [
                column
                for column in range(self.n + 1)
                if not (phase == 2 and self.N[column] == -1)
            ]
            entering = min(
                candidates,
                key=lambda column: (self.D[objective_row][column], self.N[column]),
            )
            if self.D[objective_row][entering] >= 0:
                return True

            leaving_candidates = [
                row
                for row in range(self.m)
                if self.D[row][entering] > 0
            ]
            if not leaving_candidates:
                return False
            leaving = min(
                leaving_candidates,
                key=lambda row: (
                    self.D[row][self.n + 1] / self.D[row][entering],
                    self.B[row],
                ),
            )
            self._pivot(leaving, entering)

    def solve(self) -> LinearProgramResult:
        if self.m == 0:
            if any(self.D[self.m][column] < 0 for column in range(self.n)):
                return LinearProgramResult("unbounded", None, ())
            return LinearProgramResult(
                "optimal", Fraction(0), tuple(Fraction(0) for _ in range(self.n))
            )

        row = min(range(self.m), key=lambda index: self.D[index][self.n + 1])
        if self.D[row][self.n + 1] < 0:
            self._pivot(row, self.n)
            if not self._simplex(1) or self.D[self.m + 1][self.n + 1] < 0:
                return LinearProgramResult("infeasible", None, ())
            if self.D[self.m + 1][self.n + 1] != 0:
                return LinearProgramResult("infeasible", None, ())
            artificial_row = next(
                (index for index in range(self.m) if self.B[index] == -1),
                None,
            )
            if artificial_row is not None:
                entering = min(
                    range(self.n + 1),
                    key=lambda column: (
                        self.D[artificial_row][column], self.N[column]
                    ),
                )
                if self.D[artificial_row][entering] != 0:
                    self._pivot(artificial_row, entering)

        if not self._simplex(2):
            return LinearProgramResult("unbounded", None, ())

        solution = [Fraction(0) for _ in range(self.n)]
        for row in range(self.m):
            if 0 <= self.B[row] < self.n:
                solution[self.B[row]] = self.D[row][self.n + 1]
        return LinearProgramResult(
            "optimal",
            self.D[self.m][self.n + 1],
            tuple(solution),
        )


def maximize_free_variables(
    inequalities: Sequence[Tuple[Sequence[int | Fraction], int | Fraction]],
    objective: Sequence[int | Fraction],
) -> LinearProgramResult:
    """Maximize over unrestricted variables by splitting x = x_pos - x_neg."""
    width = len(objective)
    if any(len(coefficients) != width for coefficients, _rhs in inequalities):
        raise ValueError("Inequality width mismatch")
    expanded_A = []
    expanded_b = []
    for coefficients, rhs in inequalities:
        row = [Fraction(value) for value in coefficients]
        expanded_A.append(row + [-value for value in row])
        expanded_b.append(Fraction(rhs))
    c = [Fraction(value) for value in objective]
    expanded_c = c + [-value for value in c]
    result = RationalSimplex(expanded_A, expanded_b, expanded_c).solve()
    if result.status != "optimal":
        return result
    assert result.optimum is not None
    positive = result.solution[:width]
    negative = result.solution[width:]
    original = tuple(left - right for left, right in zip(positive, negative))
    return LinearProgramResult(result.status, result.optimum, original)
