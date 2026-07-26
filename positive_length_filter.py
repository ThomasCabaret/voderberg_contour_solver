"""Exact positive word-length feasibility for the generated two-equation systems.

Every formal variable denotes a nonempty word, hence has a strictly positive
integer length.  Equality of words implies equality of their lengths.  This
module checks those necessary linear equalities before the bounded Nielsen/Levi
search is started.

The generated placement systems contain exactly two equations.  For two
homogeneous equations, strict positive feasibility can be decided exactly with
integer arithmetic by a two-dimensional cone test.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Dict, Mapping, Optional, Sequence, Tuple

import symbolic_enumerator as base


SCHEMA_VERSION = "positive-word-length-v1"

Vector2 = Tuple[int, int]


@dataclass(frozen=True)
class PositiveLengthAnalysis:
    variable_order: Tuple[str, ...]
    balance_matrix: Tuple[Tuple[int, ...], ...]
    feasible: bool
    positive_integer_witness: Optional[Tuple[Tuple[str, int], ...]]
    reason: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "feasible" if self.feasible else "exact_unsat",
            "feasible": self.feasible,
            "strictly_positive_lengths_required": True,
            "inverse_has_same_length": True,
            "variable_order": list(self.variable_order),
            "balance_matrix": [list(row) for row in self.balance_matrix],
            "positive_integer_length_witness": (
                dict(self.positive_integer_witness)
                if self.positive_integer_witness is not None
                else None
            ),
            "reason": self.reason,
            "automatic_rejection_is_sound": True,
        }


def _cross(left: Vector2, right: Vector2) -> int:
    return left[0] * right[1] - left[1] * right[0]


def _same_ray_coefficient(generator: Vector2, target: Vector2) -> Optional[Fraction]:
    if generator == (0, 0) or _cross(generator, target) != 0:
        return None
    coefficient = (
        Fraction(target[0], generator[0])
        if generator[0] != 0
        else Fraction(target[1], generator[1])
    )
    return coefficient if coefficient >= 0 else None


def _cone_representation(
    columns: Sequence[Vector2], target: Vector2
) -> Optional[Dict[int, Fraction]]:
    """Represent target as a nonnegative combination of at most two columns."""
    if target == (0, 0):
        return {}

    nonzero = [(index, column) for index, column in enumerate(columns) if column != (0, 0)]

    for index, column in nonzero:
        coefficient = _same_ray_coefficient(column, target)
        if coefficient is not None:
            return {index: coefficient}

    for left_index in range(len(nonzero)):
        index_a, column_a = nonzero[left_index]
        for right_index in range(left_index + 1, len(nonzero)):
            index_b, column_b = nonzero[right_index]
            determinant = _cross(column_a, column_b)
            if determinant == 0:
                continue
            coefficient_a = Fraction(_cross(target, column_b), determinant)
            coefficient_b = Fraction(_cross(column_a, target), determinant)
            if coefficient_a >= 0 and coefficient_b >= 0:
                return {
                    index_a: coefficient_a,
                    index_b: coefficient_b,
                }
    return None


def _integer_witness(
    variable_order: Sequence[str],
    cone_coefficients: Mapping[int, Fraction],
) -> Tuple[Tuple[str, int], ...]:
    lengths = [Fraction(1) for _ in variable_order]
    for index, coefficient in cone_coefficients.items():
        lengths[index] += coefficient

    common_denominator = 1
    for value in lengths:
        common_denominator = (
            common_denominator * value.denominator
            // gcd(common_denominator, value.denominator)
        )
    integer_lengths = [int(value * common_denominator) for value in lengths]
    common_divisor = 0
    for value in integer_lengths:
        common_divisor = gcd(common_divisor, value)
    if common_divisor > 1:
        integer_lengths = [value // common_divisor for value in integer_lengths]
    return tuple(zip(variable_order, integer_lengths))


def analyze_equations(equations: Sequence[base.Equation]) -> PositiveLengthAnalysis:
    if len(equations) > 2:
        raise ValueError(
            "The exact cone implementation currently supports at most two word equations."
        )

    variable_order = tuple(sorted({
        literal.variable
        for equation in equations
        for literal in equation.left + equation.right
    }))
    variable_index = {variable: index for index, variable in enumerate(variable_order)}

    rows = []
    for equation in equations:
        row = [0] * len(variable_order)
        for literal in equation.left:
            row[variable_index[literal.variable]] += 1
        for literal in equation.right:
            row[variable_index[literal.variable]] -= 1
        rows.append(tuple(row))
    while len(rows) < 2:
        rows.append(tuple(0 for _ in variable_order))
    balance_matrix = tuple(rows)

    if not variable_order:
        return PositiveLengthAnalysis(
            variable_order=variable_order,
            balance_matrix=balance_matrix,
            feasible=True,
            positive_integer_witness=(),
            reason=None,
        )

    columns = tuple(
        (balance_matrix[0][index], balance_matrix[1][index])
        for index in range(len(variable_order))
    )

    # If positive lengths l solve M l = 0, homogeneity lets us rescale them so
    # every l_i >= 1.  Writing l_i = 1 + u_i reduces the question to whether
    # -M(1,...,1) lies in the nonnegative cone generated by the columns of M.
    target = (
        -sum(column[0] for column in columns),
        -sum(column[1] for column in columns),
    )
    cone_coefficients = _cone_representation(columns, target)
    if cone_coefficients is None:
        return PositiveLengthAnalysis(
            variable_order=variable_order,
            balance_matrix=balance_matrix,
            feasible=False,
            positive_integer_witness=None,
            reason=(
                "The word-length equalities have no solution in which every "
                "formal variable has strictly positive length."
            ),
        )

    witness = _integer_witness(variable_order, cone_coefficients)
    witness_map = dict(witness)
    for row in balance_matrix:
        if sum(coefficient * witness_map[variable] for coefficient, variable in zip(row, variable_order)) != 0:
            raise AssertionError("Internal error: invalid positive-length witness")
    if any(length <= 0 for _, length in witness):
        raise AssertionError("Internal error: nonpositive length in witness")

    return PositiveLengthAnalysis(
        variable_order=variable_order,
        balance_matrix=balance_matrix,
        feasible=True,
        positive_integer_witness=witness,
        reason=None,
    )


def analyze_case(case: base.PlacementCase) -> PositiveLengthAnalysis:
    return analyze_equations(case.equations)
