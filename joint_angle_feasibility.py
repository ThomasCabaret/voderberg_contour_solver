#!/usr/bin/env python3
"""Exact joint feasibility of every linear angular constraint.

The historical pipeline checked three angular subsystems separately:

* total turning of the reference boundary;
* the two local pole inequalities;
* the joint inner/outer turning equations.

Separate feasibility is weaker than simultaneous feasibility.  This module
solves the shared system in one exact rational linear program.  Point-angle
classes Theta live in the open interval (-pi, pi); curve-turn classes Kappa are
unbounded.  After division by pi, the model is

    rotation equations = prescribed winding values,
    pole_0(Theta) >= 1,
    pole_1(Theta) >= 1,
    -1 < Theta_i < 1.

Strict bounds are handled by maximizing a common rational margin delta:

    -1 + delta <= Theta_i <= 1 - delta.

The original open system is feasible exactly when the optimum has delta > 0.
All arithmetic is performed by SymPy's exact simplex implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Mapping, Optional, Sequence, Tuple

import external_boundary_constraints as external
import pole_angle_filter as poles
import rational_linear_program as rational_lp


SCHEMA_VERSION = "joint-angle-feasibility-v1"


@dataclass(frozen=True)
class RationalValue:
    numerator: int
    denominator: int

    @staticmethod
    def from_value(value: object) -> "RationalValue":
        numerator = int(getattr(value, "p", getattr(value, "numerator", value)))
        denominator = int(getattr(value, "q", getattr(value, "denominator", 1)))
        return RationalValue(numerator, denominator)

    def to_dict(self) -> Dict[str, object]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "decimal": self.numerator / self.denominator,
        }


@dataclass(frozen=True)
class JointAngleFeasibility:
    feasible: bool
    status: str
    discard_reason: Optional[str]
    strict_margin: RationalValue
    theta_variables: Tuple[str, ...]
    kappa_variables: Tuple[str, ...]
    rotation_equation_count: int
    pole_inequality_count: int
    witness: Tuple[Tuple[str, RationalValue], ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "feasible": self.feasible,
            "status": self.status,
            "discard_reason": self.discard_reason,
            "strict_margin_pi_units": self.strict_margin.to_dict(),
            "theta_variables": list(self.theta_variables),
            "kappa_variables": list(self.kappa_variables),
            "rotation_equation_count": self.rotation_equation_count,
            "pole_inequality_count": self.pole_inequality_count,
            "witness": {
                name: value.to_dict() for name, value in self.witness
            },
            "interpretation": (
                "All inner/outer winding equations, both pole inequalities, and "
                "every principal point-angle bound were solved simultaneously "
                "with exact rational arithmetic."
            ),
        }


def _is_kappa(name: str) -> bool:
    return name.startswith("Kappa[") or name.startswith("KappaClass")


def _coefficient_maps(
    rotation_equations: Sequence[external.RotationEquation],
) -> Tuple[Dict[str, Fraction], ...]:
    return tuple(equation.normalized_coefficients() for equation in rotation_equations)



def _independent_equalities(
    coefficient_maps: Sequence[Mapping[str, Fraction]],
    rhs_values: Sequence[Fraction],
    variable_names: Sequence[str],
) -> Tuple[Tuple[Tuple[Fraction, ...], Fraction], ...]:
    """Return an exact row-echelon basis and detect inconsistent rows."""
    rows = [
        [mapping.get(name, Fraction(0)) for name in variable_names]
        + [Fraction(rhs)]
        for mapping, rhs in zip(coefficient_maps, rhs_values)
    ]
    pivot_row = 0
    for column in range(len(variable_names)):
        pivot = next(
            (index for index in range(pivot_row, len(rows)) if rows[index][column]),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        value = rows[pivot_row][column]
        rows[pivot_row] = [item / value for item in rows[pivot_row]]
        for index in range(len(rows)):
            if index == pivot_row or rows[index][column] == 0:
                continue
            factor = rows[index][column]
            rows[index] = [
                left - factor * right
                for left, right in zip(rows[index], rows[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == len(rows):
            break

    independent = []
    for row in rows:
        coefficients = tuple(row[:-1])
        rhs = row[-1]
        if not any(coefficients):
            if rhs != 0:
                raise ValueError("inconsistent rotation equalities")
            continue
        independent.append((coefficients, rhs))
    return tuple(independent)

def analyze_joint_angle_feasibility(
    rotation_equations: Sequence[external.RotationEquation],
    pole_analysis: poles.PoleAngleAnalysis,
) -> JointAngleFeasibility:
    """Solve the complete linear angular model exactly.

    The function is deliberately independent from the audit pipeline.  It only
    consumes already constructed rotation equations and pole coefficient maps.
    """
    coefficient_maps = _coefficient_maps(rotation_equations)
    pole_maps = tuple(item.coefficient_map() for item in pole_analysis.constraints)
    variable_names = sorted(
        {
            name
            for mapping in (*coefficient_maps, *pole_maps)
            for name in mapping
        }
    )
    theta_names = tuple(name for name in variable_names if not _is_kappa(name))
    kappa_names = tuple(name for name in variable_names if _is_kappa(name))
    index = {name: position for position, name in enumerate(variable_names)}
    delta_index = len(variable_names)
    width = delta_index + 1

    def row_from_mapping(mapping: Mapping[str, int | Fraction]) -> list[Fraction]:
        row = [Fraction(0) for _ in range(width)]
        for name, value in mapping.items():
            row[index[name]] = Fraction(value)
        return row

    inequalities: list[tuple[list[Fraction], Fraction]] = []

    # -1 + delta <= Theta <= 1 - delta.
    for theta_name in theta_names:
        upper = [Fraction(0) for _ in range(width)]
        upper[index[theta_name]] = Fraction(1)
        upper[delta_index] = Fraction(1)
        inequalities.append((upper, Fraction(1)))

        lower = [Fraction(0) for _ in range(width)]
        lower[index[theta_name]] = Fraction(-1)
        lower[delta_index] = Fraction(1)
        inequalities.append((lower, Fraction(1)))

    # Equalities are represented by their two exact half-spaces.  The rational
    # simplex handles dependent equations without the SymPy simplex bug that
    # motivated this standalone backend.
    for equation, mapping in zip(rotation_equations, coefficient_maps):
        row = row_from_mapping(mapping)
        rhs = Fraction(equation.normalized_rhs())
        inequalities.append((row, rhs))
        inequalities.append(([-value for value in row], -rhs))

    for mapping in pole_maps:
        row = row_from_mapping(mapping)
        inequalities.append(([-value for value in row], Fraction(-1)))

    delta_lower = [Fraction(0) for _ in range(width)]
    delta_lower[delta_index] = Fraction(-1)
    inequalities.append((delta_lower, Fraction(0)))
    delta_upper = [Fraction(0) for _ in range(width)]
    delta_upper[delta_index] = Fraction(1)
    inequalities.append((delta_upper, Fraction(1)))

    objective = [Fraction(0) for _ in range(width)]
    objective[delta_index] = Fraction(1)
    result = rational_lp.maximize_free_variables(inequalities, objective)

    if result.status == "infeasible":
        return JointAngleFeasibility(
            feasible=False,
            status="infeasible_closed_linear_system",
            discard_reason=(
                "The shared inner/outer winding equations and the two pole "
                "inequalities are mutually inconsistent, even before enforcing "
                "strict principal-angle bounds."
            ),
            strict_margin=RationalValue(0, 1),
            theta_variables=theta_names,
            kappa_variables=kappa_names,
            rotation_equation_count=len(rotation_equations),
            pole_inequality_count=len(pole_maps),
            witness=(),
        )
    if result.status != "optimal" or result.optimum is None:
        raise RuntimeError(f"Unexpected joint-angle LP status: {result.status}")

    optimum = result.optimum
    margin = RationalValue(optimum.numerator, optimum.denominator)
    witness_items = tuple(
        (
            name,
            RationalValue(
                result.solution[index[name]].numerator,
                result.solution[index[name]].denominator,
            ),
        )
        for name in variable_names
    )

    feasible = optimum > 0
    if feasible:
        status = "feasible_with_strict_margin"
        reason = None
    else:
        status = "only_principal_angle_boundary_feasible"
        reason = (
            "The complete angular system can be satisfied only when at least one "
            "point-angle class reaches +/-pi.  Such a cusp or instantaneous U-turn "
            "is excluded by the contour model."
        )

    return JointAngleFeasibility(
        feasible=feasible,
        status=status,
        discard_reason=reason,
        strict_margin=margin,
        theta_variables=theta_names,
        kappa_variables=kappa_names,
        rotation_equation_count=len(rotation_equations),
        pole_inequality_count=len(pole_maps),
        witness=witness_items,
    )
