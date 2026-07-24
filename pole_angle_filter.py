#!/usr/bin/env python3
"""
Local pole-contact angle filter for symbolic contour solutions.

At each pole, three copies of the tile meet:

- the reference tile;
- the copy covering A;
- the copy covering B.

For a counterclockwise prototype boundary, let tau(Q) be the signed turning
angle at prototype point Q, with tau=0 meaning straight continuation.  The
physical interior angle is

    alpha(Q) = pi - tau(Q).

At P0 the three prototype points are

    P0, A_start, B_end,

and at P1 they are

    P1, A_end, B_start.

Non-overlap of the three local interior sectors requires

    alpha(q1) + alpha(q2) + alpha(q3) <= 2*pi,

which is equivalent to

    tau(q1) + tau(q2) + tau(q3) >= pi.

After the point-angle equivalence solver, every tau is 0 or +/-Theta_i with
Theta_i in (-pi, pi).  Dividing by pi gives two homogeneous inequalities

    c0.x >= 1,  c1.x >= 1,  x_i in (-1, 1).

The exact joint capacity is

    max_{x in [-1,1]^n} min(c0.x, c1.x)
      = min_{lambda in [0,1]} ||lambda*c0 + (1-lambda)*c1||_1.

The right side is a convex piecewise-linear function.  Its minimum is found
exactly at an endpoint or at a zero crossing of one affine coordinate.  The
strict open-box system is feasible exactly when this capacity is strictly
larger than 1.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import angle_constraints as angles
import symbolic_enumerator as base
import turning_filter as turning


@dataclass(frozen=True)
class IntegerCoefficient:
    variable: str
    coefficient: int


@dataclass(frozen=True)
class PoleConstraint:
    pole: str
    contact_points: Tuple[str, str, str]
    turn_coefficients: Tuple[IntegerCoefficient, ...]
    individual_capacity_pi_units: int

    def coefficient_map(self) -> Dict[str, int]:
        return {
            item.variable: item.coefficient
            for item in self.turn_coefficients
        }

    def inequality_text(self) -> str:
        terms = []
        for item in self.turn_coefficients:
            coefficient = item.coefficient
            variable = item.variable
            if coefficient == 1:
                term = variable
            elif coefficient == -1:
                term = f"-{variable}"
            else:
                term = f"{coefficient}*{variable}"
            terms.append(term)
        if not terms:
            left = "0"
        else:
            left = terms[0]
            for term in terms[1:]:
                left += " - " + term[1:] if term.startswith("-") else " + " + term
        return f"{left} >= pi"


@dataclass(frozen=True)
class PoleAngleAnalysis:
    constraints: Tuple[PoleConstraint, PoleConstraint]
    joint_capacity_pi_units: Fraction
    feasible: bool
    discard_reason: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "convention": {
                "prototype_orientation": "counterclockwise",
                "signed_turn": "tau",
                "interior_angle": "alpha = pi - tau",
                "regular_turn_domain": "tau in (-pi, pi)",
                "pole_condition": "sum of the three physical interior angles <= 2*pi",
            },
            "constraints": [
                {
                    "pole": constraint.pole,
                    "contact_points": list(constraint.contact_points),
                    "turn_coefficients": constraint.coefficient_map(),
                    "individual_capacity_pi_units": constraint.individual_capacity_pi_units,
                    "inequality": constraint.inequality_text(),
                }
                for constraint in self.constraints
            ],
            "joint_capacity_pi_units": {
                "numerator": self.joint_capacity_pi_units.numerator,
                "denominator": self.joint_capacity_pi_units.denominator,
                "decimal": float(self.joint_capacity_pi_units),
            },
            "feasible": self.feasible,
            "discard_reason": self.discard_reason,
        }


def _sorted_coefficients(values: Mapping[str, int]) -> Tuple[IntegerCoefficient, ...]:
    return tuple(
        IntegerCoefficient(variable, coefficient)
        for variable, coefficient in sorted(values.items())
        if coefficient != 0
    )


def marker_point(case: base.PlacementCase, marker: str) -> str:
    boundaries = angles.contour_boundary_points(case)
    boundary = case.marker_boundary_map()[marker]
    return boundaries[boundary % len(boundaries)]


def pole_contact_points(case: base.PlacementCase) -> Dict[str, Tuple[str, str, str]]:
    """Return prototype points contributed by the three physical tile copies."""
    return {
        "P0": (
            "P0",
            marker_point(case, "A_start"),
            marker_point(case, "B_end"),
        ),
        "P1": (
            "P1",
            marker_point(case, "A_end"),
            marker_point(case, "B_start"),
        ),
    }


def _assignment_term(expression: str) -> Optional[Tuple[str, int]]:
    if expression == "0":
        return None
    if expression.startswith("-"):
        return expression[1:], -1
    return expression, 1


def constraint_from_contacts(
    pole: str,
    contacts: Tuple[str, str, str],
    assignment_map: Mapping[str, str],
) -> PoleConstraint:
    coefficients: Dict[str, int] = {}
    for point in contacts:
        term = _assignment_term(assignment_map[point])
        if term is None:
            continue
        variable, sign = term
        coefficients[variable] = coefficients.get(variable, 0) + sign

    coefficients = {
        variable: coefficient
        for variable, coefficient in coefficients.items()
        if coefficient != 0
    }
    return PoleConstraint(
        pole=pole,
        contact_points=contacts,
        turn_coefficients=_sorted_coefficients(coefficients),
        individual_capacity_pi_units=sum(abs(value) for value in coefficients.values()),
    )


def joint_capacity(
    first: Mapping[str, int],
    second: Mapping[str, int],
) -> Fraction:
    """Compute max_x min(first.x, second.x) over the closed unit box exactly."""
    variables = sorted(set(first) | set(second))
    if not variables:
        return Fraction(0)

    candidates = {Fraction(0), Fraction(1)}
    for variable in variables:
        c = first.get(variable, 0)
        d = second.get(variable, 0)
        denominator = c - d
        if denominator == 0:
            continue
        crossing = Fraction(-d, denominator)
        if 0 <= crossing <= 1:
            candidates.add(crossing)

    def value(lam: Fraction) -> Fraction:
        return sum(
            abs(lam * first.get(variable, 0) + (1 - lam) * second.get(variable, 0))
            for variable in variables
        )

    return min(value(candidate) for candidate in candidates)


def feasibility_from_coefficients(
    p0_coefficients: Mapping[str, int],
    p1_coefficients: Mapping[str, int],
) -> bool:
    return joint_capacity(p0_coefficients, p1_coefficients) > 1


def analyze_pole_angles(
    case: base.PlacementCase,
    state: base.SolverState,
    angle_solution: Optional[angles.AngleSolution] = None,
) -> PoleAngleAnalysis:
    if state.equations:
        raise ValueError("Pole-angle analysis requires a terminal solver state")

    solution = angle_solution or turning.complete_angle_solution(case, state)
    assignments = solution.assignment_map()
    contacts = pole_contact_points(case)

    p0 = constraint_from_contacts("P0", contacts["P0"], assignments)
    p1 = constraint_from_contacts("P1", contacts["P1"], assignments)
    capacity = joint_capacity(p0.coefficient_map(), p1.coefficient_map())
    feasible = capacity > 1

    if feasible:
        reason = None
    elif capacity == 1:
        reason = (
            "The two pole inequalities can reach their threshold only on the "
            "excluded boundary |Theta|=pi, corresponding to a U-turn/cusp or a "
            "zero/2*pi interior sector."
        )
    else:
        reason = (
            "No assignment of regular point turns can make the three interior "
            "sectors fit simultaneously at both poles."
        )

    return PoleAngleAnalysis(
        constraints=(p0, p1),
        joint_capacity_pi_units=capacity,
        feasible=feasible,
        discard_reason=reason,
    )


def enumerate_pole_analyses(
    case: base.PlacementCase,
    max_depth: Optional[int] = None,
    max_states: Optional[int] = None,
    include_discarded: bool = True,
):
    for state, derivation in base.enumerate_terminal_states(
        case,
        max_depth=max_depth,
        max_states=max_states,
    ):
        angle_solution = turning.complete_angle_solution(case, state)
        analysis = analyze_pole_angles(case, state, angle_solution=angle_solution)
        if include_discarded or analysis.feasible:
            yield state, derivation, analysis


def command_case(args: argparse.Namespace) -> int:
    case = base.find_case(args.case_id)
    emitted = 0
    for state, derivation, analysis in enumerate_pole_analyses(
        case,
        max_depth=args.max_depth,
        max_states=args.max_states,
    ):
        payload = {
            "case_id": case.case_id,
            "derivation": list(derivation),
            "environment": {
                variable: base.word_to_text(word)
                for variable, word in state.environment
            },
            "pole_angles": analysis.to_dict(),
        }
        print(json.dumps(payload, ensure_ascii=True))
        emitted += 1
        if args.max_solutions is not None and emitted >= args.max_solutions:
            break
    print(f"Emitted pole-angle analyses: {emitted}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local three-tile pole-angle feasibility filter."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    case_parser = subparsers.add_parser("case")
    case_parser.add_argument("case_id", type=int)
    case_parser.add_argument("--max-depth", type=int)
    case_parser.add_argument("--max-states", type=int)
    case_parser.add_argument("--max-solutions", type=int)
    case_parser.set_defaults(handler=command_case)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
