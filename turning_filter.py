#!/usr/bin/env python3
"""
Global turning-number filter for symbolic contour solutions.

For a positively oriented simple closed regular contour, the total signed turn
of the tangent is 2*pi.  A terminal symbolic contour contains two kinds of
turning contributions:

1. Curve-word contributions.  Every free oriented word variable X carries an
   arbitrary total turn Kappa[X].  The inverse word X^-1 contributes
   -Kappa[X].
2. Point contributions.  Every explicit contour point carries a signed corner
   turn Tau[p].  The second-pass angle solver identifies these quantities up to
   sign and can force some of them to zero.

The resulting necessary condition is a linear equation

    sum_X c_X Kappa[X] + sum_i d_i Theta[i] = 2*pi*w,

where w=1 for a counterclockwise Jordan boundary.

The curve turns Kappa[X] are currently treated as unbounded real variables.
The point parameters Theta[i] lie strictly in (-pi, pi), excluding cusps and
instantaneous U-turns.  Consequently:

- if some c_X is nonzero, this coarse filter cannot reject the profile;
- otherwise the point contribution ranges over
      (-pi * sum_i |d_i|, pi * sum_i |d_i|),
  so the profile is impossible when sum_i |d_i| <= 2*|w|.

This is a necessary-condition filter only.  Passing it does not imply planar
realizability or simplicity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import angle_constraints as angles
import symbolic_enumerator as base


@dataclass(frozen=True)
class IntegerCoefficient:
    variable: str
    coefficient: int


@dataclass(frozen=True)
class TotalTurnAnalysis:
    winding_number: int
    segment_coefficients: Tuple[IntegerCoefficient, ...]
    point_coefficients: Tuple[IntegerCoefficient, ...]
    zero_points: Tuple[str, ...]
    point_capacity_pi_units: int
    has_unbounded_segment_freedom: bool
    feasible: bool
    discard_reason: Optional[str]

    @property
    def target_pi_units(self) -> int:
        return 2 * self.winding_number

    def segment_map(self) -> Dict[str, int]:
        return {
            item.variable: item.coefficient
            for item in self.segment_coefficients
        }

    def point_map(self) -> Dict[str, int]:
        return {
            item.variable: item.coefficient
            for item in self.point_coefficients
        }

    def equation_text(self) -> str:
        terms = []
        for item in self.segment_coefficients:
            terms.append(_format_term(item.coefficient, f"Kappa[{item.variable}]"))
        for item in self.point_coefficients:
            terms.append(_format_term(item.coefficient, item.variable))
        left = _join_terms(terms)
        return f"{left} = {self.target_pi_units}*pi"

    def to_dict(self) -> Dict[str, object]:
        return {
            "convention": {
                "orientation": "counterclockwise" if self.winding_number > 0 else "clockwise",
                "total_turn": f"{self.target_pi_units}*pi",
                "point_turn_domain": "(-pi, pi)",
                "segment_total_turn_domain": "R",
            },
            "equation": self.equation_text(),
            "segment_coefficients": self.segment_map(),
            "point_coefficients": self.point_map(),
            "zero_points": list(self.zero_points),
            "point_capacity_pi_units": self.point_capacity_pi_units,
            "has_unbounded_segment_freedom": self.has_unbounded_segment_freedom,
            "feasible": self.feasible,
            "discard_reason": self.discard_reason,
        }


def _format_term(coefficient: int, variable: str) -> str:
    if coefficient == 1:
        return variable
    if coefficient == -1:
        return f"-{variable}"
    return f"{coefficient}*{variable}"


def _join_terms(terms: Sequence[str]) -> str:
    if not terms:
        return "0"
    output = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            output += " - " + term[1:]
        else:
            output += " + " + term
    return output


def _nonzero_sorted(coefficients: Mapping[str, int]) -> Tuple[IntegerCoefficient, ...]:
    return tuple(
        IntegerCoefficient(variable, coefficient)
        for variable, coefficient in sorted(coefficients.items())
        if coefficient != 0
    )


def contour_expansion(
    case: base.PlacementCase,
    state: base.SolverState,
) -> angles.ExpandedPath:
    if state.equations:
        raise ValueError("Total-turn analysis requires a terminal solver state")
    environment = state.environment_map()
    endpoints = angles.initial_segment_endpoints(case)
    expanded = angles.expand_initial_word(case.cycle_word, environment, endpoints)
    if expanded.points[0].point != expanded.points[-1].point:
        raise ValueError("The expanded contour is not closed")
    return expanded


def contour_points(
    case: base.PlacementCase,
    state: base.SolverState,
) -> Tuple[str, ...]:
    """Return each geometric contour point exactly once in traversal order."""
    expanded = contour_expansion(case, state)
    names = tuple(occurrence.point for occurrence in expanded.points[:-1])
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(
            "A geometric point appears more than once in the prototype contour: "
            + ", ".join(duplicates)
        )
    return names


def segment_turn_coefficients(
    case: base.PlacementCase,
    state: base.SolverState,
) -> Dict[str, int]:
    expanded = contour_expansion(case, state)
    coefficients: Dict[str, int] = {}
    for literal in expanded.segments:
        contribution = -1 if literal.inverse else 1
        coefficients[literal.variable] = (
            coefficients.get(literal.variable, 0) + contribution
        )
    return {
        variable: coefficient
        for variable, coefficient in coefficients.items()
        if coefficient != 0
    }


def complete_angle_solution(
    case: base.PlacementCase,
    state: base.SolverState,
    mirror_sign_a: Optional[int] = None,
    mirror_sign_b: Optional[int] = None,
) -> angles.AngleSolution:
    equations = angles.projection_angle_equations(
        case,
        state,
        mirror_sign_a=mirror_sign_a,
        mirror_sign_b=mirror_sign_b,
    )
    return angles.solve_angle_equations(
        equations,
        all_points=contour_points(case, state),
    )


def point_turn_coefficients(
    case: base.PlacementCase,
    state: base.SolverState,
    angle_solution: Optional[angles.AngleSolution] = None,
) -> Dict[str, int]:
    solution = angle_solution or complete_angle_solution(case, state)
    assignments = solution.assignment_map()
    coefficients: Dict[str, int] = {}

    for point in contour_points(case, state):
        expression = assignments[point]
        if expression == "0":
            continue
        if expression.startswith("-"):
            parameter = expression[1:]
            sign = -1
        else:
            parameter = expression
            sign = 1
        coefficients[parameter] = coefficients.get(parameter, 0) + sign

    return {
        parameter: coefficient
        for parameter, coefficient in coefficients.items()
        if coefficient != 0
    }


def analyze_total_turn(
    case: base.PlacementCase,
    state: base.SolverState,
    winding_number: int = 1,
    mirror_sign_a: Optional[int] = None,
    mirror_sign_b: Optional[int] = None,
    angle_solution: Optional[angles.AngleSolution] = None,
) -> TotalTurnAnalysis:
    if winding_number == 0:
        raise ValueError("A Jordan boundary cannot have winding number zero")

    segment_coefficients = segment_turn_coefficients(case, state)
    resolved_angle_solution = angle_solution or complete_angle_solution(
        case,
        state,
        mirror_sign_a=mirror_sign_a,
        mirror_sign_b=mirror_sign_b,
    )
    point_coefficients = point_turn_coefficients(
        case,
        state,
        angle_solution=resolved_angle_solution,
    )

    has_unbounded_segment_freedom = bool(segment_coefficients)
    point_capacity = sum(abs(value) for value in point_coefficients.values())
    target_magnitude = 2 * abs(winding_number)

    if has_unbounded_segment_freedom:
        feasible = True
        discard_reason = None
    elif point_capacity > target_magnitude:
        feasible = True
        discard_reason = None
    else:
        feasible = False
        if point_capacity == target_magnitude:
            discard_reason = (
                "The target lies only on the boundary of the attainable point-turn "
                "interval. Reaching it would require at least one excluded +/-pi "
                "U-turn/cusp."
            )
        else:
            discard_reason = (
                "All curve-word turns cancel, and the remaining point-angle classes "
                "cannot supply enough signed turn to reach the required winding."
            )

    return TotalTurnAnalysis(
        winding_number=winding_number,
        segment_coefficients=_nonzero_sorted(segment_coefficients),
        point_coefficients=_nonzero_sorted(point_coefficients),
        zero_points=resolved_angle_solution.zero_points,
        point_capacity_pi_units=point_capacity,
        has_unbounded_segment_freedom=has_unbounded_segment_freedom,
        feasible=feasible,
        discard_reason=discard_reason,
    )


def feasibility_from_coefficients(
    segment_coefficients: Mapping[str, int],
    point_coefficients: Mapping[str, int],
    winding_number: int = 1,
) -> bool:
    """Pure helper used by tests and by future external filters."""
    if any(value != 0 for value in segment_coefficients.values()):
        return True
    capacity = sum(abs(value) for value in point_coefficients.values())
    return capacity > 2 * abs(winding_number)


def enumerate_turning_analyses(
    case: base.PlacementCase,
    max_depth: Optional[int] = None,
    max_states: Optional[int] = None,
    winding_number: int = 1,
    include_discarded: bool = True,
):
    """Yield terminal states with their global-turning analysis."""
    for state, derivation in base.enumerate_terminal_states(
        case,
        max_depth=max_depth,
        max_states=max_states,
    ):
        analysis = analyze_total_turn(
            case,
            state,
            winding_number=winding_number,
        )
        if include_discarded or analysis.feasible:
            yield state, derivation, analysis


def command_case(args: argparse.Namespace) -> int:
    case = base.find_case(args.case_id)
    emitted = 0
    for state, derivation, analysis in enumerate_turning_analyses(
        case,
        max_depth=args.max_depth,
        max_states=args.max_states,
        winding_number=args.winding,
        include_discarded=args.include_discarded,
    ):
        a_text, b_text = angles.state_profile_text(case, state)
        payload = {
            "case_id": case.case_id,
            "derivation": list(derivation),
            "A": a_text,
            "B": b_text,
            "total_turn": analysis.to_dict(),
        }
        print(json.dumps(payload, ensure_ascii=True))
        emitted += 1
        if args.max_solutions is not None and emitted >= args.max_solutions:
            break
    print(f"Emitted total-turn analyses: {emitted}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Global turning-number filter for terminal contour profiles."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    case_parser = subparsers.add_parser("case")
    case_parser.add_argument("case_id", type=int)
    case_parser.add_argument("--winding", type=int, default=1)
    case_parser.add_argument("--max-depth", type=int)
    case_parser.add_argument("--max-states", type=int)
    case_parser.add_argument("--max-solutions", type=int)
    case_parser.add_argument(
        "--include-discarded",
        action="store_true",
        help="Also emit profiles rejected by the global-turning filter.",
    )
    case_parser.set_defaults(handler=command_case)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
