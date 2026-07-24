#!/usr/bin/env python3
"""
Conservative SE(2) holonomy filter for terminal contour profiles.

A free curve word X carries two intrinsic quantities:

    Kappa[X] : total signed tangent rotation along X
    D[X]     : nonzero endpoint displacement, expressed in X's start-tangent frame

A point carries its signed turn Theta.  Concatenating the contour composes these
local motions in SE(2).  The rotational component is exactly the existing total
turn equation.  The translational component is

    sum_X C_X(angles) * D[X] = 0,

where each C_X is a finite sum of unit complex phasors.  This module constructs
those phasor sums symbolically.

The complete trigonometric existential problem is deliberately not claimed to
be solved here.  The filter only rejects exact, sound obstructions:

* rotational holonomy is impossible;
* there is exactly one free curve variable and its phasor coefficient has
  one formal direction only;
* there is exactly one free curve variable and its coefficient has exactly
  two formal directions with unequal multiplicities.

In the last two cases no nonzero chord D[X] can close the contour, regardless of
how the remaining angle parameters are assigned.  All other cases are retained
as "not disproved" rather than asserted geometrically realizable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import angle_constraints as angles
import symbolic_enumerator as base
import turning_filter as turning


@dataclass(frozen=True, order=True)
class LinearAngleForm:
    """Integer affine form in Kappa[*] and Theta* variables, modulo no relations."""

    coefficients: Tuple[Tuple[str, int], ...] = ()

    @staticmethod
    def from_mapping(values: Mapping[str, int]) -> "LinearAngleForm":
        return LinearAngleForm(
            tuple(sorted((name, value) for name, value in values.items() if value != 0))
        )

    def to_mapping(self) -> Dict[str, int]:
        return dict(self.coefficients)

    def add_term(self, variable: str, coefficient: int) -> "LinearAngleForm":
        values = self.to_mapping()
        values[variable] = values.get(variable, 0) + coefficient
        return LinearAngleForm.from_mapping(values)

    def add_form(self, other: "LinearAngleForm") -> "LinearAngleForm":
        values = self.to_mapping()
        for variable, coefficient in other.coefficients:
            values[variable] = values.get(variable, 0) + coefficient
        return LinearAngleForm.from_mapping(values)

    def to_text(self) -> str:
        if not self.coefficients:
            return "0"
        terms = []
        for variable, coefficient in self.coefficients:
            if coefficient == 1:
                terms.append(variable)
            elif coefficient == -1:
                terms.append(f"-{variable}")
            else:
                terms.append(f"{coefficient}*{variable}")
        text = terms[0]
        for term in terms[1:]:
            text += " - " + term[1:] if term.startswith("-") else " + " + term
        return text


@dataclass(frozen=True)
class PhaseMultiplicity:
    phase: LinearAngleForm
    multiplicity: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "phase": self.phase.to_text(),
            "coefficients": dict(self.phase.coefficients),
            "multiplicity": self.multiplicity,
        }


@dataclass(frozen=True)
class CurveTranslationCoefficient:
    variable: str
    phases: Tuple[PhaseMultiplicity, ...]

    @property
    def occurrence_count(self) -> int:
        return sum(item.multiplicity for item in self.phases)

    def to_dict(self) -> Dict[str, object]:
        return {
            "variable": self.variable,
            "occurrence_count": self.occurrence_count,
            "formal_phasor_sum": [item.to_dict() for item in self.phases],
        }


@dataclass(frozen=True)
class TranslationClosureAnalysis:
    coefficients: Tuple[CurveTranslationCoefficient, ...]
    exact_obstruction: bool
    status: str
    discard_reason: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "endpoint_displacement_assumption": (
                "every nonempty free curve is a proper subarc of a future simple "
                "contour and therefore has nonzero endpoint displacement"
            ),
            "coefficients": [item.to_dict() for item in self.coefficients],
            "exact_obstruction": self.exact_obstruction,
            "status": self.status,
            "discard_reason": self.discard_reason,
        }


@dataclass(frozen=True)
class SE2HolonomyAnalysis:
    mirror_sign_a: int
    mirror_sign_b: int
    total_turn: turning.TotalTurnAnalysis
    translation: TranslationClosureAnalysis
    passes_filter: bool
    discard_reason: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "mirror_sign_a": self.mirror_sign_a,
            "mirror_sign_b": self.mirror_sign_b,
            "rotation_holonomy": self.total_turn.to_dict(),
            "translation_holonomy": self.translation.to_dict(),
            "passes_filter": self.passes_filter,
            "interpretation": (
                "passes_filter means no contradiction was proved by this conservative "
                "necessary-condition filter; it is not a proof of geometric realization"
            ),
            "discard_reason": self.discard_reason,
        }


def _turn_form(expression: str) -> LinearAngleForm:
    if expression == "0":
        return LinearAngleForm()
    if expression.startswith("-"):
        return LinearAngleForm.from_mapping({expression[1:]: -1})
    return LinearAngleForm.from_mapping({expression: 1})


def translation_phase_groups(
    case: base.PlacementCase,
    state: base.SolverState,
    angle_solution: angles.AngleSolution,
) -> Tuple[CurveTranslationCoefficient, ...]:
    """Construct each formal coefficient C_X as grouped unit phasor phases."""
    expanded = turning.contour_expansion(case, state)
    assignments = angle_solution.assignment_map()
    heading = LinearAngleForm()
    groups: Dict[str, Dict[LinearAngleForm, int]] = {}

    for index, literal in enumerate(expanded.segments):
        kappa = f"Kappa[{literal.variable}]"
        phase = heading
        if literal.inverse:
            # D[X^-1] in the reverse start-tangent frame is R(-Kappa[X]) D[X].
            phase = phase.add_term(kappa, -1)

        variable_groups = groups.setdefault(literal.variable, {})
        variable_groups[phase] = variable_groups.get(phase, 0) + 1

        heading = heading.add_term(kappa, -1 if literal.inverse else 1)
        next_point = expanded.points[index + 1].point
        heading = heading.add_form(_turn_form(assignments[next_point]))

    return tuple(
        CurveTranslationCoefficient(
            variable=variable,
            phases=tuple(
                PhaseMultiplicity(phase=phase, multiplicity=multiplicity)
                for phase, multiplicity in sorted(phase_map.items())
            ),
        )
        for variable, phase_map in sorted(groups.items())
    )


def analyze_translation_closure(
    case: base.PlacementCase,
    state: base.SolverState,
    angle_solution: angles.AngleSolution,
) -> TranslationClosureAnalysis:
    coefficients = translation_phase_groups(case, state, angle_solution)

    if not coefficients:
        return TranslationClosureAnalysis(
            coefficients=(),
            exact_obstruction=True,
            status="discarded",
            discard_reason="The terminal contour contains no nonempty free curve variable.",
        )

    if len(coefficients) > 1:
        return TranslationClosureAnalysis(
            coefficients=coefficients,
            exact_obstruction=False,
            status="undecided_no_exact_obstruction",
            discard_reason=None,
        )

    coefficient = coefficients[0]
    multiplicities = tuple(item.multiplicity for item in coefficient.phases)

    if len(multiplicities) == 1:
        return TranslationClosureAnalysis(
            coefficients=coefficients,
            exact_obstruction=True,
            status="discarded",
            discard_reason=(
                f"The contour has only the nonzero chord D[{coefficient.variable}], "
                "and every occurrence contributes in the same formal direction. "
                "Their positive multiples cannot sum to zero."
            ),
        )

    if len(multiplicities) == 2 and multiplicities[0] != multiplicities[1]:
        return TranslationClosureAnalysis(
            coefficients=coefficients,
            exact_obstruction=True,
            status="discarded",
            discard_reason=(
                f"The only free chord D[{coefficient.variable}] occurs in exactly "
                "two formal directions with unequal multiplicities. Two vector "
                "groups can cancel only when their magnitudes are equal and their "
                "directions are opposite."
            ),
        )

    return TranslationClosureAnalysis(
        coefficients=coefficients,
        exact_obstruction=False,
        status="undecided_no_exact_obstruction",
        discard_reason=None,
    )


def analyze_se2_holonomy(
    case: base.PlacementCase,
    state: base.SolverState,
    mirror_sign_a: Optional[int] = None,
    mirror_sign_b: Optional[int] = None,
    angle_solution: Optional[angles.AngleSolution] = None,
    total_turn_analysis: Optional[turning.TotalTurnAnalysis] = None,
) -> SE2HolonomyAnalysis:
    if state.equations:
        raise ValueError("SE(2) analysis requires a terminal solver state")

    mirror_sign_a, mirror_sign_b = angles.resolve_mirror_signs(
        case, mirror_sign_a, mirror_sign_b
    )
    solution = angle_solution or turning.complete_angle_solution(
        case,
        state,
        mirror_sign_a=mirror_sign_a,
        mirror_sign_b=mirror_sign_b,
    )
    total = total_turn_analysis or turning.analyze_total_turn(
        case,
        state,
        mirror_sign_a=mirror_sign_a,
        mirror_sign_b=mirror_sign_b,
    )
    translation = analyze_translation_closure(case, state, solution)

    if not total.feasible:
        passes = False
        reason = total.discard_reason
    elif translation.exact_obstruction:
        passes = False
        reason = translation.discard_reason
    else:
        passes = True
        reason = None

    return SE2HolonomyAnalysis(
        mirror_sign_a=mirror_sign_a,
        mirror_sign_b=mirror_sign_b,
        total_turn=total,
        translation=translation,
        passes_filter=passes,
        discard_reason=reason,
    )


def command_case(args: argparse.Namespace) -> int:
    case = base.find_case(args.case_id)
    emitted = 0
    for state, derivation in base.enumerate_terminal_states(
        case,
        max_depth=args.max_depth,
        max_states=args.max_states,
    ):
        analysis = analyze_se2_holonomy(
            case,
            state,
            mirror_sign_a=args.mirror_a,
            mirror_sign_b=args.mirror_b,
        )
        payload = {
            "case_id": case.case_id,
            "derivation": list(derivation),
            "environment": {
                variable: base.word_to_text(word)
                for variable, word in state.environment
            },
            "se2_holonomy": analysis.to_dict(),
        }
        print(json.dumps(payload, ensure_ascii=True))
        emitted += 1
        if args.max_solutions is not None and emitted >= args.max_solutions:
            break
    print(f"Emitted SE(2) analyses: {emitted}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conservative rotational/translational holonomy filter."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    case_parser = subparsers.add_parser("case")
    case_parser.add_argument("case_id", type=int)
    case_parser.add_argument(
        "--mirror-a", type=int, choices=(-1, 1),
        help="Optional diagnostic override; default uses the placement parity.",
    )
    case_parser.add_argument(
        "--mirror-b", type=int, choices=(-1, 1),
        help="Optional diagnostic override; default uses the placement parity.",
    )
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
