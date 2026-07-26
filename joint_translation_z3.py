#!/usr/bin/env python3
"""Polynomial Z3/NLSAT filter for the shared inner/outer closure system.

The backend is intentionally one-sided.  It encodes a polynomial relaxation of
all currently constructed rotation-phasor and translation-closure constraints.
Therefore:

* ``unsat`` is a sound rejection for the encoded geometric model;
* ``sat`` is only a candidate and is not a proof that a simple tile exists;
* ``unknown`` or a timeout leaves the profile undecided.

Angles are never represented with ``sin`` or ``cos``.  Each angular variable is
represented by a unit complex number ``(c, s)``.  All coefficients occurring in
the current contour systems are integral, so every phasor is built by complex
multiplication and conjugation.  The resulting formula is quantifier-free
nonlinear real arithmetic and is handled by Z3's NLSAT tactic.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import external_boundary_constraints as external
import placed_copy_geometry as placed_geometry
import settings
import symbolic_enumerator as base


@dataclass(frozen=True)
class ComplexExpr:
    real: str
    imag: str


@dataclass(frozen=True)
class Z3Problem:
    assertions_smt2: str
    script_smt2: str
    angle_symbol_map: Tuple[Tuple[str, Tuple[str, str]], ...]
    chord_symbol_map: Tuple[Tuple[str, Tuple[str, str]], ...]
    rotation_equation_count: int
    translation_equation_count: int
    contact_point_equation_count: int
    distinguished_point_inequality_count: int
    global_isometry_enforced: bool
    require_all_chords_nonzero: bool
    relaxation_notes: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "angle_symbol_map": {
                name: {"cos": symbols[0], "sin": symbols[1]}
                for name, symbols in self.angle_symbol_map
            },
            "chord_symbol_map": {
                name: {"x": symbols[0], "y": symbols[1]}
                for name, symbols in self.chord_symbol_map
            },
            "rotation_equation_count": self.rotation_equation_count,
            "translation_equation_count": self.translation_equation_count,
            "contact_point_equation_count": self.contact_point_equation_count,
            "distinguished_point_inequality_count": self.distinguished_point_inequality_count,
            "global_isometry_enforced": self.global_isometry_enforced,
            "require_all_chords_nonzero": self.require_all_chords_nonzero,
            "relaxation_notes": list(self.relaxation_notes),
        }


@dataclass(frozen=True)
class Z3Result:
    status: str
    exact_unsat: bool
    sat_candidate: bool
    elapsed_seconds: float
    reason: Optional[str]
    model_text: Optional[str]
    z3_version: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "exact_unsat": self.exact_unsat,
            "sat_candidate": self.sat_candidate,
            "elapsed_seconds": self.elapsed_seconds,
            "reason": self.reason,
            "model_text": self.model_text,
            "z3_version": self.z3_version,
        }


def _safe_symbol(prefix: str, index: int) -> str:
    return f"{prefix}_{index}"


def _fraction_smt(value: int | Fraction) -> str:
    value = Fraction(value)
    numerator = (
        str(value.numerator)
        if value.numerator >= 0
        else f"(- {-value.numerator})"
    )
    if value.denominator == 1:
        return numerator
    return f"(/ {numerator} {value.denominator})"


def _add(items: Iterable[str]) -> str:
    values = [item for item in items if item != "0"]
    if not values:
        return "0"
    if len(values) == 1:
        return values[0]
    return f"(+ {' '.join(values)})"


def _mul(items: Iterable[str]) -> str:
    values = [item for item in items if item != "1"]
    if any(item == "0" for item in values):
        return "0"
    if not values:
        return "1"
    if len(values) == 1:
        return values[0]
    return f"(* {' '.join(values)})"


def _neg(item: str) -> str:
    if item == "0":
        return "0"
    return f"(- {item})"


def _scale(value: int | Fraction, expression: str) -> str:
    value = Fraction(value)
    if value == 0:
        return "0"
    if value == 1:
        return expression
    if value == -1:
        return _neg(expression)
    return _mul((_fraction_smt(value), expression))


def _complex_add(left: ComplexExpr, right: ComplexExpr) -> ComplexExpr:
    return ComplexExpr(_add((left.real, right.real)), _add((left.imag, right.imag)))


def _complex_scale(value: int | Fraction, item: ComplexExpr) -> ComplexExpr:
    return ComplexExpr(_scale(value, item.real), _scale(value, item.imag))


def _complex_mul(left: ComplexExpr, right: ComplexExpr) -> ComplexExpr:
    return ComplexExpr(
        _add((_mul((left.real, right.real)), _neg(_mul((left.imag, right.imag))))),
        _add((_mul((left.real, right.imag)), _mul((left.imag, right.real)))),
    )


def _complex_conjugate(item: ComplexExpr) -> ComplexExpr:
    return ComplexExpr(item.real, _neg(item.imag))


def _complex_power(item: ComplexExpr, exponent: int) -> ComplexExpr:
    if exponent < 0:
        return _complex_power(_complex_conjugate(item), -exponent)
    result = ComplexExpr("1", "0")
    factor = item
    power = exponent
    while power:
        if power & 1:
            result = _complex_mul(result, factor)
        power >>= 1
        if power:
            factor = _complex_mul(factor, factor)
    return result


def _vector_angle_variables(expression: object) -> set[str]:
    return {
        variable
        for basis, _coefficient in getattr(expression, "terms")
        for variable, _value in basis.phase.coefficients
    }


def _vector_chord_variables(expression: object) -> set[str]:
    return {
        basis.curve_variable
        for basis, _coefficient in getattr(expression, "terms")
    }


def _all_angle_variables(
    system: external.JointBoundarySystem,
    placed: Optional[placed_geometry.PlacedCopyGeometryAnalysis] = None,
) -> Tuple[str, ...]:
    names = {
        variable
        for equation in system.rotation_equations
        for variable, _coefficient in equation.lhs.coefficients
    }
    names.update(
        variable
        for equation in system.translation_equations
        for coefficient in equation.coefficients
        for phase_item in coefficient.phases
        for variable, _coefficient in phase_item.phase.coefficients
    )
    if placed is not None:
        for equation in placed.contact_point_equations:
            names.update(_vector_angle_variables(equation.reference_position))
            names.update(_vector_angle_variables(equation.copy_position))
        reference_points = [
            point for point in placed.points if point.copy == placed_geometry.REFERENCE
        ]
        for point in reference_points:
            names.update(_vector_angle_variables(point.expression))
    return tuple(sorted(names))


def _all_chord_variables(
    system: external.JointBoundarySystem,
    placed: Optional[placed_geometry.PlacedCopyGeometryAnalysis] = None,
) -> Tuple[str, ...]:
    names = {
        coefficient.chord.variable
        for equation in system.translation_equations
        for coefficient in equation.coefficients
    }
    if placed is not None:
        for equation in placed.contact_point_equations:
            names.update(_vector_chord_variables(equation.reference_position))
            names.update(_vector_chord_variables(equation.copy_position))
        for point in placed.points:
            if point.copy == placed_geometry.REFERENCE:
                names.update(_vector_chord_variables(point.expression))
    return tuple(sorted(names))


def _require_integral_angle_form(form: external.AngleForm, context: str) -> None:
    if form.pi_constant.denominator != 1:
        raise NotImplementedError(
            f"{context} contains a non-integral multiple of pi: {form.to_text()}"
        )
    for variable, coefficient in form.coefficients:
        if coefficient.denominator != 1:
            raise NotImplementedError(
                f"{context} contains a non-integral coefficient for {variable}: "
                f"{form.to_text()}"
            )


def _phasor(
    form: external.AngleForm,
    symbols: Mapping[str, Tuple[str, str]],
) -> ComplexExpr:
    _require_integral_angle_form(form, "phasor")
    result = ComplexExpr("(- 1)" if form.pi_constant.numerator % 2 else "1", "0")
    for variable, coefficient in form.coefficients:
        try:
            cosine, sine = symbols[variable]
        except KeyError as exc:
            raise KeyError(f"No unit-rotation symbols declared for {variable}") from exc
        result = _complex_mul(
            result,
            _complex_power(ComplexExpr(cosine, sine), coefficient.numerator),
        )
    return result


def _rotation_residual(equation: external.RotationEquation) -> external.AngleForm:
    return external.AngleForm(
        coefficients=equation.lhs.coefficients,
        pi_constant=equation.lhs.pi_constant - equation.target_pi,
    )


def _translation_expression(
    equation: external.TranslationEquation,
    angle_symbols: Mapping[str, Tuple[str, str]],
    chord_symbols: Mapping[str, Tuple[str, str]],
) -> ComplexExpr:
    total = ComplexExpr("0", "0")
    for coefficient in equation.coefficients:
        x_symbol, y_symbol = chord_symbols[coefficient.chord.variable]
        chord = ComplexExpr(x_symbol, y_symbol)
        if coefficient.chord.conjugated:
            chord = _complex_conjugate(chord)
        for phase_item in coefficient.phases:
            rotated = _complex_mul(_phasor(phase_item.phase, angle_symbols), chord)
            total = _complex_add(
                total,
                _complex_scale(phase_item.multiplicity, rotated),
            )
    return total


def _vector_expression(
    expression: object,
    angle_symbols: Mapping[str, Tuple[str, str]],
    chord_symbols: Mapping[str, Tuple[str, str]],
) -> ComplexExpr:
    total = ComplexExpr("0", "0")
    for basis, coefficient in getattr(expression, "terms"):
        x_symbol, y_symbol = chord_symbols[basis.curve_variable]
        chord = ComplexExpr(x_symbol, y_symbol)
        if basis.conjugated:
            chord = _complex_conjugate(chord)
        phase = external.AngleForm(
            coefficients=tuple(basis.phase.coefficients),
            pi_constant=basis.phase.pi_constant,
        )
        term = _complex_mul(_phasor(phase, angle_symbols), chord)
        total = _complex_add(total, _complex_scale(coefficient, term))
    return total


def _complex_subtract(left: ComplexExpr, right: ComplexExpr) -> ComplexExpr:
    return ComplexExpr(_add((left.real, _neg(right.real))), _add((left.imag, _neg(right.imag))))


def _squared_norm(item: ComplexExpr) -> str:
    return _add((_mul((item.real, item.real)), _mul((item.imag, item.imag))))


def build_z3_problem(
    system: external.JointBoundarySystem,
    *,
    placed_geometry_analysis: Optional[placed_geometry.PlacedCopyGeometryAnalysis] = None,
    require_all_chords_nonzero: bool = settings.Z3_REQUIRE_ALL_CHORDS_NONZERO,
) -> Z3Problem:
    """Build a polynomial QF_NRA relaxation for Z3/NLSAT.

    The unit-complex representation preserves every phasor relation used by the
    translation equations.  It intentionally forgets winding-number and
    principal-angle interval information.  This makes the formula weaker than
    the full angle model, so an ``unsat`` result remains a sound discard.
    """
    angle_variables = _all_angle_variables(system, placed_geometry_analysis)
    chord_variables = _all_chord_variables(system, placed_geometry_analysis)
    angle_symbols = {
        name: (_safe_symbol("rot_c", index), _safe_symbol("rot_s", index))
        for index, name in enumerate(angle_variables)
    }
    chord_symbols = {
        name: (_safe_symbol("dx", index), _safe_symbol("dy", index))
        for index, name in enumerate(chord_variables)
    }

    lines: List[str] = [
        "; Generated by joint_translation_z3.py",
        "; QF_NRA polynomial relaxation for Z3/NLSAT.",
        "; UNSAT is a sound rejection; SAT is only a candidate.",
        "(set-logic QF_NRA)",
    ]
    for name in angle_variables:
        cosine, sine = angle_symbols[name]
        lines.append(f"; ({cosine}, {sine}) represents exp(i*{name})")
        lines.append(f"(declare-fun {cosine} () Real)")
        lines.append(f"(declare-fun {sine} () Real)")
        lines.append(
            f"(assert (= {_add((_mul((cosine, cosine)), _mul((sine, sine))))} 1))"
        )

    for name in chord_variables:
        x_symbol, y_symbol = chord_symbols[name]
        lines.append(f"; ({x_symbol}, {y_symbol}) represents D[{name}]")
        lines.append(f"(declare-fun {x_symbol} () Real)")
        lines.append(f"(declare-fun {y_symbol} () Real)")

    for equation in system.rotation_equations:
        residual = _rotation_residual(equation)
        phasor = _phasor(residual, angle_symbols)
        lines.append(f"; Rotation phasor closure: {equation.to_text()}")
        lines.append(f"(assert (= {phasor.real} 1))")
        lines.append(f"(assert (= {phasor.imag} 0))")

    for equation in system.translation_equations:
        expression = _translation_expression(
            equation,
            angle_symbols,
            chord_symbols,
        )
        lines.append(f"; Translation closure: {equation.boundary}")
        lines.append(f"(assert (= {expression.real} 0))")
        lines.append(f"(assert (= {expression.imag} 0))")

    contact_point_equation_count = 0
    distinguished_point_inequality_count = 0
    if placed_geometry_analysis is not None:
        lines.append("; One shared direct/reflected isometry per copy, enforced pointwise")
        for equation in placed_geometry_analysis.contact_point_equations:
            reference_position = _vector_expression(
                equation.reference_position, angle_symbols, chord_symbols
            )
            copy_position = _vector_expression(
                equation.copy_position, angle_symbols, chord_symbols
            )
            residual = _complex_subtract(reference_position, copy_position)
            lines.append(
                f"; Contact {equation.projection}[{equation.boundary_index}]: "
                f"{equation.reference_label} = {equation.copy_label}"
            )
            lines.append(f"(assert (= {residual.real} 0))")
            lines.append(f"(assert (= {residual.imag} 0))")
            contact_point_equation_count += 1

        reference_points = [
            point
            for point in placed_geometry_analysis.points
            if point.copy == placed_geometry.REFERENCE
        ]
        for left_index in range(len(reference_points)):
            for right_index in range(left_index + 1, len(reference_points)):
                left = _vector_expression(
                    reference_points[left_index].expression, angle_symbols, chord_symbols
                )
                right = _vector_expression(
                    reference_points[right_index].expression, angle_symbols, chord_symbols
                )
                residual = _complex_subtract(left, right)
                lines.append(
                    "; Distinct prototype cut points cannot coincide: "
                    f"{reference_points[left_index].label} != "
                    f"{reference_points[right_index].label}"
                )
                lines.append(f"(assert (> {_squared_norm(residual)} 0))")
                distinguished_point_inequality_count += 1

    norms: List[str] = []
    for name in chord_variables:
        x_symbol, y_symbol = chord_symbols[name]
        norm = _add((_mul((x_symbol, x_symbol)), _mul((y_symbol, y_symbol))))
        norms.append(norm)
        if require_all_chords_nonzero:
            lines.append(f"; D[{name}] must connect two distinct contour points")
            lines.append(f"(assert (> {norm} 0))")

    if norms:
        lines.append("; Homogeneous normalization removes the all-zero chord solution")
        lines.append(f"(assert (= {_add(norms)} 1))")
    else:
        lines.append("(assert false)")

    assertions_smt2 = "\n".join(lines) + "\n"
    script_smt2 = assertions_smt2 + "(check-sat)\n(get-model)\n"
    return Z3Problem(
        assertions_smt2=assertions_smt2,
        script_smt2=script_smt2,
        angle_symbol_map=tuple(sorted(angle_symbols.items())),
        chord_symbol_map=tuple(sorted(chord_symbols.items())),
        rotation_equation_count=2 * len(system.rotation_equations),
        translation_equation_count=2 * len(system.translation_equations),
        contact_point_equation_count=2 * contact_point_equation_count,
        distinguished_point_inequality_count=distinguished_point_inequality_count,
        global_isometry_enforced=placed_geometry_analysis is not None,
        require_all_chords_nonzero=require_all_chords_nonzero,
        relaxation_notes=(
            "Angles are represented only by unit phasors; winding numbers are forgotten.",
            "Principal point-angle bounds and pole inequalities are enforced by the exact joint linear filter, not repeated here.",
            (
                "A single global direct/reflected isometry per copy is enforced at every distinguished contact point."
                if placed_geometry_analysis is not None
                else "Global copy isometries were not supplied to this standalone problem."
            ),
            "Generic intersections between interiors of curved arcs are not encoded.",
        ),
    )


def run_z3_problem(
    problem: Z3Problem,
    *,
    timeout_ms: int = settings.Z3_DEFAULT_TIMEOUT_MS,
    include_model: bool = settings.Z3_INCLUDE_MODEL_IN_REPORT,
) -> Z3Result:
    started = time.perf_counter()
    try:
        import z3  # type: ignore
    except ImportError:
        return Z3Result(
            status="z3_not_installed",
            exact_unsat=False,
            sat_candidate=False,
            elapsed_seconds=time.perf_counter() - started,
            reason=(
                "The z3-solver Python package is not installed. Run: "
                "py -3 -m pip install -r requirements.txt"
            ),
            model_text=None,
            z3_version=None,
        )

    try:
        solver = z3.Tactic("qfnra-nlsat").solver()
        solver.set(timeout=max(1, int(timeout_ms)))
        assertions = z3.parse_smt2_string(problem.assertions_smt2)
        solver.add(assertions)
        result = solver.check()
        elapsed = time.perf_counter() - started
        version = z3.get_version_string()
        if result == z3.unsat:
            return Z3Result(
                status="unsat",
                exact_unsat=True,
                sat_candidate=False,
                elapsed_seconds=elapsed,
                reason=None,
                model_text=None,
                z3_version=version,
            )
        if result == z3.sat:
            model_text = str(solver.model()) if include_model else None
            return Z3Result(
                status="sat_candidate",
                exact_unsat=False,
                sat_candidate=True,
                elapsed_seconds=elapsed,
                reason=(
                    "The polynomial relaxation is satisfiable. This is not a proof "
                    "of a simple realizable tile."
                ),
                model_text=model_text,
                z3_version=version,
            )
        reason = solver.reason_unknown()
        status = "timeout" if "timeout" in reason.lower() else "unknown"
        return Z3Result(
            status=status,
            exact_unsat=False,
            sat_candidate=False,
            elapsed_seconds=elapsed,
            reason=reason,
            model_text=None,
            z3_version=version,
        )
    except Exception as exc:
        return Z3Result(
            status="z3_error",
            exact_unsat=False,
            sat_candidate=False,
            elapsed_seconds=time.perf_counter() - started,
            reason=f"{type(exc).__name__}: {exc}",
            model_text=None,
            z3_version=None,
        )


def find_case_and_state(
    case_id: int,
    derivation: Sequence[str] = (),
) -> Tuple[base.PlacementCase, base.SolverState]:
    case = next(
        (item for item in base.enumerate_placement_cases() if item.case_id == case_id),
        None,
    )
    if case is None:
        raise ValueError(f"Unknown placement case {case_id}")
    expected = tuple(derivation)
    candidates = list(base.enumerate_terminal_states(case))
    if expected:
        state = next(
            (state for state, actual in candidates if actual == expected),
            None,
        )
    else:
        state = candidates[0][0] if candidates else None
    if state is None:
        raise ValueError(
            f"No terminal state for case {case_id} with derivation {expected}"
        )
    return case, state


def find_voderberg_case_and_state() -> Tuple[base.PlacementCase, base.SolverState]:
    expected_loci = {
        "A_start": "P1",
        "A_end": "P0",
        "B_start": "P1",
        "B_end": "A",
    }
    case = next(
        case
        for case in base.enumerate_placement_cases()
        if (
            case.marker_locus_map() == expected_loci
            and case.a_interior_blocks == (("B_end",),)
            and case.b_interior_blocks == ()
            and case.a_direction == base.REVERSE
            and case.b_direction == base.REVERSE
        )
    )
    derivation = (
        "equal_length",
        "left_strictly_shorter",
        "involutive_palindrome",
    )
    state = next(
        state
        for state, actual in base.enumerate_terminal_states(case)
        if actual == derivation
    )
    return case, state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or solve the polynomial Z3 closure filter."
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--voderberg", action="store_true")
    selector.add_argument("--case-id", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.Z3_DEFAULT_SMT2_FILENAME),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(settings.Z3_DEFAULT_METADATA_FILENAME),
    )
    parser.add_argument("--derivation", default="")
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=settings.Z3_DEFAULT_TIMEOUT_MS,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.voderberg:
        case, state = find_voderberg_case_and_state()
    else:
        derivation = tuple(item for item in args.derivation.split(",") if item)
        case, state = find_case_and_state(args.case_id, derivation)
    system = external.build_joint_boundary_system(case, state)
    placed = placed_geometry.analyze_placed_copy_geometry(case, state, system)
    problem = build_z3_problem(system, placed_geometry_analysis=placed)
    args.output.write_text(problem.script_smt2, encoding="utf-8")
    payload: Dict[str, object] = {
        "case_id": case.case_id,
        "problem": problem.to_dict(),
        "result": None,
    }
    if args.run:
        payload["result"] = run_z3_problem(
            problem,
            timeout_ms=args.timeout_ms,
        ).to_dict()
    args.metadata.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.metadata}")
    if payload["result"] is not None:
        print(json.dumps(payload["result"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
