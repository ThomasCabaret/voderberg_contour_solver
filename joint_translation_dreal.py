#!/usr/bin/env python3
"""Standalone dReal backend for the shared inner/outer translation system.

This module is experimental and is not imported by the production pipeline.
It consumes ``JointBoundarySystem`` objects produced by
``external_boundary_constraints.py`` and builds one bounded existential formula
containing:

* the exact independent total-turn equations, eliminated by rational Gaussian
  elimination before SMT generation;
* the two complex translation-closure equations (four real equations) over the
  same chord variables;
* the same free point-angle and curve-turn parameters in every occurrence;
* one global normalization of chord lengths, so the homogeneous zero solution
  is removed;
* optionally, a strict nonzero condition for every chord.

The generated formula uses dReal's ``sin`` and ``cos`` extensions. dReal is a
Delta-complete solver: ``unsat`` is an exact impossibility certificate for the
encoded model; ``delta-sat`` is a solution of a Delta-perturbed formula and must
be numerically validated before being treated as a geometric witness.

Scope warning
-------------
This solves the *current shared closure model*. It does not yet encode every
pointwise condition saying that one single rigid isometry maps an entire contact
arc of the prototype to the corresponding physical arc. Therefore:

* ``unsat`` is a sound discard for the current model;
* ``delta-sat`` is only a candidate and is not a complete tile realization.

The file is deliberately standalone so the algebra and the backend can be
reviewed before integration.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import external_boundary_constraints as external
import settings
import symbolic_enumerator as base


PI_DEFINITION = settings.DREAL_PI_DEFINITION
DEFAULT_PRECISION = settings.DREAL_DEFAULT_PRECISION
DEFAULT_TIMEOUT_SECONDS = settings.DREAL_DEFAULT_TIMEOUT_SECONDS
DIRECT_NAMES = settings.DREAL_EXECUTABLE_CANDIDATES


@dataclass(frozen=True)
class RotationReduction:
    feasible: bool
    reason: Optional[str]
    substitutions: Tuple[Tuple[str, external.AngleForm], ...]
    free_theta: Tuple[str, ...]
    free_kappa: Tuple[str, ...]
    theta_forms: Tuple[Tuple[str, external.AngleForm], ...]
    bounded_kappa_representatives_are_sound: bool

    def substitution_map(self) -> Dict[str, external.AngleForm]:
        return dict(self.substitutions)

    def to_dict(self) -> Dict[str, object]:
        return {
            "feasible": self.feasible,
            "reason": self.reason,
            "substitutions": {
                name: form.to_text() for name, form in self.substitutions
            },
            "free_theta": list(self.free_theta),
            "free_kappa": list(self.free_kappa),
            "theta_forms": {
                name: form.to_text() for name, form in self.theta_forms
            },
            "bounded_kappa_representatives_are_sound": (
                self.bounded_kappa_representatives_are_sound
            ),
        }


@dataclass(frozen=True)
class DRealProblem:
    smt2: str
    rotation_reduction: RotationReduction
    angle_symbol_map: Tuple[Tuple[str, str], ...]
    chord_symbol_map: Tuple[Tuple[str, Tuple[str, str]], ...]
    translation_equation_count: int
    require_all_chords_nonzero: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "rotation_reduction": self.rotation_reduction.to_dict(),
            "angle_symbol_map": dict(self.angle_symbol_map),
            "chord_symbol_map": {
                name: {"x": xy[0], "y": xy[1]}
                for name, xy in self.chord_symbol_map
            },
            "translation_equation_count": self.translation_equation_count,
            "require_all_chords_nonzero": self.require_all_chords_nonzero,
        }


@dataclass(frozen=True)
class DRealResult:
    status: str
    exact_unsat: bool
    delta_sat: bool
    return_code: Optional[int]
    stdout: str
    stderr: str
    command: Tuple[str, ...]
    smt2_path: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "exact_unsat": self.exact_unsat,
            "delta_sat": self.delta_sat,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "command": list(self.command),
            "smt2_path": self.smt2_path,
        }


def _is_kappa(name: str) -> bool:
    return name.startswith("Kappa[") or name.startswith("KappaClass")


def _identity_form(name: str) -> external.AngleForm:
    return external.AngleForm.from_mapping({name: 1})


def _substitute_form(
    form: external.AngleForm,
    substitutions: Mapping[str, external.AngleForm],
) -> external.AngleForm:
    result = external.AngleForm(pi_constant=form.pi_constant)
    for variable, coefficient in form.coefficients:
        replacement = substitutions.get(variable, _identity_form(variable))
        result = result.add(replacement.scale(coefficient))
    return result


def _rref_with_column_preference(
    rows: Sequence[Sequence[Fraction]],
    rhs: Sequence[Fraction],
    variables: Sequence[str],
) -> Tuple[List[List[Fraction]], List[Fraction], List[str], List[int], Optional[str]]:
    """RREF with Kappa/unit pivots preferred.

    Columns are physically swapped. Returned ``variables`` follow the final
    column order and ``pivots`` contain the pivot-column indices in that order.
    """
    matrix = [list(row) for row in rows]
    values = list(rhs)
    names = list(variables)
    if len(matrix) != len(values):
        raise ValueError("rows/rhs size mismatch")
    if matrix and any(len(row) != len(names) for row in matrix):
        raise ValueError("row width mismatch")

    rank = 0
    pivots: List[int] = []
    while rank < len(matrix) and rank < len(names):
        candidates: List[Tuple[Tuple[int, int, Fraction, str], int, int]] = []
        for row_index in range(rank, len(matrix)):
            for column_index in range(rank, len(names)):
                value = matrix[row_index][column_index]
                if value == 0:
                    continue
                name = names[column_index]
                score = (
                    0 if _is_kappa(name) else 1,
                    0 if abs(value) == 1 else 1,
                    abs(value),
                    name,
                )
                candidates.append((score, row_index, column_index))
        if not candidates:
            break
        _score, pivot_row, pivot_column = min(candidates)

        matrix[rank], matrix[pivot_row] = matrix[pivot_row], matrix[rank]
        values[rank], values[pivot_row] = values[pivot_row], values[rank]
        if pivot_column != rank:
            for row in matrix:
                row[rank], row[pivot_column] = row[pivot_column], row[rank]
            names[rank], names[pivot_column] = names[pivot_column], names[rank]

        pivot_value = matrix[rank][rank]
        matrix[rank] = [value / pivot_value for value in matrix[rank]]
        values[rank] /= pivot_value
        for row_index in range(len(matrix)):
            if row_index == rank:
                continue
            factor = matrix[row_index][rank]
            if factor == 0:
                continue
            matrix[row_index] = [
                value - factor * pivot
                for value, pivot in zip(matrix[row_index], matrix[rank])
            ]
            values[row_index] -= factor * values[rank]

        pivots.append(rank)
        rank += 1

    for row, value in zip(matrix, values):
        if all(item == 0 for item in row) and value != 0:
            return matrix, values, names, pivots, (
                "The independent total-turn equations are inconsistent."
            )
    return matrix, values, names, pivots, None


def reduce_rotation_equations(
    equations: Sequence[external.RotationEquation],
) -> RotationReduction:
    """Eliminate exact linear total-turn equations before invoking dReal.

    Every angle is represented in radians. The right-hand sides and constants
    are stored as multiples of pi, so the affine substitutions remain exact over
    rational coefficients.
    """
    all_variables = sorted(
        {
            variable
            for equation in equations
            for variable in equation.normalized_coefficients()
        },
        key=lambda name: (0 if _is_kappa(name) else 1, name),
    )
    coefficient_maps = [equation.normalized_coefficients() for equation in equations]
    rows = [
        [mapping.get(variable, Fraction(0)) for variable in all_variables]
        for mapping in coefficient_maps
    ]
    rhs = [equation.normalized_rhs() for equation in equations]

    matrix, reduced_rhs, ordered_variables, pivots, contradiction = (
        _rref_with_column_preference(rows, rhs, all_variables)
    )
    if contradiction is not None:
        return RotationReduction(
            feasible=False,
            reason=contradiction,
            substitutions=(),
            free_theta=(),
            free_kappa=(),
            theta_forms=(),
            bounded_kappa_representatives_are_sound=False,
        )

    pivot_set = set(pivots)
    free_columns = [
        index for index in range(len(ordered_variables)) if index not in pivot_set
    ]
    substitutions: Dict[str, external.AngleForm] = {
        ordered_variables[index]: _identity_form(ordered_variables[index])
        for index in free_columns
    }

    for row_index, pivot_column in enumerate(pivots):
        values: Dict[str, Fraction] = {}
        for free_column in free_columns:
            coefficient = matrix[row_index][free_column]
            if coefficient != 0:
                values[ordered_variables[free_column]] = -coefficient
        substitutions[ordered_variables[pivot_column]] = external.AngleForm.from_mapping(
            values,
            pi_constant=reduced_rhs[row_index],
        )

    free_variables = [ordered_variables[index] for index in free_columns]
    free_theta = tuple(sorted(name for name in free_variables if not _is_kappa(name)))
    free_kappa = tuple(sorted(name for name in free_variables if _is_kappa(name)))

    original_theta = sorted(name for name in all_variables if not _is_kappa(name))
    theta_forms = tuple(
        (name, substitutions.get(name, _identity_form(name)))
        for name in original_theta
    )

    # A free Kappa can be reduced modulo 2*pi without changing any phasor.
    # Recomputing pivot Kappas preserves their phasors only when their
    # coefficients in free Kappas are integers. This is true for the current
    # model and is checked rather than assumed.
    bounded_sound = True
    for variable, form in substitutions.items():
        if not _is_kappa(variable):
            continue
        for free_variable, coefficient in form.coefficients:
            if _is_kappa(free_variable) and coefficient.denominator != 1:
                bounded_sound = False

    return RotationReduction(
        feasible=True,
        reason=None,
        substitutions=tuple(sorted(substitutions.items())),
        free_theta=free_theta,
        free_kappa=free_kappa,
        theta_forms=theta_forms,
        bounded_kappa_representatives_are_sound=bounded_sound,
    )



def _translation_phase_variables(
    equations: Sequence[external.TranslationEquation],
) -> Tuple[str, ...]:
    return tuple(sorted({
        variable
        for equation in equations
        for coefficient in equation.coefficients
        for phase_item in coefficient.phases
        for variable, _coefficient in phase_item.phase.coefficients
    }))


def _extend_reduction_with_unconstrained_phase_variables(
    reduction: RotationReduction,
    phase_variables: Sequence[str],
) -> RotationReduction:
    if not reduction.feasible:
        return reduction
    substitutions = reduction.substitution_map()
    free_theta = set(reduction.free_theta)
    free_kappa = set(reduction.free_kappa)
    theta_forms = dict(reduction.theta_forms)
    for variable in phase_variables:
        if variable not in substitutions:
            substitutions[variable] = _identity_form(variable)
            if _is_kappa(variable):
                free_kappa.add(variable)
            else:
                free_theta.add(variable)
                theta_forms[variable] = _identity_form(variable)
    return RotationReduction(
        feasible=True,
        reason=None,
        substitutions=tuple(sorted(substitutions.items())),
        free_theta=tuple(sorted(free_theta)),
        free_kappa=tuple(sorted(free_kappa)),
        theta_forms=tuple(sorted(theta_forms.items())),
        bounded_kappa_representatives_are_sound=(
            reduction.bounded_kappa_representatives_are_sound
        ),
    )

def _safe_symbol(prefix: str, index: int) -> str:
    return f"{prefix}_{index}"


def _decimal_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"(/ {value.numerator} {value.denominator})"


def _smt_add(items: Sequence[str]) -> str:
    values = [item for item in items if item != "0"]
    if not values:
        return "0"
    if len(values) == 1:
        return values[0]
    return f"(+ {' '.join(values)})"


def _smt_mul(items: Sequence[str]) -> str:
    values = [item for item in items if item != "1"]
    if not values:
        return "1"
    if any(item == "0" for item in values):
        return "0"
    if len(values) == 1:
        return values[0]
    return f"(* {' '.join(values)})"


def _smt_neg(item: str) -> str:
    if item == "0":
        return "0"
    return f"(- {item})"


def _smt_scaled(coefficient: Fraction | int, expression: str) -> str:
    coefficient = Fraction(coefficient)
    if coefficient == 0:
        return "0"
    if coefficient == 1:
        return expression
    if coefficient == -1:
        return _smt_neg(expression)
    return _smt_mul((_decimal_fraction(coefficient), expression))


def _angle_form_smt(
    form: external.AngleForm,
    angle_symbols: Mapping[str, str],
) -> str:
    terms: List[str] = []
    for variable, coefficient in form.coefficients:
        try:
            symbol = angle_symbols[variable]
        except KeyError as exc:
            raise KeyError(f"No SMT symbol declared for angle variable {variable}") from exc
        terms.append(_smt_scaled(coefficient, symbol))
    if form.pi_constant:
        terms.append(
            _smt_scaled(form.pi_constant, "PI")
        )
    return _smt_add(terms)


def _normalize_phase_for_trig(
    form: external.AngleForm,
) -> Tuple[external.AngleForm, int]:
    """Remove exact integer multiples of pi from a phasor phase.

    exp(i * (phi + k*pi)) = (-1)^k exp(i*phi).
    """
    if form.pi_constant.denominator != 1:
        return form, 1
    integer = form.pi_constant.numerator
    sign = -1 if integer % 2 else 1
    return external.AngleForm(form.coefficients, Fraction(0)), sign


def _translation_real_imag_terms(
    equation: external.TranslationEquation,
    substitutions: Mapping[str, external.AngleForm],
    angle_symbols: Mapping[str, str],
    chord_symbols: Mapping[str, Tuple[str, str]],
) -> Tuple[List[str], List[str]]:
    real_terms: List[str] = []
    imag_terms: List[str] = []
    for coefficient in equation.coefficients:
        x_symbol, y_symbol = chord_symbols[coefficient.chord.variable]
        for phase_item in coefficient.phases:
            phase = _substitute_form(phase_item.phase, substitutions)
            phase, phase_sign = _normalize_phase_for_trig(phase)
            phase_smt = _angle_form_smt(phase, angle_symbols)
            if phase_smt == "0":
                cosine = "1"
                sine = "0"
            else:
                cosine = f"(cos {phase_smt})"
                sine = f"(sin {phase_smt})"
            multiplicity = Fraction(phase_item.multiplicity * phase_sign)

            if coefficient.chord.conjugated:
                real = _smt_add((
                    _smt_mul((cosine, x_symbol)),
                    _smt_mul((sine, y_symbol)),
                ))
                imag = _smt_add((
                    _smt_mul((sine, x_symbol)),
                    _smt_neg(_smt_mul((cosine, y_symbol))),
                ))
            else:
                real = _smt_add((
                    _smt_mul((cosine, x_symbol)),
                    _smt_neg(_smt_mul((sine, y_symbol))),
                ))
                imag = _smt_add((
                    _smt_mul((sine, x_symbol)),
                    _smt_mul((cosine, y_symbol)),
                ))

            real_terms.append(_smt_scaled(multiplicity, real))
            imag_terms.append(_smt_scaled(multiplicity, imag))
    return real_terms, imag_terms


def build_dreal_problem(
    system: external.JointBoundarySystem,
    *,
    precision: str = DEFAULT_PRECISION,
    require_all_chords_nonzero: bool = True,
) -> DRealProblem:
    """Build one bounded dReal formula for the shared closure problem."""
    reduction = reduce_rotation_equations(system.rotation_equations)
    reduction = _extend_reduction_with_unconstrained_phase_variables(
        reduction,
        _translation_phase_variables(system.translation_equations),
    )
    if not reduction.feasible:
        return DRealProblem(
            smt2=(
                "; Rotation equations are already inconsistent.\n"
                "(set-logic QF_NRA)\n(assert false)\n(check-sat)\n(exit)\n"
            ),
            rotation_reduction=reduction,
            angle_symbol_map=(),
            chord_symbol_map=(),
            translation_equation_count=0,
            require_all_chords_nonzero=require_all_chords_nonzero,
        )
    if not reduction.bounded_kappa_representatives_are_sound:
        raise NotImplementedError(
            "The rotation reduction produced fractional dependence between free "
            "Kappa variables. A finite modulo-2*pi representative box is not "
            "implemented for that case."
        )

    substitutions = reduction.substitution_map()
    free_angles = tuple(sorted((*reduction.free_kappa, *reduction.free_theta)))
    angle_symbols = {
        name: _safe_symbol("angle", index)
        for index, name in enumerate(free_angles)
    }
    curve_variables = sorted(
        {
            coefficient.chord.variable
            for equation in system.translation_equations
            for coefficient in equation.coefficients
        }
    )
    chord_symbols = {
        variable: (_safe_symbol("dx", index), _safe_symbol("dy", index))
        for index, variable in enumerate(curve_variables)
    }

    lines: List[str] = [
        "; Generated by joint_translation_dreal.py",
        "; UNSAT is exact for this encoded model; delta-SAT is approximate.",
        "(set-logic QF_NRA)",
        f"(set-info :precision {precision})",
        f"(define-fun PI () Real {PI_DEFINITION})",
    ]
    for name in free_angles:
        lines.append(f"; {angle_symbols[name]} = {name}")
        lines.append(f"(declare-fun {angle_symbols[name]} () Real)")
    for variable in curve_variables:
        x_symbol, y_symbol = chord_symbols[variable]
        lines.append(f"; ({x_symbol}, {y_symbol}) = D[{variable}]")
        lines.append(f"(declare-fun {x_symbol} () Real)")
        lines.append(f"(declare-fun {y_symbol} () Real)")

    # Point-turn variables are principal turns in (-pi, pi). This includes
    # eliminated Theta variables through their exact affine expressions.
    for original_name, theta_form in reduction.theta_forms:
        expression = _angle_form_smt(theta_form, angle_symbols)
        lines.append(f"; principal-angle bound for {original_name} = {theta_form.to_text()}")
        lines.append(f"(assert (< (- PI) {expression}))")
        lines.append(f"(assert (< {expression} PI))")

    # Free Kappa parameters only matter modulo 2*pi after exact pivot
    # reconstruction. A closed representative interval is sufficient.
    for name in reduction.free_kappa:
        symbol = angle_symbols[name]
        lines.append(f"(assert (<= (- PI) {symbol}))")
        lines.append(f"(assert (<= {symbol} PI))")

    real_equations = 0
    for equation in system.translation_equations:
        real_terms, imag_terms = _translation_real_imag_terms(
            equation,
            substitutions,
            angle_symbols,
            chord_symbols,
        )
        lines.append(f"; Translation closure: {equation.boundary}, real part")
        lines.append(f"(assert (= {_smt_add(real_terms)} 0))")
        lines.append(f"; Translation closure: {equation.boundary}, imaginary part")
        lines.append(f"(assert (= {_smt_add(imag_terms)} 0))")
        real_equations += 2

    norm_terms: List[str] = []
    for variable in curve_variables:
        x_symbol, y_symbol = chord_symbols[variable]
        norm = _smt_add((
            _smt_mul((x_symbol, x_symbol)),
            _smt_mul((y_symbol, y_symbol)),
        ))
        norm_terms.append(norm)
        if require_all_chords_nonzero:
            lines.append(f"; Nonzero endpoint displacement for D[{variable}]")
            lines.append(f"(assert (> {norm} 0))")

    # The system is homogeneous in all chords. Normalization removes the zero
    # solution and makes the search box compact without losing any nonzero
    # solution (rescale every chord by the same positive factor).
    if norm_terms:
        lines.append("; Global homogeneous normalization")
        lines.append(f"(assert (= {_smt_add(norm_terms)} 1))")
    else:
        lines.append("(assert false)")

    lines.extend(("(check-sat)", "(exit)"))
    return DRealProblem(
        smt2="\n".join(lines) + "\n",
        rotation_reduction=reduction,
        angle_symbol_map=tuple(sorted(angle_symbols.items())),
        chord_symbol_map=tuple(sorted(chord_symbols.items())),
        translation_equation_count=real_equations,
        require_all_chords_nonzero=require_all_chords_nonzero,
    )


def find_dreal_executable(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    for name in DIRECT_NAMES:
        path = shutil.which(name)
        if path:
            return path
    return None


def run_dreal_problem(
    problem: DRealProblem,
    *,
    executable: Optional[str] = None,
    precision: str = DEFAULT_PRECISION,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    keep_smt2: Optional[Path] = None,
    request_model: bool = True,
) -> DRealResult:
    dreal = find_dreal_executable(executable)
    if dreal is None:
        return DRealResult(
            status="dreal_not_found",
            exact_unsat=False,
            delta_sat=False,
            return_code=None,
            stdout="",
            stderr=(
                "dReal was not found on PATH. Emit the SMT2 file and run it in "
                "a dReal installation, container, or WSL environment."
            ),
            command=(),
            smt2_path=str(keep_smt2) if keep_smt2 else None,
        )

    temporary: Optional[tempfile.NamedTemporaryFile] = None
    if keep_smt2 is not None:
        keep_smt2.write_text(problem.smt2, encoding="utf-8")
        smt_path = keep_smt2
    else:
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".smt2",
            encoding="utf-8",
            delete=False,
        )
        temporary.write(problem.smt2)
        temporary.close()
        smt_path = Path(temporary.name)

    command = [dreal, "--precision", precision]
    if request_model:
        command.append("--model")
    command.append(str(smt_path))
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        output = f"{completed.stdout}\n{completed.stderr}".lower()
        if re.search(r"(^|\s)unsat($|\s)", output):
            status = "unsat"
        elif "delta-sat" in output or "δ-sat" in output:
            status = "delta-sat"
        elif completed.returncode == 0:
            status = "unknown_output"
        else:
            status = "solver_error"
        return DRealResult(
            status=status,
            exact_unsat=status == "unsat",
            delta_sat=status == "delta-sat",
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            command=tuple(command),
            smt2_path=str(smt_path),
        )
    except FileNotFoundError as exc:
        return DRealResult(
            status="dreal_not_found",
            exact_unsat=False,
            delta_sat=False,
            return_code=None,
            stdout="",
            stderr=str(exc),
            command=tuple(command),
            smt2_path=str(smt_path),
        )
    except subprocess.TimeoutExpired as exc:
        return DRealResult(
            status="timeout",
            exact_unsat=False,
            delta_sat=False,
            return_code=None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            command=tuple(command),
            smt2_path=str(smt_path),
        )
    finally:
        if temporary is not None and keep_smt2 is None:
            try:
                Path(temporary.name).unlink(missing_ok=True)
            except OSError:
                pass


def find_case_and_state(
    case_id: int,
    derivation: Sequence[str],
) -> Tuple[base.PlacementCase, base.SolverState]:
    case = next(
        (item for item in base.enumerate_placement_cases() if item.case_id == case_id),
        None,
    )
    if case is None:
        raise ValueError(f"Unknown placement case {case_id}")
    expected = tuple(derivation)
    state = next(
        (
            state
            for state, actual_derivation in base.enumerate_terminal_states(case)
            if actual_derivation == expected
        ),
        None,
    )
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
        description="Emit or run the experimental joint translation dReal problem."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--voderberg", action="store_true")
    source.add_argument("--case-id", type=int)
    parser.add_argument(
        "--derivation",
        default="",
        help="Comma-separated terminal derivation, required with --case-id.",
    )
    parser.add_argument("--output", type=Path, default=Path(settings.DREAL_DEFAULT_SMT2_FILENAME))
    parser.add_argument("--metadata", type=Path, default=Path(settings.DREAL_DEFAULT_METADATA_FILENAME))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dreal")
    parser.add_argument("--precision", default=DEFAULT_PRECISION)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--allow-zero-chords",
        action="store_true",
        help="Only require one globally nonzero chord instead of every chord.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.voderberg:
        case, state = find_voderberg_case_and_state()
    else:
        if not args.derivation:
            raise SystemExit("--derivation is required with --case-id")
        case, state = find_case_and_state(
            args.case_id,
            tuple(part for part in args.derivation.split(",") if part),
        )

    system = external.build_joint_boundary_system(case, state)
    problem = build_dreal_problem(
        system,
        precision=args.precision,
        require_all_chords_nonzero=not args.allow_zero_chords,
    )
    args.output.write_text(problem.smt2, encoding="utf-8")
    metadata = {
        "case_id": case.case_id,
        "problem": problem.to_dict(),
    }
    if args.run:
        result = run_dreal_problem(
            problem,
            executable=args.dreal,
            precision=args.precision,
            timeout_seconds=args.timeout,
            keep_smt2=args.output,
        )
        metadata["result"] = result.to_dict()
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Wrote {args.output}")

    if args.metadata:
        args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
