#!/usr/bin/env python3
"""Prototype construction of the external boundary of the three-tile union.

This module is deliberately standalone: it is not imported by the current
analysis pipeline.  It explores the next model layer discussed in the project:

* construct the positively oriented boundary of the union of the reference tile
  and the two copies covering A and B;
* express the reference-tile and union-boundary turning equations with the same
  Kappa/Theta variables;
* express both translation-closure equations with the same chord variables;
* solve the two turning equations jointly;
* retain translation closure as one shared symbolic system, with only small,
  sound obstruction checks for now.

The prototype contour is counterclockwise.  A copy with isometry parity m has
its positively oriented physical boundary represented in prototype coordinates
with direction m.  Since placement construction already enforces m = -d for a
contact read in direction d, the free complement of each contact is obtained by
reading between the same endpoints in direction m.

For the union boundary this gives:

    outer_A : A_start -> A_end in direction a_mirror_sign, physical P0 -> P1
    outer_B : B_start -> B_end in direction b_mirror_sign, physical P1 -> P0

The two outer-pole turns are derived from the three physical tile sectors:

    tau_outer(P) = tau_ref + tau_copy_A + tau_copy_B - 2*pi.

All angle forms below are stored in radians symbolically, with ``pi_constant``
recording an integer/rational multiple of pi.  For joint feasibility we divide
all angle variables by pi, so free point-turn parameters lie in (-1, 1), while
Kappa variables remain unbounded.

Important limitation
--------------------
The joint rotation solver is exact for the two linear turning equations.  The
translation equations contain phasors whose phases depend on the same angle
variables and, for reflected copies, both D[X] and conjugate(D[X]).  This file
constructs that shared system exactly, but only applies conservative elementary
obstruction checks.  It does not claim to solve the full trigonometric
existential problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import angle_constraints as angles
import symbolic_enumerator as base
import turning_filter as turning


REFERENCE_COPY = "reference"
A_COPY = "copy_A"
B_COPY = "copy_B"
OUTER_P0 = "outer:P0"
OUTER_P1 = "outer:P1"


@dataclass(frozen=True, order=True)
class AngleForm:
    """Affine integer/rational form in Kappa/Theta variables plus pi constant."""

    coefficients: Tuple[Tuple[str, Fraction], ...] = ()
    pi_constant: Fraction = Fraction(0)

    @staticmethod
    def zero() -> "AngleForm":
        return AngleForm()

    @staticmethod
    def from_mapping(
        values: Mapping[str, int | Fraction],
        pi_constant: int | Fraction = 0,
    ) -> "AngleForm":
        return AngleForm(
            coefficients=tuple(
                sorted(
                    (name, Fraction(value))
                    for name, value in values.items()
                    if value != 0
                )
            ),
            pi_constant=Fraction(pi_constant),
        )

    def to_mapping(self) -> Dict[str, Fraction]:
        return dict(self.coefficients)

    def add(self, other: "AngleForm") -> "AngleForm":
        values = self.to_mapping()
        for variable, coefficient in other.coefficients:
            values[variable] = values.get(variable, Fraction(0)) + coefficient
        return AngleForm.from_mapping(
            values,
            pi_constant=self.pi_constant + other.pi_constant,
        )

    def scale(self, factor: int | Fraction) -> "AngleForm":
        factor = Fraction(factor)
        return AngleForm.from_mapping(
            {
                variable: coefficient * factor
                for variable, coefficient in self.coefficients
            },
            pi_constant=self.pi_constant * factor,
        )

    def add_term(self, variable: str, coefficient: int | Fraction) -> "AngleForm":
        return self.add(AngleForm.from_mapping({variable: coefficient}))

    def without_pi_constant(self) -> "AngleForm":
        return AngleForm(self.coefficients, Fraction(0))

    def to_text(self) -> str:
        terms: List[str] = []
        for variable, coefficient in self.coefficients:
            if coefficient == 1:
                terms.append(variable)
            elif coefficient == -1:
                terms.append(f"-{variable}")
            else:
                terms.append(f"{_fraction_text(coefficient)}*{variable}")
        if self.pi_constant:
            if self.pi_constant == 1:
                terms.append("pi")
            elif self.pi_constant == -1:
                terms.append("-pi")
            else:
                terms.append(f"{_fraction_text(self.pi_constant)}*pi")
        return _join_signed_terms(terms) if terms else "0"


@dataclass(frozen=True)
class BoundarySegment:
    copy: str
    mirror_sign: int
    literal: base.Literal
    occurrence_index: int

    @property
    def traversal_sign(self) -> int:
        return -1 if self.literal.inverse else 1

    @property
    def physical_turn_sign(self) -> int:
        """Coefficient multiplying Kappa[literal.variable]."""
        return self.mirror_sign * self.traversal_sign

    @property
    def conjugated_chord(self) -> bool:
        return self.mirror_sign == base.REFLECTED

    def to_dict(self) -> Dict[str, object]:
        return {
            "copy": self.copy,
            "mirror_sign": self.mirror_sign,
            "literal": self.literal.to_text(),
            "occurrence_index": self.occurrence_index,
            "physical_turn_sign": self.physical_turn_sign,
            "conjugated_chord": self.conjugated_chord,
        }


@dataclass(frozen=True)
class BoundaryPoint:
    physical_point: str
    source_points: Tuple[str, ...]
    turn: AngleForm
    kind: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "physical_point": self.physical_point,
            "source_points": list(self.source_points),
            "turn": self.turn.to_text(),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class BoundaryPath:
    name: str
    segments: Tuple[BoundarySegment, ...]
    points: Tuple[BoundaryPoint, ...]
    initial_words: Tuple[Tuple[str, base.Word], ...]

    def __post_init__(self) -> None:
        if len(self.points) != len(self.segments) + 1:
            raise ValueError("A boundary path with n segments requires n+1 points")
        if self.points[0].physical_point != self.points[-1].physical_point:
            raise ValueError("A closed boundary path must repeat its start point at the end")

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "segments": [segment.to_dict() for segment in self.segments],
            "points": [point.to_dict() for point in self.points],
            "initial_words": {
                name: base.word_to_text(word)
                for name, word in self.initial_words
            },
        }


@dataclass(frozen=True)
class RotationEquation:
    boundary: str
    lhs: AngleForm
    target_pi: Fraction = Fraction(2)

    def normalized_coefficients(self) -> Dict[str, Fraction]:
        return self.lhs.to_mapping()

    def normalized_rhs(self) -> Fraction:
        """RHS after dividing every angle variable by pi."""
        return self.target_pi - self.lhs.pi_constant

    def to_text(self) -> str:
        return f"{self.lhs.to_text()} = {_pi_text(self.target_pi)}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "boundary": self.boundary,
            "equation": self.to_text(),
            "coefficients": {
                name: _fraction_json(value)
                for name, value in self.lhs.coefficients
            },
            "pi_constant": _fraction_json(self.lhs.pi_constant),
            "target_pi": _fraction_json(self.target_pi),
        }


@dataclass(frozen=True, order=True)
class ChordKey:
    variable: str
    conjugated: bool

    def to_text(self) -> str:
        return f"conj(D[{self.variable}])" if self.conjugated else f"D[{self.variable}]"


@dataclass(frozen=True)
class PhaseMultiplicity:
    phase: AngleForm
    multiplicity: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "phase": self.phase.to_text(),
            "multiplicity": self.multiplicity,
        }


@dataclass(frozen=True)
class TranslationCoefficient:
    chord: ChordKey
    phases: Tuple[PhaseMultiplicity, ...]

    @property
    def occurrence_count(self) -> int:
        return sum(item.multiplicity for item in self.phases)

    def to_dict(self) -> Dict[str, object]:
        return {
            "chord": self.chord.to_text(),
            "variable": self.chord.variable,
            "conjugated": self.chord.conjugated,
            "occurrence_count": self.occurrence_count,
            "formal_phasor_sum": [item.to_dict() for item in self.phases],
        }


@dataclass(frozen=True)
class TranslationEquation:
    boundary: str
    coefficients: Tuple[TranslationCoefficient, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "boundary": self.boundary,
            "equation": " + ".join(
                f"C[{item.chord.to_text()}]*{item.chord.to_text()}"
                for item in self.coefficients
            )
            + " = 0",
            "coefficients": [item.to_dict() for item in self.coefficients],
        }


@dataclass(frozen=True)
class CurveTurnEquation:
    left_variable: str
    right_variable: str
    sign: int
    projection: str
    segment_index: int

    def to_text(self) -> str:
        operator = "=" if self.sign == 1 else "=-"
        return (
            f"Kappa[{self.left_variable}] {operator} "
            f"Kappa[{self.right_variable}]"
        )


@dataclass(frozen=True)
class CurveTurnSolution:
    equations: Tuple[CurveTurnEquation, ...]
    assignments: Tuple[Tuple[str, str], ...]
    zero_variables: Tuple[str, ...]

    def assignment_map(self) -> Dict[str, str]:
        return dict(self.assignments)

    def to_dict(self) -> Dict[str, object]:
        return {
            "equations": [equation.to_text() for equation in self.equations],
            "assignments": dict(self.assignments),
            "zero_variables": list(self.zero_variables),
        }


@dataclass(frozen=True)
class JointRotationAnalysis:
    feasible: bool
    reason: Optional[str]
    kappa_rank: int
    residual_theta_rank: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "feasible": self.feasible,
            "reason": self.reason,
            "kappa_rank": self.kappa_rank,
            "residual_theta_rank": self.residual_theta_rank,
        }


@dataclass(frozen=True)
class JointTranslationAnalysis:
    exact_obstruction: bool
    status: str
    reason: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "exact_obstruction": self.exact_obstruction,
            "status": self.status,
            "reason": self.reason,
            "scope": (
                "Only elementary sound obstructions are checked. The shared "
                "two-contour phasor system is constructed but not solved completely."
            ),
        }


@dataclass(frozen=True)
class JointBoundarySystem:
    inner_boundary: BoundaryPath
    outer_boundary: BoundaryPath
    curve_turn_solution: CurveTurnSolution
    rotation_equations: Tuple[RotationEquation, RotationEquation]
    translation_equations: Tuple[TranslationEquation, TranslationEquation]
    rotation_analysis: JointRotationAnalysis
    translation_analysis: JointTranslationAnalysis

    def to_dict(self) -> Dict[str, object]:
        return {
            "inner_boundary": self.inner_boundary.to_dict(),
            "outer_boundary": self.outer_boundary.to_dict(),
            "projection_curve_turn_constraints": self.curve_turn_solution.to_dict(),
            "rotation_equations": [item.to_dict() for item in self.rotation_equations],
            "translation_equations": [
                item.to_dict() for item in self.translation_equations
            ],
            "joint_rotation_analysis": self.rotation_analysis.to_dict(),
            "joint_translation_analysis": self.translation_analysis.to_dict(),
        }


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _fraction_json(value: Fraction) -> Dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _pi_text(value: Fraction) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "pi"
    if value == -1:
        return "-pi"
    return f"{_fraction_text(value)}*pi"


def _join_signed_terms(terms: Sequence[str]) -> str:
    if not terms:
        return "0"
    text = terms[0]
    for term in terms[1:]:
        text += f" - {term[1:]}" if term.startswith("-") else f" + {term}"
    return text


def angle_expression_form(expression: str) -> AngleForm:
    if expression == "0":
        return AngleForm.zero()
    if expression.startswith("-"):
        return AngleForm.from_mapping({expression[1:]: -1})
    return AngleForm.from_mapping({expression: 1})


def pole_outer_turn(
    contact_points: Sequence[str],
    assignments: Mapping[str, str],
) -> AngleForm:
    """Return sum(tile turns) - 2*pi for one union-boundary pole."""
    result = AngleForm(pi_constant=Fraction(-2))
    for point in contact_points:
        result = result.add(angle_expression_form(assignments[point]))
    return result


def _marker_prototype_point(case: base.PlacementCase, marker: str) -> str:
    boundaries = angles.contour_boundary_points(case)
    boundary = case.marker_boundary_map()[marker]
    return boundaries[boundary % len(boundaries)]


def pole_contact_points(case: base.PlacementCase) -> Dict[str, Tuple[str, str, str]]:
    return {
        "P0": (
            "P0",
            _marker_prototype_point(case, "A_start"),
            _marker_prototype_point(case, "B_end"),
        ),
        "P1": (
            "P1",
            _marker_prototype_point(case, "A_end"),
            _marker_prototype_point(case, "B_start"),
        ),
    }


def free_copy_initial_words(case: base.PlacementCase) -> Tuple[base.Word, base.Word]:
    """Return the two free complement factors in positive physical orientation."""
    boundaries = case.marker_boundary_map()
    free_a = base.cyclic_factor(
        case.cycle_word,
        boundaries["A_start"],
        boundaries["A_end"],
        case.a_mirror_sign,
    )
    free_b = base.cyclic_factor(
        case.cycle_word,
        boundaries["B_start"],
        boundaries["B_end"],
        case.b_mirror_sign,
    )
    if not free_a or not free_b:
        raise ValueError("Each copy must retain a nonempty free boundary complement")
    return free_a, free_b


def _scaled_assignment_turn(
    prototype_point: str,
    orientation: int,
    mirror_sign: int,
    assignments: Mapping[str, str],
) -> AngleForm:
    return angle_expression_form(assignments[prototype_point]).scale(
        orientation * mirror_sign
    )


def _expanded_copy_arc(
    *,
    copy_name: str,
    mirror_sign: int,
    initial_word: base.Word,
    environment: Mapping[str, base.Word],
    endpoints: Mapping[str, Tuple[str, str]],
    assignments: Mapping[str, str],
    physical_start: str,
    physical_end: str,
    occurrence_offset: int,
) -> Tuple[Tuple[BoundarySegment, ...], Tuple[BoundaryPoint, ...]]:
    expanded = angles.expand_initial_word(initial_word, environment, endpoints)
    segments = tuple(
        BoundarySegment(
            copy=copy_name,
            mirror_sign=mirror_sign,
            literal=literal,
            occurrence_index=occurrence_offset + index,
        )
        for index, literal in enumerate(expanded.segments)
    )

    points: List[BoundaryPoint] = []
    for index, occurrence in enumerate(expanded.points):
        if index == 0:
            physical_point = physical_start
            kind = "pole_endpoint"
        elif index == len(expanded.points) - 1:
            physical_point = physical_end
            kind = "pole_endpoint"
        else:
            physical_point = f"{copy_name}:{occurrence.point}"
            kind = "copy_internal"
        turn = _scaled_assignment_turn(
            occurrence.point,
            occurrence.orientation,
            mirror_sign,
            assignments,
        )
        points.append(
            BoundaryPoint(
                physical_point=physical_point,
                source_points=(occurrence.point,),
                turn=turn,
                kind=kind,
            )
        )
    return segments, tuple(points)


def build_inner_boundary(
    case: base.PlacementCase,
    state: base.SolverState,
    angle_solution: angles.AngleSolution,
) -> BoundaryPath:
    environment = state.environment_map()
    endpoints = angles.initial_segment_endpoints(case)
    expanded = angles.expand_initial_word(case.cycle_word, environment, endpoints)
    assignments = angle_solution.assignment_map()

    segments = tuple(
        BoundarySegment(
            copy=REFERENCE_COPY,
            mirror_sign=base.DIRECT,
            literal=literal,
            occurrence_index=index,
        )
        for index, literal in enumerate(expanded.segments)
    )
    points = tuple(
        BoundaryPoint(
            physical_point=occurrence.point,
            source_points=(occurrence.point,),
            turn=angle_expression_form(assignments[occurrence.point]),
            kind="reference",
        )
        for occurrence in expanded.points
    )
    return BoundaryPath(
        name="reference_tile_boundary",
        segments=segments,
        points=points,
        initial_words=(("reference", case.cycle_word),),
    )


def build_outer_boundary(
    case: base.PlacementCase,
    state: base.SolverState,
    angle_solution: angles.AngleSolution,
) -> BoundaryPath:
    """Build the positive external contour of the union of the three copies."""
    if state.equations:
        raise ValueError("External-boundary construction requires a terminal state")

    environment = state.environment_map()
    endpoints = angles.initial_segment_endpoints(case)
    assignments = angle_solution.assignment_map()
    contacts = pole_contact_points(case)
    free_a, free_b = free_copy_initial_words(case)

    a_segments, a_points = _expanded_copy_arc(
        copy_name=A_COPY,
        mirror_sign=case.a_mirror_sign,
        initial_word=free_a,
        environment=environment,
        endpoints=endpoints,
        assignments=assignments,
        physical_start=OUTER_P0,
        physical_end=OUTER_P1,
        occurrence_offset=0,
    )
    b_segments, b_points = _expanded_copy_arc(
        copy_name=B_COPY,
        mirror_sign=case.b_mirror_sign,
        initial_word=free_b,
        environment=environment,
        endpoints=endpoints,
        assignments=assignments,
        physical_start=OUTER_P1,
        physical_end=OUTER_P0,
        occurrence_offset=len(a_segments),
    )

    outer_p0 = BoundaryPoint(
        physical_point=OUTER_P0,
        source_points=contacts["P0"],
        turn=pole_outer_turn(contacts["P0"], assignments),
        kind="union_pole",
    )
    outer_p1 = BoundaryPoint(
        physical_point=OUTER_P1,
        source_points=contacts["P1"],
        turn=pole_outer_turn(contacts["P1"], assignments),
        kind="union_pole",
    )

    # Replace copy endpoint turns with the union-boundary pole turns.  Internal
    # points of each free arc retain their physical copy-boundary turn.
    points = (
        outer_p0,
        *a_points[1:-1],
        outer_p1,
        *b_points[1:-1],
        outer_p0,
    )
    return BoundaryPath(
        name="three_tile_union_boundary",
        segments=a_segments + b_segments,
        points=tuple(points),
        initial_words=((A_COPY, free_a), (B_COPY, free_b)),
    )


class _SignedScalarDSU:
    """Relations value(x) = sign * value(y), with contradictions forcing zero."""

    def __init__(self, variables: Iterable[str]) -> None:
        names = sorted(set(variables))
        self.index = {name: index for index, name in enumerate(names)}
        self.names = names
        self.parent = list(range(len(names)))
        self.parity = [1] * len(names)
        self.rank = [0] * len(names)
        self.zero_root = [False] * len(names)

    def find(self, item: int) -> Tuple[int, int]:
        if self.parent[item] != item:
            root, sign = self.find(self.parent[item])
            self.parity[item] *= sign
            self.parent[item] = root
        return self.parent[item], self.parity[item]

    def union(self, left: str, right: str, sign: int) -> None:
        li = self.index[left]
        ri = self.index[right]
        lr, ls = self.find(li)
        rr, rs = self.find(ri)
        if lr == rr:
            if ls != sign * rs:
                self.zero_root[lr] = True
            return
        relation = ls * sign * rs
        if self.rank[lr] < self.rank[rr]:
            self.parent[lr] = rr
            self.parity[lr] = relation
            self.zero_root[rr] = self.zero_root[rr] or self.zero_root[lr]
        else:
            self.parent[rr] = lr
            self.parity[rr] = relation
            self.zero_root[lr] = self.zero_root[lr] or self.zero_root[rr]
            if self.rank[lr] == self.rank[rr]:
                self.rank[lr] += 1

    def solve(self) -> Tuple[Tuple[Tuple[str, str], ...], Tuple[str, ...]]:
        components: Dict[int, List[Tuple[str, int]]] = {}
        for name, index in self.index.items():
            root, sign = self.find(index)
            components.setdefault(root, []).append((name, sign))
        assignments: List[Tuple[str, str]] = []
        zero: List[str] = []
        parameter_index = 0
        for root in sorted(components, key=lambda r: min(name for name, _ in components[r])):
            members = sorted(components[root])
            if self.zero_root[self.find(root)[0]]:
                for name, _sign in members:
                    assignments.append((name, "0"))
                    zero.append(name)
                continue
            parameter = f"KappaClass{parameter_index}"
            parameter_index += 1
            first_sign = members[0][1]
            for name, sign in members:
                normalized = sign * first_sign
                assignments.append((name, parameter if normalized == 1 else f"-{parameter}"))
        return tuple(sorted(assignments)), tuple(sorted(zero))


def projection_curve_turn_constraints(
    case: base.PlacementCase,
    state: base.SolverState,
) -> CurveTurnSolution:
    """Derive Kappa relations induced by direct/reflected contact isometries."""
    environment = state.environment_map()
    endpoints = angles.initial_segment_endpoints(case)
    all_variables = {
        literal.variable
        for literal in angles.expand_initial_word(
            case.cycle_word, environment, endpoints
        ).segments
    }
    equations: List[CurveTurnEquation] = []
    data = (
        ("A", case.a_word, case.a_target, case.a_mirror_sign),
        ("B", case.b_word, case.b_target, case.b_mirror_sign),
    )
    for projection, left_word, right_word, mirror_sign in data:
        left = angles.expand_initial_word(left_word, environment, endpoints)
        right = angles.expand_initial_word(right_word, environment, endpoints)
        if left.segments != right.segments:
            raise ValueError(f"Terminal state does not solve projection {projection}")
        for index, (left_literal, right_literal) in enumerate(
            zip(left.segments, right.segments)
        ):
            left_sign = -1 if left_literal.inverse else 1
            right_sign = -1 if right_literal.inverse else 1
            relation = mirror_sign * left_sign * right_sign
            equations.append(
                CurveTurnEquation(
                    left_variable=left_literal.variable,
                    right_variable=right_literal.variable,
                    sign=relation,
                    projection=projection,
                    segment_index=index,
                )
            )

    dsu = _SignedScalarDSU(all_variables)
    for equation in equations:
        dsu.union(
            equation.left_variable,
            equation.right_variable,
            equation.sign,
        )
    assignments, zero = dsu.solve()
    return CurveTurnSolution(
        equations=tuple(equations),
        assignments=assignments,
        zero_variables=zero,
    )


def apply_curve_turn_solution(
    form: AngleForm,
    solution: CurveTurnSolution,
) -> AngleForm:
    assignments = solution.assignment_map()
    result = AngleForm(pi_constant=form.pi_constant)
    for variable, coefficient in form.coefficients:
        if not variable.startswith("Kappa["):
            result = result.add_term(variable, coefficient)
            continue
        curve = variable[len("Kappa["):-1]
        expression = assignments.get(curve, variable)
        if expression == "0":
            continue
        if expression.startswith("-"):
            result = result.add_term(expression[1:], -coefficient)
        else:
            result = result.add_term(expression, coefficient)
    return result


def boundary_rotation_equation(boundary: BoundaryPath) -> RotationEquation:
    total = AngleForm.zero()
    for index, segment in enumerate(boundary.segments):
        total = total.add_term(
            f"Kappa[{segment.literal.variable}]",
            segment.physical_turn_sign,
        )
        total = total.add(boundary.points[index + 1].turn)
    return RotationEquation(boundary=boundary.name, lhs=total)


def boundary_translation_equation(boundary: BoundaryPath) -> TranslationEquation:
    """Construct sum C_X(angles) D_X + Cbar_X(angles) conj(D_X) = 0."""
    heading = AngleForm.zero()
    grouped: Dict[ChordKey, Dict[AngleForm, int]] = {}

    for index, segment in enumerate(boundary.segments):
        kappa = f"Kappa[{segment.literal.variable}]"
        phase = heading
        if segment.literal.inverse:
            # Reverse traversal in the start-tangent frame. Reflection conjugates
            # R(-Kappa)D into R(+Kappa)conj(D).
            phase = phase.add_term(kappa, -segment.mirror_sign)

        key = ChordKey(
            variable=segment.literal.variable,
            conjugated=segment.conjugated_chord,
        )
        phase_map = grouped.setdefault(key, {})
        phase_map[phase] = phase_map.get(phase, 0) + 1

        heading = heading.add_term(kappa, segment.physical_turn_sign)
        heading = heading.add(boundary.points[index + 1].turn)

    coefficients = tuple(
        TranslationCoefficient(
            chord=key,
            phases=tuple(
                PhaseMultiplicity(phase=phase, multiplicity=multiplicity)
                for phase, multiplicity in sorted(phase_map.items())
            ),
        )
        for key, phase_map in sorted(grouped.items())
    )
    return TranslationEquation(boundary=boundary.name, coefficients=coefficients)


def apply_curve_turns_to_translation(
    equation: TranslationEquation,
    solution: CurveTurnSolution,
) -> TranslationEquation:
    grouped: Dict[ChordKey, Dict[AngleForm, int]] = {}
    for coefficient in equation.coefficients:
        phase_map = grouped.setdefault(coefficient.chord, {})
        for item in coefficient.phases:
            phase = apply_curve_turn_solution(item.phase, solution)
            phase_map[phase] = phase_map.get(phase, 0) + item.multiplicity
    return TranslationEquation(
        boundary=equation.boundary,
        coefficients=tuple(
            TranslationCoefficient(
                chord=key,
                phases=tuple(
                    PhaseMultiplicity(phase=phase, multiplicity=multiplicity)
                    for phase, multiplicity in sorted(phase_map.items())
                ),
            )
            for key, phase_map in sorted(grouped.items())
        ),
    )


def _matrix_rank(rows: Sequence[Sequence[Fraction]]) -> int:
    matrix = [list(row) for row in rows if any(value != 0 for value in row)]
    if not matrix:
        return 0
    width = len(matrix[0])
    rank = 0
    column = 0
    while rank < len(matrix) and column < width:
        pivot = next(
            (row for row in range(rank, len(matrix)) if matrix[row][column] != 0),
            None,
        )
        if pivot is None:
            column += 1
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][column]
        matrix[rank] = [value / pivot_value for value in matrix[rank]]
        for row in range(len(matrix)):
            if row == rank or matrix[row][column] == 0:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                value - factor * pivot_item
                for value, pivot_item in zip(matrix[row], matrix[rank])
            ]
        rank += 1
        column += 1
    return rank


def _bounded_theta_feasible(
    rows: Sequence[Sequence[Fraction]],
    rhs: Sequence[Fraction],
) -> Tuple[bool, int]:
    """Exact feasibility of up to two equations with x_i in the open cube (-1,1)."""
    if len(rows) != len(rhs):
        raise ValueError("rows/rhs size mismatch")
    if not rows:
        return True, 0

    rank = _matrix_rank(rows)
    if rank == 0:
        return all(value == 0 for value in rhs), 0

    if len(rows) == 1 or rank == 1:
        if len(rows) == 1:
            coefficients = list(rows[0])
            target = rhs[0]
        else:
            row0, row1 = rows
            if any(value != 0 for value in row0):
                pivot = next(index for index, value in enumerate(row0) if value != 0)
                ratio = row1[pivot] / row0[pivot]
                if any(b != ratio * a for a, b in zip(row0, row1)):
                    raise RuntimeError("rank-one reduction received non-proportional rows")
                if rhs[1] != ratio * rhs[0]:
                    return False, 1
                coefficients = list(row0)
                target = rhs[0]
            else:
                coefficients = list(row1)
                target = rhs[1]
        capacity = sum(abs(value) for value in coefficients)
        if capacity == 0:
            return target == 0, 1
        return abs(target) < capacity, 1

    # Rank two: the image of the open cube is the strict interior of a 2D
    # zonotope. Its facets have normals perpendicular to the nonzero columns.
    row0, row1 = rows
    columns = [
        (row0[index], row1[index])
        for index in range(len(row0))
        if row0[index] != 0 or row1[index] != 0
    ]
    target = (rhs[0], rhs[1])
    for vx, vy in columns:
        nx, ny = -vy, vx
        support = sum(abs(nx * wx + ny * wy) for wx, wy in columns)
        projected = abs(nx * target[0] + ny * target[1])
        if projected >= support:
            return False, 2
    return True, 2


def analyze_joint_rotations(
    equations: Sequence[RotationEquation],
) -> JointRotationAnalysis:
    if len(equations) != 2:
        raise ValueError("This prototype expects exactly inner and outer equations")

    all_variables = sorted(
        {
            variable
            for equation in equations
            for variable in equation.normalized_coefficients()
        }
    )
    def is_kappa_variable(name: str) -> bool:
        return name.startswith("Kappa[") or name.startswith("KappaClass")

    kappa_variables = [name for name in all_variables if is_kappa_variable(name)]
    theta_variables = [name for name in all_variables if not is_kappa_variable(name)]

    coefficient_maps = [equation.normalized_coefficients() for equation in equations]
    kappa_rows = [
        [mapping.get(variable, Fraction(0)) for variable in kappa_variables]
        for mapping in coefficient_maps
    ]
    theta_rows = [
        [mapping.get(variable, Fraction(0)) for variable in theta_variables]
        for mapping in coefficient_maps
    ]
    rhs = [equation.normalized_rhs() for equation in equations]
    kappa_rank = _matrix_rank(kappa_rows)

    if kappa_rank == 2:
        return JointRotationAnalysis(
            feasible=True,
            reason=None,
            kappa_rank=2,
            residual_theta_rank=0,
        )

    if kappa_rank == 1:
        row0, row1 = kappa_rows
        if any(value != 0 for value in row0):
            pivot = next(index for index, value in enumerate(row0) if value != 0)
            ratio = row1[pivot] / row0[pivot]
            residual_row = [
                b - ratio * a for a, b in zip(theta_rows[0], theta_rows[1])
            ]
            residual_rhs = rhs[1] - ratio * rhs[0]
        else:
            residual_row = theta_rows[0]
            residual_rhs = rhs[0]
        feasible, residual_rank = _bounded_theta_feasible(
            [residual_row], [residual_rhs]
        )
        return JointRotationAnalysis(
            feasible=feasible,
            reason=None if feasible else (
                "The inner and outer turning equations require incompatible values "
                "from the shared bounded point-angle classes."
            ),
            kappa_rank=1,
            residual_theta_rank=residual_rank,
        )

    feasible, residual_rank = _bounded_theta_feasible(theta_rows, rhs)
    return JointRotationAnalysis(
        feasible=feasible,
        reason=None if feasible else (
            "With every curve-turn contribution cancelled, the two shared point-angle "
            "equations cannot be satisfied simultaneously inside (-pi, pi)."
        ),
        kappa_rank=0,
        residual_theta_rank=residual_rank,
    )


def _single_equation_translation_obstruction(
    equation: TranslationEquation,
) -> Optional[str]:
    """Small sound subset of the existing one-contour obstruction logic."""
    if not equation.coefficients:
        return "The boundary contains no nonempty free curve chord."

    variables = {item.chord.variable for item in equation.coefficients}
    if len(variables) != 1:
        return None

    conjugation_modes = {item.chord.conjugated for item in equation.coefficients}
    if len(conjugation_modes) != 1:
        return None

    phases = [
        phase
        for coefficient in equation.coefficients
        for phase in coefficient.phases
    ]
    if len(phases) == 1:
        return (
            "The only nonzero free chord contributes in one formal direction, so "
            "positive multiplicities cannot close the boundary."
        )
    if len(phases) == 2 and phases[0].multiplicity != phases[1].multiplicity:
        return (
            "The only free chord contributes in two formal directions with unequal "
            "multiplicities; two vector groups cannot cancel unless their magnitudes match."
        )
    return None


def analyze_joint_translations(
    equations: Sequence[TranslationEquation],
) -> JointTranslationAnalysis:
    for equation in equations:
        obstruction = _single_equation_translation_obstruction(equation)
        if obstruction is not None:
            return JointTranslationAnalysis(
                exact_obstruction=True,
                status="discarded_by_individual_boundary",
                reason=f"{equation.boundary}: {obstruction}",
            )

    return JointTranslationAnalysis(
        exact_obstruction=False,
        status="shared_system_constructed_not_fully_solved",
        reason=None,
    )


def build_joint_boundary_system(
    case: base.PlacementCase,
    state: base.SolverState,
    angle_solution: Optional[angles.AngleSolution] = None,
) -> JointBoundarySystem:
    """Build and partially solve the shared inner/outer closure system."""
    if state.equations:
        raise ValueError("Joint boundary construction requires a terminal state")

    solution = angle_solution or turning.complete_angle_solution(case, state)
    inner = build_inner_boundary(case, state, solution)
    outer = build_outer_boundary(case, state, solution)
    curve_turn_solution = projection_curve_turn_constraints(case, state)
    raw_rotation_equations = (
        boundary_rotation_equation(inner),
        boundary_rotation_equation(outer),
    )
    rotation_equations = tuple(
        RotationEquation(
            boundary=equation.boundary,
            lhs=apply_curve_turn_solution(equation.lhs, curve_turn_solution),
            target_pi=equation.target_pi,
        )
        for equation in raw_rotation_equations
    )
    raw_translation_equations = (
        boundary_translation_equation(inner),
        boundary_translation_equation(outer),
    )
    translation_equations = tuple(
        apply_curve_turns_to_translation(equation, curve_turn_solution)
        for equation in raw_translation_equations
    )
    return JointBoundarySystem(
        inner_boundary=inner,
        outer_boundary=outer,
        curve_turn_solution=curve_turn_solution,
        rotation_equations=rotation_equations,
        translation_equations=translation_equations,
        rotation_analysis=analyze_joint_rotations(rotation_equations),
        translation_analysis=analyze_joint_translations(translation_equations),
    )
