#!/usr/bin/env python3
"""Exact symbolic detector for forced coincidences of contour points.

This module can be used standalone and is also consumed by the experimental
inner/outer-boundary pipeline.

Model
-----
A free oriented curve variable ``X`` carries:

* a total tangent rotation ``Kappa[X]``;
* an endpoint displacement vector ``D[X]`` expressed in the start-tangent
  frame of ``X``.

An occurrence of ``X`` contributes ``R(heading) D[X]`` to the current point.
An occurrence of ``X^-1`` contributes
``R(heading - Kappa[X]) D[X]``. This is the reversed curve expressed in its
own start-tangent frame; it is not assumed to retrace the previous occurrence.

The detector constructs a formal vector expression for every contour point.
Two occurrences are declared forcibly coincident only when:

* their formal vector expressions are exactly identical; or
* an explicit equality constraint, possibly through transitive closure,
  identifies them.

The result is sound but deliberately incomplete. Trigonometric identities or
nonlinear consequences that are not exact formal cancellations are not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


ZERO = Fraction(0, 1)
HALF_TURN_PI_UNITS = Fraction(1, 1)
FULL_TURN_PI_UNITS = Fraction(2, 1)


def _as_fraction(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value, 1)


@dataclass(frozen=True, order=True)
class AngleForm:
    """Affine angle expression measured in multiples of pi.

    Example:

        pi_constant = 1
        coefficients = (("a0", 1), ("Kappa[X]", -1))

    represents ``pi + a0 - Kappa[X]``.
    """

    pi_constant: Fraction = ZERO
    coefficients: Tuple[Tuple[str, Fraction], ...] = ()

    @staticmethod
    def zero() -> "AngleForm":
        return AngleForm()

    @staticmethod
    def variable(name: str, coefficient: int | Fraction = 1) -> "AngleForm":
        return AngleForm.from_mapping({name: coefficient})

    @staticmethod
    def from_mapping(
        coefficients: Mapping[str, int | Fraction],
        *,
        pi_constant: int | Fraction = 0,
    ) -> "AngleForm":
        normalized = tuple(
            sorted(
                (name, _as_fraction(value))
                for name, value in coefficients.items()
                if _as_fraction(value) != 0
            )
        )
        return AngleForm(_as_fraction(pi_constant), normalized)

    def to_mapping(self) -> Dict[str, Fraction]:
        return dict(self.coefficients)

    def add(self, other: "AngleForm") -> "AngleForm":
        values = self.to_mapping()
        for name, value in other.coefficients:
            values[name] = values.get(name, ZERO) + value
        return AngleForm.from_mapping(
            values,
            pi_constant=self.pi_constant + other.pi_constant,
        )

    def subtract(self, other: "AngleForm") -> "AngleForm":
        return self.add(other.scale(-1))

    def scale(self, factor: int | Fraction) -> "AngleForm":
        factor_fraction = _as_fraction(factor)
        return AngleForm.from_mapping(
            {
                name: value * factor_fraction
                for name, value in self.coefficients
            },
            pi_constant=self.pi_constant * factor_fraction,
        )

    def add_variable(
        self,
        name: str,
        coefficient: int | Fraction = 1,
    ) -> "AngleForm":
        return self.add(AngleForm.variable(name, coefficient))

    def phasor_normal_form(self) -> Tuple[int, "AngleForm"]:
        """Return ``sign, phase`` with phase constant in [0, 1) pi units.

        A phase shift by pi is absorbed as a coefficient sign because
        ``R(phi + pi) D = -R(phi) D``. A full turn is discarded.
        """
        reduced = self.pi_constant % FULL_TURN_PI_UNITS
        sign = 1
        if reduced >= HALF_TURN_PI_UNITS:
            reduced -= HALF_TURN_PI_UNITS
            sign = -1
        return sign, AngleForm(reduced, self.coefficients)

    def to_text(self) -> str:
        terms: list[str] = []
        if self.pi_constant:
            if self.pi_constant == 1:
                terms.append("pi")
            elif self.pi_constant == -1:
                terms.append("-pi")
            else:
                terms.append(f"{self.pi_constant}*pi")

        for name, coefficient in self.coefficients:
            if coefficient == 1:
                term = name
            elif coefficient == -1:
                term = f"-{name}"
            else:
                term = f"{coefficient}*{name}"
            terms.append(term)

        if not terms:
            return "0"

        text = terms[0]
        for term in terms[1:]:
            text += " - " + term[1:] if term.startswith("-") else " + " + term
        return text


@dataclass(frozen=True, order=True)
class VectorBasis:
    curve_variable: str
    phase: AngleForm
    conjugated: bool = False

    def to_text(self) -> str:
        chord = (
            f"conj(D[{self.curve_variable}])"
            if self.conjugated
            else f"D[{self.curve_variable}]"
        )
        return f"R({self.phase.to_text()}){chord}"


@dataclass(frozen=True)
class VectorExpression:
    """Integer linear combination of formally rotated curve chords."""

    terms: Tuple[Tuple[VectorBasis, int], ...] = ()

    @staticmethod
    def zero() -> "VectorExpression":
        return VectorExpression()

    @staticmethod
    def from_mapping(values: Mapping[VectorBasis, int]) -> "VectorExpression":
        return VectorExpression(
            tuple(sorted((basis, value) for basis, value in values.items() if value))
        )

    def to_mapping(self) -> Dict[VectorBasis, int]:
        return dict(self.terms)

    def add_chord(
        self,
        curve_variable: str,
        phase: AngleForm,
        coefficient: int = 1,
        *,
        conjugated: bool = False,
    ) -> "VectorExpression":
        sign, normalized_phase = phase.phasor_normal_form()
        basis = VectorBasis(curve_variable, normalized_phase, conjugated)
        values = self.to_mapping()
        values[basis] = values.get(basis, 0) + coefficient * sign
        return VectorExpression.from_mapping(values)

    def add(self, other: "VectorExpression") -> "VectorExpression":
        values = self.to_mapping()
        for basis, coefficient in other.terms:
            values[basis] = values.get(basis, 0) + coefficient
        return VectorExpression.from_mapping(values)

    def subtract(self, other: "VectorExpression") -> "VectorExpression":
        values = self.to_mapping()
        for basis, coefficient in other.terms:
            values[basis] = values.get(basis, 0) - coefficient
        return VectorExpression.from_mapping(values)

    @property
    def is_zero(self) -> bool:
        return not self.terms

    def to_text(self) -> str:
        if not self.terms:
            return "0"

        pieces: list[str] = []
        for basis, coefficient in self.terms:
            if coefficient == 1:
                term = basis.to_text()
            elif coefficient == -1:
                term = f"-{basis.to_text()}"
            else:
                term = f"{coefficient}*{basis.to_text()}"
            pieces.append(term)

        text = pieces[0]
        for term in pieces[1:]:
            text += " - " + term[1:] if term.startswith("-") else " + " + term
        return text


@dataclass(frozen=True)
class CurveOccurrence:
    variable: str
    inverse: bool = False

    def to_text(self) -> str:
        return f"{self.variable}^-1" if self.inverse else self.variable


@dataclass(frozen=True)
class PointEquality:
    left_index: int
    right_index: int
    reason: str = "explicit point equality"


@dataclass(frozen=True)
class PositionRecord:
    index: int
    label: str
    expression: VectorExpression

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "label": self.label,
            "expression": self.expression.to_text(),
        }


@dataclass(frozen=True)
class ForcedCoincidenceClass:
    member_indices: Tuple[int, ...]
    member_labels: Tuple[str, ...]
    violating_pairs: Tuple[Tuple[int, int], ...]
    sources: Tuple[str, ...]
    common_expression: Optional[VectorExpression]
    contains_consecutive_points: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "member_indices": list(self.member_indices),
            "member_labels": list(self.member_labels),
            "violating_pairs": [list(pair) for pair in self.violating_pairs],
            "sources": list(self.sources),
            "common_expression": (
                None if self.common_expression is None else self.common_expression.to_text()
            ),
            "contains_consecutive_points": self.contains_consecutive_points,
        }


@dataclass(frozen=True)
class ForcedPointCoincidenceAnalysis:
    positions: Tuple[PositionRecord, ...]
    coincidence_classes: Tuple[ForcedCoincidenceClass, ...]
    passes_filter: bool
    status: str
    discard_reason: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "positions": [position.to_dict() for position in self.positions],
            "coincidence_classes": [
                coincidence.to_dict() for coincidence in self.coincidence_classes
            ],
            "passes_filter": self.passes_filter,
            "status": self.status,
            "discard_reason": self.discard_reason,
            "interpretation": (
                "A rejection is exact. A retained result only means that no forced "
                "point coincidence was proved by formal vector cancellation or the "
                "provided explicit equalities."
            ),
        }


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        else:
            self.parent[right_root] = left_root
            if self.rank[left_root] == self.rank[right_root]:
                self.rank[left_root] += 1


def curve_rotation_variable(curve_variable: str) -> str:
    return f"Kappa[{curve_variable}]"


def build_symbolic_point_positions(
    segments: Sequence[CurveOccurrence],
    turns_after_segment: Sequence[AngleForm],
    *,
    point_labels: Optional[Sequence[str]] = None,
    initial_heading: AngleForm = AngleForm(),
    zero_displacement_variables: Iterable[str] = (),
) -> Tuple[PositionRecord, ...]:
    """Build the formal position of every boundary between contour segments.

    ``turns_after_segment[i]`` is the point turn encountered immediately after
    ``segments[i]`` and before the next segment. The returned sequence has
    ``len(segments) + 1`` positions.
    """
    if len(segments) != len(turns_after_segment):
        raise ValueError("segments and turns_after_segment must have equal length")

    expected_points = len(segments) + 1
    labels = (
        tuple(point_labels)
        if point_labels is not None
        else tuple(f"q{index}" for index in range(expected_points))
    )
    if len(labels) != expected_points:
        raise ValueError("point_labels must contain len(segments) + 1 labels")

    zero_variables = frozenset(zero_displacement_variables)
    heading = initial_heading
    position = VectorExpression.zero()
    output = [PositionRecord(index=0, label=labels[0], expression=position)]

    for index, occurrence in enumerate(segments):
        kappa_name = curve_rotation_variable(occurrence.variable)
        chord_phase = heading
        if occurrence.inverse:
            chord_phase = chord_phase.add_variable(kappa_name, -1)

        if occurrence.variable not in zero_variables:
            position = position.add_chord(occurrence.variable, chord_phase)

        heading = heading.add_variable(kappa_name, -1 if occurrence.inverse else 1)
        heading = heading.add(turns_after_segment[index])
        output.append(
            PositionRecord(
                index=index + 1,
                label=labels[index + 1],
                expression=position,
            )
        )

    return tuple(output)


def analyze_forced_point_coincidences(
    segments: Sequence[CurveOccurrence],
    turns_after_segment: Sequence[AngleForm],
    *,
    point_labels: Optional[Sequence[str]] = None,
    initial_heading: AngleForm = AngleForm(),
    explicit_equalities: Sequence[PointEquality] = (),
    allowed_coincidences: Iterable[Tuple[int, int]] = (),
    allow_cycle_closure: bool = True,
    zero_displacement_variables: Iterable[str] = (),
) -> ForcedPointCoincidenceAnalysis:
    """Detect positions that are forced to coincide.

    Consecutive points are checked exactly like nonconsecutive points. The only
    automatic exception is the first/last pair when ``allow_cycle_closure`` is
    true. Additional intended coincidences can be supplied through
    ``allowed_coincidences``.
    """
    positions = build_symbolic_point_positions(
        segments,
        turns_after_segment,
        point_labels=point_labels,
        initial_heading=initial_heading,
        zero_displacement_variables=zero_displacement_variables,
    )
    count = len(positions)

    allowed = {
        tuple(sorted((left, right)))
        for left, right in allowed_coincidences
    }
    if allow_cycle_closure and count > 1:
        allowed.add((0, count - 1))

    dsu = _DisjointSet(count)
    edge_sources: list[Tuple[int, int, str]] = []
    if allow_cycle_closure and count > 1:
        dsu.union(0, count - 1)
        edge_sources.append((0, count - 1, "closed boundary endpoint identity"))

    expression_groups: Dict[VectorExpression, list[int]] = {}
    for position in positions:
        expression_groups.setdefault(position.expression, []).append(position.index)

    for indices in expression_groups.values():
        if len(indices) < 2:
            continue
        anchor = indices[0]
        for other in indices[1:]:
            dsu.union(anchor, other)
            edge_sources.append((anchor, other, "exact symbolic position identity"))

    for equality in explicit_equalities:
        if not 0 <= equality.left_index < count:
            raise IndexError(f"left_index out of range: {equality.left_index}")
        if not 0 <= equality.right_index < count:
            raise IndexError(f"right_index out of range: {equality.right_index}")
        dsu.union(equality.left_index, equality.right_index)
        edge_sources.append(
            (equality.left_index, equality.right_index, equality.reason)
        )

    classes: Dict[int, list[int]] = {}
    for index in range(count):
        classes.setdefault(dsu.find(index), []).append(index)

    coincidence_classes: list[ForcedCoincidenceClass] = []
    for members in classes.values():
        if len(members) < 2:
            continue

        violating_pairs = tuple(
            pair
            for pair in combinations(sorted(members), 2)
            if tuple(sorted(pair)) not in allowed
        )
        if not violating_pairs:
            continue

        member_set = set(members)
        sources = tuple(
            sorted(
                {
                    source
                    for left, right, source in edge_sources
                    if left in member_set and right in member_set
                }
            )
        )
        expressions = {positions[index].expression for index in members}
        common_expression = next(iter(expressions)) if len(expressions) == 1 else None
        contains_consecutive = any(
            abs(left - right) == 1 for left, right in violating_pairs
        )

        coincidence_classes.append(
            ForcedCoincidenceClass(
                member_indices=tuple(sorted(members)),
                member_labels=tuple(positions[index].label for index in sorted(members)),
                violating_pairs=violating_pairs,
                sources=sources,
                common_expression=common_expression,
                contains_consecutive_points=contains_consecutive,
            )
        )

    coincidence_classes.sort(key=lambda item: item.member_indices)
    passes = not coincidence_classes
    if passes:
        status = "no_forced_coincidence_proved"
        reason = None
    else:
        status = "discarded"
        first = coincidence_classes[0]
        reason = (
            "Distinct contour positions are forced to the same planar point: "
            + ", ".join(first.member_labels)
            + "."
        )

    return ForcedPointCoincidenceAnalysis(
        positions=positions,
        coincidence_classes=tuple(coincidence_classes),
        passes_filter=passes,
        status=status,
        discard_reason=reason,
    )

def _external_angle_to_local(form: object) -> AngleForm:
    """Convert the structurally compatible external-boundary angle form."""
    return AngleForm(
        pi_constant=getattr(form, "pi_constant"),
        coefficients=tuple(getattr(form, "coefficients")),
    )


def analyze_boundary_path_forced_coincidences(
    boundary: object,
    curve_turn_solution: object,
) -> ForcedPointCoincidenceAnalysis:
    """Analyze one ``external_boundary_constraints.BoundaryPath``.

    The implementation mirrors the exact symbolic prefix construction used by
    ``boundary_translation_equation``.  It therefore supports direct and
    reflected copies, including ``conj(D[X])`` terms, and first applies the
    projection-induced relations between the curve-turn variables.

    Only the repeated start/end point of the closed boundary is allowed.  Any
    other pair of distinct traversal positions with the same exact symbolic
    prefix displacement is a sound rejection.
    """
    import external_boundary_constraints as external

    segments = tuple(getattr(boundary, "segments"))
    points = tuple(getattr(boundary, "points"))
    if len(points) != len(segments) + 1:
        raise ValueError("A boundary with n segments must contain n+1 points")

    heading_external = external.AngleForm.zero()
    position = VectorExpression.zero()
    labels = tuple(
        f"{getattr(boundary, 'name')}:{index}:{point.physical_point}"
        for index, point in enumerate(points)
    )
    positions = [PositionRecord(0, labels[0], position)]

    for index, segment in enumerate(segments):
        kappa = f"Kappa[{segment.literal.variable}]"
        phase_external = heading_external
        if segment.literal.inverse:
            phase_external = phase_external.add_term(kappa, -segment.mirror_sign)
        phase_external = external.apply_curve_turn_solution(
            phase_external, curve_turn_solution
        )
        position = position.add_chord(
            segment.literal.variable,
            _external_angle_to_local(phase_external),
            conjugated=segment.conjugated_chord,
        )

        heading_external = heading_external.add_term(
            kappa, segment.physical_turn_sign
        )
        heading_external = heading_external.add(points[index + 1].turn)
        heading_external = external.apply_curve_turn_solution(
            heading_external, curve_turn_solution
        )
        positions.append(PositionRecord(index + 1, labels[index + 1], position))

    return _analyze_position_records(positions, allow_cycle_closure=True)


def _analyze_position_records(
    positions: Sequence[PositionRecord],
    *,
    allow_cycle_closure: bool,
) -> ForcedPointCoincidenceAnalysis:
    """Shared exact-identity analysis for already constructed point positions."""
    count = len(positions)
    allowed = {(0, count - 1)} if allow_cycle_closure and count > 1 else set()
    dsu = _DisjointSet(count)
    edge_sources: list[Tuple[int, int, str]] = []
    if allow_cycle_closure and count > 1:
        dsu.union(0, count - 1)
        edge_sources.append((0, count - 1, "closed boundary endpoint identity"))

    expression_groups: Dict[VectorExpression, list[int]] = {}
    for position in positions:
        expression_groups.setdefault(position.expression, []).append(position.index)
    for indices in expression_groups.values():
        if len(indices) < 2:
            continue
        anchor = indices[0]
        for other in indices[1:]:
            dsu.union(anchor, other)
            edge_sources.append((anchor, other, "exact symbolic position identity"))

    classes: Dict[int, list[int]] = {}
    for index in range(count):
        classes.setdefault(dsu.find(index), []).append(index)

    coincidence_classes: list[ForcedCoincidenceClass] = []
    for members in classes.values():
        if len(members) < 2:
            continue
        violating_pairs = tuple(
            pair
            for pair in combinations(sorted(members), 2)
            if tuple(sorted(pair)) not in allowed
        )
        if not violating_pairs:
            continue
        member_set = set(members)
        sources = tuple(
            sorted(
                {
                    source
                    for left, right, source in edge_sources
                    if left in member_set and right in member_set
                }
            )
        )
        expressions = {positions[index].expression for index in members}
        common_expression = next(iter(expressions)) if len(expressions) == 1 else None
        coincidence_classes.append(
            ForcedCoincidenceClass(
                member_indices=tuple(sorted(members)),
                member_labels=tuple(positions[index].label for index in sorted(members)),
                violating_pairs=violating_pairs,
                sources=sources,
                common_expression=common_expression,
                contains_consecutive_points=any(
                    abs(left - right) == 1 for left, right in violating_pairs
                ),
            )
        )

    coincidence_classes.sort(key=lambda item: item.member_indices)
    passes = not coincidence_classes
    reason = None if passes else (
        "Distinct contour positions are forced to the same planar point: "
        + ", ".join(coincidence_classes[0].member_labels)
        + "."
    )
    return ForcedPointCoincidenceAnalysis(
        positions=tuple(positions),
        coincidence_classes=tuple(coincidence_classes),
        passes_filter=passes,
        status="no_forced_coincidence_proved" if passes else "discarded",
        discard_reason=reason,
    )

