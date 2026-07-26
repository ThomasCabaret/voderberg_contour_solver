#!/usr/bin/env python3
"""Shared-frame symbolic geometry for the reference tile and its two copies.

The formal word and boundary modules describe contacts locally.  This module
constructs one symbolic Euclidean frame containing all three congruent copies.
For each copy, one global isometry is fixed by:

* the source contact start point on the prototype;
* the corresponding start point on the reference contact;
* the source and target tangent headings;
* the direct/reflected parity fixed by placement enumeration.

Every distinguished prototype point is then transformed by that same isometry.
The resulting model supports two sound, deliberately conservative checks:

1. unintended point identities forced by exact symbolic cancellation or by the
   transitive closure of intended contact identifications;
2. exact same-side overlap of two identically placed directed curve
   occurrences, which would put two tile interiors locally on the same side.

Additional cross-copy point contacts are reported but are not rejected: two
closed tiles may touch at an extra point without overlapping.  Likewise, this
module does not claim to detect a generic crossing of arbitrary curved arcs.
The complete contact point equations are exported for the polynomial Z3 model,
where they enforce the single global isometry pointwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import angle_constraints as angles
import external_boundary_constraints as external
import forced_point_coincidence as vectors
import symbolic_enumerator as base


SCHEMA_VERSION = "placed-copy-geometry-v1"

REFERENCE = external.REFERENCE_COPY
COPY_A = external.A_COPY
COPY_B = external.B_COPY


@dataclass(frozen=True)
class PlacedPoint:
    copy: str
    prototype_point: str
    expression: vectors.VectorExpression

    @property
    def label(self) -> str:
        return f"{self.copy}:{self.prototype_point}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "copy": self.copy,
            "prototype_point": self.prototype_point,
            "label": self.label,
            "position": self.expression.to_text(),
        }


@dataclass(frozen=True)
class PlacedSegment:
    copy: str
    prototype_segment_index: int
    variable: str
    inverse: bool
    reflected: bool
    start_point: str
    end_point: str
    start_position: vectors.VectorExpression
    end_position: vectors.VectorExpression
    start_heading: vectors.AngleForm
    end_heading: vectors.AngleForm
    interior_side: int

    @property
    def label(self) -> str:
        direction = "^-1" if self.inverse else ""
        return (
            f"{self.copy}:segment[{self.prototype_segment_index}]="
            f"{self.variable}{direction}"
        )

    def exact_directed_placement_key(self) -> Tuple[object, ...]:
        return (
            self.variable,
            self.inverse,
            self.reflected,
            self.start_position,
            self.end_position,
            self.start_heading,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "copy": self.copy,
            "prototype_segment_index": self.prototype_segment_index,
            "variable": self.variable,
            "inverse": self.inverse,
            "reflected": self.reflected,
            "start_point": self.start_point,
            "end_point": self.end_point,
            "start_position": self.start_position.to_text(),
            "end_position": self.end_position.to_text(),
            "start_heading": self.start_heading.to_text(),
            "end_heading_before_corner": self.end_heading.to_text(),
            "interior_side": self.interior_side,
        }


@dataclass(frozen=True)
class ContactPointEquation:
    projection: str
    boundary_index: int
    reference_label: str
    copy_label: str
    reference_position: vectors.VectorExpression
    copy_position: vectors.VectorExpression

    def to_dict(self) -> Dict[str, object]:
        return {
            "projection": self.projection,
            "boundary_index": self.boundary_index,
            "reference_label": self.reference_label,
            "copy_label": self.copy_label,
            "reference_position": self.reference_position.to_text(),
            "copy_position": self.copy_position.to_text(),
        }


@dataclass(frozen=True)
class ForcedPointClass:
    labels: Tuple[str, ...]
    sources: Tuple[str, ...]
    crosses_copies: bool
    intended: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "labels": list(self.labels),
            "sources": list(self.sources),
            "crosses_copies": self.crosses_copies,
            "intended": self.intended,
        }


@dataclass(frozen=True)
class SameSideOverlap:
    first_segment: str
    second_segment: str
    common_start: str
    common_end: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "first_segment": self.first_segment,
            "second_segment": self.second_segment,
            "common_start": self.common_start,
            "common_end": self.common_end,
            "reason": (
                "The same directed curve occurrence is forced into the same place "
                "with both tile interiors on the same local side."
            ),
        }


@dataclass(frozen=True)
class PlacedCopyGeometryAnalysis:
    points: Tuple[PlacedPoint, ...]
    segments: Tuple[PlacedSegment, ...]
    contact_point_equations: Tuple[ContactPointEquation, ...]
    point_classes: Tuple[ForcedPointClass, ...]
    same_side_overlaps: Tuple[SameSideOverlap, ...]
    passes_filter: bool
    status: str
    discard_reason: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "points": [item.to_dict() for item in self.points],
            "segments": [item.to_dict() for item in self.segments],
            "contact_point_equations": [
                item.to_dict() for item in self.contact_point_equations
            ],
            "point_classes": [item.to_dict() for item in self.point_classes],
            "same_side_overlaps": [
                item.to_dict() for item in self.same_side_overlaps
            ],
            "passes_filter": self.passes_filter,
            "status": self.status,
            "discard_reason": self.discard_reason,
            "scope": {
                "exact": True,
                "rejects_extra_cross_copy_point_contacts": False,
                "detects_generic_curved_arc_crossings": False,
                "detects_exact_same_side_directed_arc_overlap": True,
                "enforces_contact_points_in_polynomial_backend": True,
            },
        }


class _DSU:
    def __init__(self, labels: Iterable[str]) -> None:
        self.parent = {label: label for label in labels}

    def find(self, label: str) -> str:
        parent = self.parent[label]
        if parent != label:
            self.parent[label] = self.find(parent)
        return self.parent[label]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _local_angle(form: external.AngleForm) -> vectors.AngleForm:
    return vectors.AngleForm(
        pi_constant=form.pi_constant,
        coefficients=tuple(form.coefficients),
    )


def _external_angle(form: vectors.AngleForm) -> external.AngleForm:
    return external.AngleForm(
        coefficients=tuple(form.coefficients),
        pi_constant=form.pi_constant,
    )


def _walk_reference_boundary(
    boundary: external.BoundaryPath,
    curve_turn_solution: external.CurveTurnSolution,
) -> Tuple[
    Tuple[PlacedPoint, ...],
    Tuple[PlacedSegment, ...],
    Tuple[vectors.AngleForm, ...],
]:
    heading = external.AngleForm.zero()
    position = vectors.VectorExpression.zero()
    points: List[PlacedPoint] = []
    segments: List[PlacedSegment] = []
    outgoing_headings: List[vectors.AngleForm] = []

    for index, segment in enumerate(boundary.segments):
        heading = external.apply_curve_turn_solution(heading, curve_turn_solution)
        point = boundary.points[index]
        points.append(PlacedPoint(REFERENCE, point.physical_point, position))
        outgoing_headings.append(_local_angle(heading))

        chord_phase = heading
        if segment.literal.inverse:
            chord_phase = chord_phase.add_term(
                f"Kappa[{segment.literal.variable}]", -segment.mirror_sign
            )
        chord_phase = external.apply_curve_turn_solution(
            chord_phase, curve_turn_solution
        )
        end_position = position.add_chord(
            segment.literal.variable,
            _local_angle(chord_phase),
            conjugated=segment.conjugated_chord,
        )
        end_heading = heading.add_term(
            f"Kappa[{segment.literal.variable}]", segment.physical_turn_sign
        )
        end_heading = external.apply_curve_turn_solution(
            end_heading, curve_turn_solution
        )
        segments.append(
            PlacedSegment(
                copy=REFERENCE,
                prototype_segment_index=index,
                variable=segment.literal.variable,
                inverse=segment.literal.inverse,
                reflected=False,
                start_point=point.physical_point,
                end_point=boundary.points[index + 1].physical_point,
                start_position=position,
                end_position=end_position,
                start_heading=_local_angle(heading),
                end_heading=_local_angle(end_heading),
                interior_side=1,
            )
        )
        position = end_position
        heading = end_heading.add(boundary.points[index + 1].turn)

    # The repeated closure point is represented only once in the shared-frame
    # point set.  Its equality to the start is already a boundary closure law.
    return tuple(points), tuple(segments), tuple(outgoing_headings)


@dataclass(frozen=True)
class LocatedPath:
    point_indices: Tuple[int, ...]
    segment_indices: Tuple[int, ...]
    traversal_sign: int
    start_heading: vectors.AngleForm


def _locate_expanded_path(
    full_segments: Sequence[PlacedSegment],
    full_points: Sequence[PlacedPoint],
    reference_headings: Sequence[vectors.AngleForm],
    path: angles.ExpandedPath,
) -> LocatedPath:
    count = len(full_segments)
    if not path.segments:
        raise ValueError("A contact path must contain at least one segment")
    traversal_sign = path.points[0].orientation
    if traversal_sign not in (base.FORWARD, base.REVERSE):
        raise ValueError("Unsupported path traversal orientation")

    for point_start in range(count):
        if full_points[point_start].prototype_point != path.points[0].point:
            continue
        if traversal_sign == base.FORWARD:
            segment_indices = tuple(
                (point_start + offset) % count
                for offset in range(len(path.segments))
            )
            point_indices = tuple(
                (point_start + offset) % count
                for offset in range(len(path.segments) + 1)
            )
            literals = tuple(
                base.Literal(full_segments[index].variable, full_segments[index].inverse)
                for index in segment_indices
            )
            start_heading = reference_headings[point_start]
        else:
            segment_indices = tuple(
                (point_start - 1 - offset) % count
                for offset in range(len(path.segments))
            )
            point_indices = tuple(
                (point_start - offset) % count
                for offset in range(len(path.segments) + 1)
            )
            literals = tuple(
                base.Literal(
                    full_segments[index].variable,
                    not full_segments[index].inverse,
                )
                for index in segment_indices
            )
            first_segment = full_segments[segment_indices[0]]
            start_heading = first_segment.end_heading.add(
                vectors.AngleForm(pi_constant=1)
            )
        if literals == path.segments and all(
            full_points[index].prototype_point == occurrence.point
            for index, occurrence in zip(point_indices, path.points)
        ):
            return LocatedPath(
                point_indices=point_indices,
                segment_indices=segment_indices,
                traversal_sign=traversal_sign,
                start_heading=start_heading,
            )
    raise ValueError(
        "Could not locate an expanded contact path in the prototype boundary: "
        + base.word_to_text(path.segments)
    )


def _transform_angle(
    angle: vectors.AngleForm,
    rotation: vectors.AngleForm,
    mirror_sign: int,
) -> vectors.AngleForm:
    return rotation.add(angle.scale(mirror_sign))


def _transform_vector_delta(
    delta: vectors.VectorExpression,
    rotation: vectors.AngleForm,
    mirror_sign: int,
) -> vectors.VectorExpression:
    output = vectors.VectorExpression.zero()
    for basis, coefficient in delta.terms:
        if mirror_sign == base.DIRECT:
            phase = rotation.add(basis.phase)
            conjugated = basis.conjugated
        else:
            phase = rotation.subtract(basis.phase)
            conjugated = not basis.conjugated
        output = output.add_chord(
            basis.curve_variable,
            phase,
            coefficient,
            conjugated=conjugated,
        )
    return output


def _transform_position(
    position: vectors.VectorExpression,
    source_anchor: vectors.VectorExpression,
    target_anchor: vectors.VectorExpression,
    rotation: vectors.AngleForm,
    mirror_sign: int,
) -> vectors.VectorExpression:
    return target_anchor.add(
        _transform_vector_delta(
            position.subtract(source_anchor), rotation, mirror_sign
        )
    )


def _build_copy(
    copy_name: str,
    mirror_sign: int,
    source_start_index: int,
    target_start_index: int,
    source_path_heading: vectors.AngleForm,
    target_path_heading: vectors.AngleForm,
    reference_points: Sequence[PlacedPoint],
    reference_segments: Sequence[PlacedSegment],
) -> Tuple[Tuple[PlacedPoint, ...], Tuple[PlacedSegment, ...]]:
    source_anchor = reference_points[source_start_index].expression
    target_anchor = reference_points[target_start_index].expression
    source_heading = source_path_heading
    target_heading = target_path_heading
    # A direct isometry maps h -> rho+h; a reflected one maps h -> rho-h.
    rotation = target_heading.subtract(source_heading.scale(mirror_sign))

    points = tuple(
        PlacedPoint(
            copy_name,
            point.prototype_point,
            _transform_position(
                point.expression,
                source_anchor,
                target_anchor,
                rotation,
                mirror_sign,
            ),
        )
        for point in reference_points
    )
    point_map = {point.prototype_point: point for point in points}

    segments = []
    for segment in reference_segments:
        segments.append(
            PlacedSegment(
                copy=copy_name,
                prototype_segment_index=segment.prototype_segment_index,
                variable=segment.variable,
                inverse=segment.inverse,
                reflected=(mirror_sign == base.REFLECTED),
                start_point=segment.start_point,
                end_point=segment.end_point,
                start_position=point_map[segment.start_point].expression,
                end_position=point_map[segment.end_point].expression,
                start_heading=_transform_angle(
                    segment.start_heading, rotation, mirror_sign
                ),
                end_heading=_transform_angle(
                    segment.end_heading, rotation, mirror_sign
                ),
                # Image of a CCW prototype has its interior on the left for a
                # direct isometry and on the right for a reflection.
                interior_side=mirror_sign,
            )
        )
    return points, tuple(segments)


def _path_point_indices(start: int, segment_count: int, full_count: int) -> Tuple[int, ...]:
    return tuple((start + offset) % full_count for offset in range(segment_count + 1))


def _analyze_point_classes(
    points: Sequence[PlacedPoint],
    intended_equalities: Sequence[Tuple[str, str, str]],
) -> Tuple[ForcedPointClass, ...]:
    labels = [point.label for point in points]
    dsu = _DSU(labels)
    source_edges: List[Tuple[str, str, str]] = []
    intended_pairs = set()
    for left, right, reason in intended_equalities:
        pair = tuple(sorted((left, right)))
        intended_pairs.add(pair)
        dsu.union(left, right)
        source_edges.append((left, right, reason))

    by_expression: Dict[vectors.VectorExpression, List[str]] = {}
    for point in points:
        by_expression.setdefault(point.expression, []).append(point.label)
    for expression_labels in by_expression.values():
        if len(expression_labels) < 2:
            continue
        anchor = expression_labels[0]
        for other in expression_labels[1:]:
            dsu.union(anchor, other)
            source_edges.append((anchor, other, "exact symbolic position identity"))

    classes: Dict[str, List[str]] = {}
    for label in labels:
        classes.setdefault(dsu.find(label), []).append(label)

    output = []
    for members in classes.values():
        if len(members) < 2:
            continue
        member_set = set(members)
        sources = tuple(sorted({
            reason
            for left, right, reason in source_edges
            if left in member_set and right in member_set
        }))
        copies = {label.split(":", 1)[0] for label in members}
        all_pairs_intended = all(
            tuple(sorted(pair)) in intended_pairs
            for pair in combinations(sorted(members), 2)
        )
        output.append(
            ForcedPointClass(
                labels=tuple(sorted(members)),
                sources=sources,
                crosses_copies=len(copies) > 1,
                intended=all_pairs_intended,
            )
        )
    output.sort(key=lambda item: item.labels)
    return tuple(output)


def _same_side_overlaps(
    segments: Sequence[PlacedSegment],
    allowed_contact_pairs: Iterable[Tuple[str, str]],
) -> Tuple[SameSideOverlap, ...]:
    allowed = {tuple(sorted(pair)) for pair in allowed_contact_pairs}
    groups: Dict[Tuple[object, ...], List[PlacedSegment]] = {}
    for segment in segments:
        groups.setdefault(segment.exact_directed_placement_key(), []).append(segment)

    output = []
    for group in groups.values():
        if len(group) < 2:
            continue
        for left, right in combinations(group, 2):
            if tuple(sorted((left.label, right.label))) in allowed:
                continue
            if left.copy == right.copy:
                # A repeated directed arc on one Jordan boundary is invalid.
                same_side = True
            else:
                same_side = left.interior_side == right.interior_side
            if not same_side:
                continue
            output.append(
                SameSideOverlap(
                    first_segment=left.label,
                    second_segment=right.label,
                    common_start=left.start_position.to_text(),
                    common_end=left.end_position.to_text(),
                )
            )
    output.sort(key=lambda item: (item.first_segment, item.second_segment))
    return tuple(output)


def analyze_placed_copy_geometry(
    case: base.PlacementCase,
    state: base.SolverState,
    system: external.JointBoundarySystem,
) -> PlacedCopyGeometryAnalysis:
    """Construct all three copies in one symbolic frame and apply exact checks."""
    if state.equations:
        raise ValueError("Placed-copy geometry requires a terminal solver state")

    reference_points, reference_segments, reference_headings = _walk_reference_boundary(
        system.inner_boundary, system.curve_turn_solution
    )
    environment = state.environment_map()
    endpoints = angles.initial_segment_endpoints(case)
    a_reference_path = angles.expand_initial_word(case.a_word, environment, endpoints)
    b_reference_path = angles.expand_initial_word(case.b_word, environment, endpoints)
    a_source_path = angles.expand_initial_word(case.a_target, environment, endpoints)
    b_source_path = angles.expand_initial_word(case.b_target, environment, endpoints)

    a_target = _locate_expanded_path(
        reference_segments, reference_points, reference_headings, a_reference_path
    )
    b_target = _locate_expanded_path(
        reference_segments, reference_points, reference_headings, b_reference_path
    )
    a_source = _locate_expanded_path(
        reference_segments, reference_points, reference_headings, a_source_path
    )
    b_source = _locate_expanded_path(
        reference_segments, reference_points, reference_headings, b_source_path
    )

    a_points, a_segments = _build_copy(
        COPY_A,
        case.a_mirror_sign,
        a_source.point_indices[0],
        a_target.point_indices[0],
        a_source.start_heading,
        a_target.start_heading,
        reference_points,
        reference_segments,
    )
    b_points, b_segments = _build_copy(
        COPY_B,
        case.b_mirror_sign,
        b_source.point_indices[0],
        b_target.point_indices[0],
        b_source.start_heading,
        b_target.start_heading,
        reference_points,
        reference_segments,
    )

    segment_by_copy_index = {
        (segment.copy, segment.prototype_segment_index): segment
        for segment in (*reference_segments, *a_segments, *b_segments)
    }

    intended_equalities: List[Tuple[str, str, str]] = []
    contact_equations: List[ContactPointEquation] = []
    allowed_contact_segments: List[Tuple[str, str]] = []
    contact_data = (
        ("A", COPY_A, a_target, a_source),
        ("B", COPY_B, b_target, b_source),
    )
    for projection, copy_name, target_path, source_path in contact_data:
        for boundary_index, (target_index, source_index) in enumerate(
            zip(target_path.point_indices, source_path.point_indices)
        ):
            reference_point = reference_points[target_index]
            copy_point = (
                a_points[source_index] if copy_name == COPY_A else b_points[source_index]
            )
            reason = f"intended {projection} contact point {boundary_index}"
            intended_equalities.append((reference_point.label, copy_point.label, reason))
            contact_equations.append(
                ContactPointEquation(
                    projection=projection,
                    boundary_index=boundary_index,
                    reference_label=reference_point.label,
                    copy_label=copy_point.label,
                    reference_position=reference_point.expression,
                    copy_position=copy_point.expression,
                )
            )
        for target_segment_index, source_segment_index in zip(
            target_path.segment_indices, source_path.segment_indices
        ):
            reference_segment = segment_by_copy_index[
                (REFERENCE, target_segment_index)
            ]
            copy_segment = segment_by_copy_index[
                (copy_name, source_segment_index)
            ]
            allowed_contact_segments.append(
                (reference_segment.label, copy_segment.label)
            )

    points = tuple((*reference_points, *a_points, *b_points))
    segments = tuple((*reference_segments, *a_segments, *b_segments))
    point_classes = _analyze_point_classes(points, intended_equalities)
    overlaps = _same_side_overlaps(segments, allowed_contact_segments)

    # A point class created only by intended contact identifications is valid.
    # Extra cross-copy point contacts are diagnostic.  A class containing two
    # distinct points of the same copy is a forced self-intersection.
    hard_point_classes = []
    for point_class in point_classes:
        by_copy: Dict[str, int] = {}
        for label in point_class.labels:
            copy_name = label.split(":", 1)[0]
            by_copy[copy_name] = by_copy.get(copy_name, 0) + 1
        if any(count > 1 for count in by_copy.values()):
            hard_point_classes.append(point_class)

    passes = not hard_point_classes and not overlaps
    if overlaps:
        reason = (
            "Two boundary arcs are forced into the same directed placement with "
            "both tile interiors on the same side: "
            f"{overlaps[0].first_segment} and {overlaps[0].second_segment}."
        )
        status = "discarded_by_same_side_arc_overlap"
    elif hard_point_classes:
        reason = (
            "A placed copy has two distinct prototype boundary points forced to "
            "the same planar point: " + ", ".join(hard_point_classes[0].labels)
        )
        status = "discarded_by_forced_self_coincidence"
    else:
        reason = None
        status = "no_forced_global_topology_obstruction_proved"

    return PlacedCopyGeometryAnalysis(
        points=points,
        segments=segments,
        contact_point_equations=tuple(contact_equations),
        point_classes=point_classes,
        same_side_overlaps=overlaps,
        passes_filter=passes,
        status=status,
        discard_reason=reason,
    )
