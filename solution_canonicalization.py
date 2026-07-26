#!/usr/bin/env python3
"""Canonical keys for complete decorated formal solutions.

The canonicalized object is not only the closed contour word.  It also contains
both copy-to-prototype contact mappings after terminal substitution, including
segment-to-segment pairing, traversal orientation, and direct/reflected parity.

Equivalence currently quotients by:

* exchange of P0 and P1;
* cyclic change of the distinguished pole used as the printed origin;
* independent renaming/reorientation of formal curve variables;
* signed renaming of point-angle classes;
* permutation of the two identical copies;
* global mirror reflection of the complete configuration.

A global mirror is allowed, but changing the parity of only one copy is not.
The implementation deliberately does not recognize parametric cycle families.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import profile_formatter
import symbolic_enumerator as base


SCHEMA_VERSION = "decorated-solution-v1"


@dataclass(frozen=True)
class DirectedSegmentRef:
    position: int
    orientation: int

    def to_tuple(self) -> Tuple[int, int]:
        return self.position, self.orientation


@dataclass(frozen=True)
class ContactMapping:
    source_start_boundary: int
    source_end_boundary: int
    target_start_boundary: int
    target_end_boundary: int
    mirror_sign: int
    pairs: Tuple[Tuple[DirectedSegmentRef, DirectedSegmentRef], ...]


@dataclass(frozen=True)
class PointDecoration:
    class_id: str
    sign: int
    fixed_zero: bool
    pole: Optional[str]


@dataclass(frozen=True)
class DecoratedSolution:
    segments: Tuple[base.Literal, ...]
    points: Tuple[PointDecoration, ...]
    mappings: Tuple[ContactMapping, ContactMapping]


@dataclass(frozen=True)
class NormalizedLiteral:
    variable: str
    inverse: bool

    @property
    def text(self) -> str:
        return self.variable + ("^-1" if self.inverse else "")

    def flipped(self) -> "NormalizedLiteral":
        return NormalizedLiteral(self.variable, not self.inverse)


@dataclass(frozen=True)
class NormalizedPoint:
    class_id: str
    sign: int
    fixed_zero: bool
    pole: Optional[str]

    @property
    def angle_text(self) -> str:
        if self.fixed_zero:
            return f"{self.class_id}=0"
        return self.class_id if self.sign >= 0 else f"-{self.class_id}"


@dataclass(frozen=True)
class NormalizedDecoratedSolution:
    segments: Tuple[NormalizedLiteral, ...]
    points: Tuple[NormalizedPoint, ...]
    mappings: Tuple[ContactMapping, ...]
    transform_label: str
    serialized: str

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def variable_count(self) -> int:
        return len({literal.variable for literal in self.segments})

    @property
    def free_angle_count(self) -> int:
        return len({point.class_id for point in self.points if not point.fixed_zero})

    @property
    def fixed_zero_count(self) -> int:
        return len({point.class_id for point in self.points if point.fixed_zero})

    def cycle_record(self) -> List[Tuple[str, str, str]]:
        return [
            (point.pole or "", point.angle_text, literal.text)
            for literal, point in zip(self.segments, self.points)
        ]


@dataclass(frozen=True)
class CanonicalSolution:
    key: str
    canonical_json: str
    transform_label: str
    terminal_mapping: Mapping[str, object]

    def to_record(self, *, include_canonical_json: bool = False) -> Dict[str, object]:
        record: Dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "key": self.key,
            "global_mirror_identified": True,
            "pole_exchange_identified": True,
            "copy_permutation_identified": True,
            "curve_renaming_and_reorientation_identified": True,
            "signed_angle_class_renaming_identified": True,
            "copy_mapping_included": True,
            "parametric_cycle_families_identified": False,
            "canonical_witness_transform": self.transform_label,
        }
        if include_canonical_json:
            record["canonical_json"] = self.canonical_json
        return record


def _expand_literal(literal: base.Literal, environment: Mapping[str, base.Word]) -> base.Word:
    word = environment[literal.variable]
    return base.inverse_word(word) if literal.inverse else word


def _expand_word(word: Sequence[base.Literal], environment: Mapping[str, base.Word]) -> base.Word:
    output: List[base.Literal] = []
    for literal in word:
        output.extend(_expand_literal(literal, environment))
    return tuple(output)


def _initial_boundary_to_terminal(
    cycle: Sequence[base.Literal],
    environment: Mapping[str, base.Word],
) -> Tuple[int, ...]:
    boundaries = [0]
    total = 0
    for literal in cycle:
        total += len(_expand_literal(literal, environment))
        boundaries.append(total)
    return tuple(boundaries)


def _directed_factor_refs(
    start: int,
    end: int,
    direction: int,
    segment_count: int,
) -> Tuple[DirectedSegmentRef, ...]:
    if start == end:
        return ()
    current = start
    output: List[DirectedSegmentRef] = []
    if direction == base.FORWARD:
        while current != end:
            output.append(DirectedSegmentRef(current, 1))
            current = (current + 1) % segment_count
    else:
        while current != end:
            position = (current - 1) % segment_count
            output.append(DirectedSegmentRef(position, -1))
            current = position
    return tuple(output)


def _point_sign(expression: str, fixed_zero: bool) -> int:
    if fixed_zero:
        return 0
    return -1 if expression.startswith("-") else 1


def _build_mapping(
    *,
    source_start: int,
    source_end: int,
    target_start: int,
    target_end: int,
    target_direction: int,
    mirror_sign: int,
    segment_count: int,
) -> ContactMapping:
    source = _directed_factor_refs(
        source_start, source_end, base.FORWARD, segment_count
    )
    target = _directed_factor_refs(
        target_start, target_end, target_direction, segment_count
    )
    if len(source) != len(target):
        raise ValueError(
            "Terminal contact mapping length mismatch: "
            f"source has {len(source)} segments, target has {len(target)}"
        )
    return ContactMapping(
        source_start_boundary=source_start,
        source_end_boundary=source_end,
        target_start_boundary=target_start,
        target_end_boundary=target_end,
        mirror_sign=mirror_sign,
        pairs=tuple(zip(source, target)),
    )


def build_decorated_solution(
    case: base.PlacementCase,
    state: base.SolverState,
    formal_profile: profile_formatter.FormalContourProfile,
) -> DecoratedSolution:
    if state.equations:
        raise ValueError("Canonicalization requires a terminal solver state")
    environment = state.environment_map()
    segments = _expand_word(case.cycle_word, environment)
    if len(segments) != len(formal_profile.point_occurrences):
        raise ValueError(
            "Formal point/segment count mismatch during decorated canonicalization"
        )

    points = tuple(
        PointDecoration(
            class_id=point.angle_class,
            sign=_point_sign(point.expression, point.fixed_zero),
            fixed_zero=point.fixed_zero,
            pole=point.point if point.point in ("P0", "P1") else None,
        )
        for point in formal_profile.point_occurrences
    )

    initial_boundaries = _initial_boundary_to_terminal(case.cycle_word, environment)
    marker_boundaries = case.marker_boundary_map()
    p0 = 0
    p1 = initial_boundaries[len(case.a_word)]
    segment_count = len(segments)

    def terminal_marker(marker: str) -> int:
        initial_boundary = marker_boundaries[marker]
        return initial_boundaries[initial_boundary] % segment_count

    mapping_a = _build_mapping(
        source_start=p0,
        source_end=p1,
        target_start=terminal_marker("A_start"),
        target_end=terminal_marker("A_end"),
        target_direction=case.a_direction,
        mirror_sign=case.a_mirror_sign,
        segment_count=segment_count,
    )
    mapping_b = _build_mapping(
        source_start=p1,
        source_end=p0,
        target_start=terminal_marker("B_start"),
        target_end=terminal_marker("B_end"),
        target_direction=case.b_direction,
        mirror_sign=case.b_mirror_sign,
        segment_count=segment_count,
    )
    return DecoratedSolution(
        segments=segments,
        points=points,
        mappings=(mapping_a, mapping_b),
    )


def _transform_boundary(old_boundary: int, origin: int, reversed_cycle: bool, size: int) -> int:
    if not reversed_cycle:
        return (old_boundary - origin) % size
    return (origin - old_boundary) % size


def _transform_directed_ref(
    reference: DirectedSegmentRef,
    origin: int,
    reversed_cycle: bool,
    size: int,
) -> DirectedSegmentRef:
    if not reversed_cycle:
        return DirectedSegmentRef(
            (reference.position - origin) % size,
            reference.orientation,
        )
    return DirectedSegmentRef(
        (origin - reference.position - 1) % size,
        -reference.orientation,
    )


def _transform_mapping(
    mapping: ContactMapping,
    origin: int,
    reversed_cycle: bool,
    size: int,
) -> ContactMapping:
    transformed_pairs = [
        (
            _transform_directed_ref(source, origin, reversed_cycle, size),
            _transform_directed_ref(target, origin, reversed_cycle, size),
        )
        for source, target in mapping.pairs
    ]
    source_start = _transform_boundary(
        mapping.source_start_boundary, origin, reversed_cycle, size
    )
    source_end = _transform_boundary(
        mapping.source_end_boundary, origin, reversed_cycle, size
    )
    target_start = _transform_boundary(
        mapping.target_start_boundary, origin, reversed_cycle, size
    )
    target_end = _transform_boundary(
        mapping.target_end_boundary, origin, reversed_cycle, size
    )

    if reversed_cycle:
        # Reorient the copied contour back to the standard positive boundary
        # convention.  Both sides of every segment pairing are reversed.
        transformed_pairs = [
            (
                DirectedSegmentRef(source.position, -source.orientation),
                DirectedSegmentRef(target.position, -target.orientation),
            )
            for source, target in reversed(transformed_pairs)
        ]
        source_start, source_end = source_end, source_start
        target_start, target_end = target_end, target_start

    return ContactMapping(
        source_start_boundary=source_start,
        source_end_boundary=source_end,
        target_start_boundary=target_start,
        target_end_boundary=target_end,
        mirror_sign=mapping.mirror_sign,
        pairs=tuple(transformed_pairs),
    )


def _transformed_cycle(
    solution: DecoratedSolution,
    origin: int,
    reversed_cycle: bool,
) -> Tuple[List[base.Literal], List[PointDecoration]]:
    size = len(solution.segments)
    segments: List[Optional[base.Literal]] = [None] * size
    points: List[Optional[PointDecoration]] = [None] * size

    original_poles = {
        index: point.pole
        for index, point in enumerate(solution.points)
        if point.pole is not None
    }
    if origin not in original_poles:
        raise ValueError("Canonical origin must be one of the two poles")
    other_poles = [index for index in original_poles if index != origin]
    if len(other_poles) != 1:
        raise ValueError("Decorated contour must contain exactly two distinct poles")
    other_pole = other_poles[0]

    for old_position, literal in enumerate(solution.segments):
        if not reversed_cycle:
            new_position = (old_position - origin) % size
            new_literal = literal
        else:
            new_position = (origin - old_position - 1) % size
            new_literal = literal.flipped()
        segments[new_position] = new_literal

    for old_boundary, point in enumerate(solution.points):
        new_boundary = _transform_boundary(
            old_boundary, origin, reversed_cycle, size
        )
        sign = point.sign
        if reversed_cycle and sign:
            sign = -sign
        pole = None
        if old_boundary == origin:
            pole = "P0"
        elif old_boundary == other_pole:
            pole = "P1"
        points[new_boundary] = PointDecoration(
            class_id=point.class_id,
            sign=sign,
            fixed_zero=point.fixed_zero,
            pole=pole,
        )

    if any(item is None for item in segments) or any(item is None for item in points):
        raise RuntimeError("Incomplete dihedral contour transformation")
    return [item for item in segments if item is not None], [item for item in points if item is not None]


def _normalize_cycle(
    segments: Sequence[base.Literal],
    points: Sequence[PointDecoration],
) -> Tuple[Tuple[NormalizedLiteral, ...], Tuple[NormalizedPoint, ...]]:
    variable_names: Dict[str, Tuple[str, bool]] = {}
    angle_names: Dict[str, Tuple[str, int]] = {}
    normalized_segments: List[NormalizedLiteral] = []
    normalized_points: List[NormalizedPoint] = []

    for literal, point in zip(segments, points):
        if literal.variable not in variable_names:
            variable_names[literal.variable] = (
                f"V{len(variable_names)}",
                literal.inverse,
            )
        variable_name, first_inverse = variable_names[literal.variable]
        normalized_segments.append(
            NormalizedLiteral(variable_name, literal.inverse ^ first_inverse)
        )

        if point.class_id not in angle_names:
            orientation = 0 if point.fixed_zero else (point.sign or 1)
            angle_names[point.class_id] = (f"a{len(angle_names)}", orientation)
        angle_name, class_orientation = angle_names[point.class_id]
        if point.fixed_zero:
            normalized_sign = 0
        else:
            normalized_sign = point.sign * class_orientation
        normalized_points.append(
            NormalizedPoint(
                class_id=angle_name,
                sign=normalized_sign,
                fixed_zero=point.fixed_zero,
                pole=point.pole,
            )
        )
    return tuple(normalized_segments), tuple(normalized_points)


def _normalize_cycle_tokens(
    segments: Sequence[base.Literal],
    points: Sequence[PointDecoration],
) -> List[Tuple[str, str, str]]:
    normalized_segments, normalized_points = _normalize_cycle(segments, points)
    return [
        (point.pole or "", point.angle_text, literal.text)
        for literal, point in zip(normalized_segments, normalized_points)
    ]


def normalized_variants(solution: DecoratedSolution) -> Tuple[NormalizedDecoratedSolution, ...]:
    if not solution.segments:
        raise ValueError("Cannot normalize an empty contour")
    if len(solution.segments) != len(solution.points):
        raise ValueError("Every contour segment must have one preceding point")
    pole_positions = [
        index for index, point in enumerate(solution.points) if point.pole is not None
    ]
    if len(pole_positions) != 2:
        raise ValueError("Decorated contour must contain exactly two poles")

    output: List[NormalizedDecoratedSolution] = []
    seen: set[str] = set()
    size = len(solution.segments)
    for reversed_cycle in (False, True):
        for pole_choice, origin in enumerate(pole_positions):
            transformed_segments, transformed_points = _transformed_cycle(
                solution, origin, reversed_cycle
            )
            normalized_segments, normalized_points = _normalize_cycle(
                transformed_segments, transformed_points
            )
            mappings = tuple(
                sorted(
                    (
                        _transform_mapping(mapping, origin, reversed_cycle, size)
                        for mapping in solution.mappings
                    ),
                    key=_mapping_tuple,
                )
            )
            payload = {
                "schema": SCHEMA_VERSION,
                "cycle": [
                    (point.pole or "", point.angle_text, literal.text)
                    for literal, point in zip(normalized_segments, normalized_points)
                ],
                "mappings": [_mapping_tuple(mapping) for mapping in mappings],
            }
            serialized = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            if serialized in seen:
                continue
            seen.add(serialized)
            label = (
                ("mirror" if reversed_cycle else "direct")
                + f"_pole_choice_{pole_choice}"
            )
            output.append(
                NormalizedDecoratedSolution(
                    segments=normalized_segments,
                    points=normalized_points,
                    mappings=mappings,
                    transform_label=label,
                    serialized=serialized,
                )
            )
    return tuple(output)


def canonical_normalized_solution(
    solution: DecoratedSolution,
) -> NormalizedDecoratedSolution:
    return min(normalized_variants(solution), key=lambda item: item.serialized)


def _mapping_tuple(mapping: ContactMapping) -> Tuple[object, ...]:
    return (
        mapping.source_start_boundary,
        mapping.source_end_boundary,
        mapping.target_start_boundary,
        mapping.target_end_boundary,
        mapping.mirror_sign,
        tuple(
            (
                source.position,
                source.orientation,
                target.position,
                target.orientation,
            )
            for source, target in mapping.pairs
        ),
    )


def _terminal_mapping_record(solution: DecoratedSolution) -> Dict[str, object]:
    mappings = []
    for index, mapping in enumerate(solution.mappings):
        mappings.append(
            {
                "copy_index": index,
                "source_start_boundary": mapping.source_start_boundary,
                "source_end_boundary": mapping.source_end_boundary,
                "target_start_boundary": mapping.target_start_boundary,
                "target_end_boundary": mapping.target_end_boundary,
                "mirror_sign": mapping.mirror_sign,
                "isometry": "direct" if mapping.mirror_sign == base.DIRECT else "reflected",
                "segment_pairs": [
                    {
                        "source_position": source.position,
                        "source_orientation": source.orientation,
                        "target_position": target.position,
                        "target_orientation": target.orientation,
                    }
                    for source, target in mapping.pairs
                ],
            }
        )
    return {
        "schema_version": "terminal-contact-mapping-v1",
        "segment_count": len(solution.segments),
        "mappings": mappings,
    }


def terminal_mapping_record(
    case: base.PlacementCase,
    state: base.SolverState,
    formal_profile: profile_formatter.FormalContourProfile,
) -> Dict[str, object]:
    """Export the terminal contact mapping independently of canonicalization.

    The geometric solver needs this mapping even when decorated-solution
    canonicalization is disabled, because it determines how every formal curve
    occurrence must transform under the two copy isometries.
    """
    return _terminal_mapping_record(
        build_decorated_solution(case, state, formal_profile)
    )


def canonicalize_decorated_data(solution: DecoratedSolution) -> CanonicalSolution:
    canonical = canonical_normalized_solution(solution)
    key = hashlib.sha256(canonical.serialized.encode("ascii")).hexdigest()
    return CanonicalSolution(
        key=key,
        canonical_json=canonical.serialized,
        transform_label=canonical.transform_label,
        terminal_mapping=_terminal_mapping_record(solution),
    )


def canonicalize_terminal_solution(
    case: base.PlacementCase,
    state: base.SolverState,
    formal_profile: profile_formatter.FormalContourProfile,
) -> CanonicalSolution:
    return canonicalize_decorated_data(
        build_decorated_solution(case, state, formal_profile)
    )
