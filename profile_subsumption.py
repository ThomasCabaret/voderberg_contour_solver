#!/usr/bin/env python3
"""Exact reduction of decorated contour profiles under curve substitution.

A profile ``general`` absorbs a profile ``refined`` when every formal curve
variable of ``general`` can be replaced by a nonempty decorated path of
``refined`` so that:

* the complete cyclic contour is reproduced;
* inversion reverses the path and negates its internal turns;
* the pre-existing point-angle classes of the general profile specialize
  consistently to refined angle classes or zero;
The primary reduction concerns contour-shape families: once a refined contour
is an instance of a more general solved profile, the general profile already
supplies its own valid contact mapping for that contour.  A separate stronger
API is provided when exact refinement of both copy mappings is required.

The module is deliberately independent of the geometric filters. It operates
only on complete ``DecoratedSolution`` objects after the word and point-angle
solvers have terminated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from itertools import permutations

import solution_canonicalization as canonical


SCHEMA_VERSION = "decorated-profile-subsumption-v1"


@dataclass(frozen=True)
class PathPoint:
    class_id: str
    sign: int
    fixed_zero: bool

    def inverted(self) -> "PathPoint":
        if self.fixed_zero:
            return self
        return PathPoint(self.class_id, -self.sign, False)

    @property
    def text(self) -> str:
        if self.fixed_zero:
            return f"{self.class_id}=0"
        return self.class_id if self.sign >= 0 else f"-{self.class_id}"


@dataclass(frozen=True)
class DecoratedPath:
    segments: Tuple[canonical.NormalizedLiteral, ...]
    internal_points: Tuple[PathPoint, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("A substitution path must be nonempty")
        if len(self.internal_points) != len(self.segments) - 1:
            raise ValueError("A decorated path needs one internal point between segments")

    def inverted(self) -> "DecoratedPath":
        return DecoratedPath(
            segments=tuple(segment.flipped() for segment in reversed(self.segments)),
            internal_points=tuple(
                point.inverted() for point in reversed(self.internal_points)
            ),
        )

    @property
    def text(self) -> str:
        pieces: List[str] = []
        for index, segment in enumerate(self.segments):
            if index:
                pieces.append(f"({self.internal_points[index - 1].text})")
            pieces.append(segment.text)
        return " ".join(pieces)


@dataclass(frozen=True)
class AngleImage:
    target_class: Optional[str]
    sign: int = 0

    @property
    def text(self) -> str:
        if self.target_class is None:
            return "0"
        return self.target_class if self.sign >= 0 else f"-{self.target_class}"


@dataclass(frozen=True)
class SubsumptionCertificate:
    general_variant_label: str
    refined_variant_label: str
    variable_substitution: Mapping[str, DecoratedPath]
    angle_substitution: Mapping[str, AngleImage]
    boundary_map: Tuple[int, ...]
    scope: str = "contour_shape_family"
    copy_mappings_checked_exactly: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "general_variant": self.general_variant_label,
            "refined_variant": self.refined_variant_label,
            "nonerasing": True,
            "variable_substitution": {
                name: path.text
                for name, path in sorted(self.variable_substitution.items())
            },
            "angle_substitution": {
                name: image.text
                for name, image in sorted(self.angle_substitution.items())
            },
            "general_boundary_to_refined_boundary": list(self.boundary_map),
            "scope": self.scope,
            "copy_mappings_checked_exactly": self.copy_mappings_checked_exactly,
        }


@dataclass(frozen=True)
class ProfileReductionEntry:
    profile_id: int
    decorated_solution: canonical.DecoratedSolution
    canonical_key: str
    canonical_json: str
    all_curve_components_free: bool = True


@dataclass(frozen=True)
class AbsorptionRecord:
    absorbed_profile_id: int
    representative_profile_id: int
    certificate: SubsumptionCertificate

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "absorbed_profile_id": self.absorbed_profile_id,
            "representative_profile_id": self.representative_profile_id,
            "certificate": self.certificate.to_dict(),
        }


def _point_path(point: canonical.NormalizedPoint) -> PathPoint:
    if point.pole is not None:
        raise ValueError("A pole cannot be internal to a substitution path")
    return PathPoint(point.class_id, point.sign, point.fixed_zero)


def _extract_path(
    profile: canonical.NormalizedDecoratedSolution,
    start: int,
    end: int,
) -> DecoratedPath:
    if not (0 <= start < end <= profile.segment_count):
        raise ValueError(f"Invalid linear path interval {start}:{end}")
    return DecoratedPath(
        segments=profile.segments[start:end],
        internal_points=tuple(
            _point_path(profile.points[index])
            for index in range(start + 1, end)
        ),
    )


def _pole_boundary(profile: canonical.NormalizedDecoratedSolution, pole: str) -> int:
    matches = [index for index, point in enumerate(profile.points) if point.pole == pole]
    if len(matches) != 1:
        raise ValueError(f"Expected one {pole} boundary, got {len(matches)}")
    return matches[0]


def _angle_extension(
    general_point: canonical.NormalizedPoint,
    refined_point: canonical.NormalizedPoint,
    current: Mapping[str, AngleImage],
) -> Optional[Dict[str, AngleImage]]:
    if general_point.pole != refined_point.pole:
        return None
    if general_point.fixed_zero:
        if not refined_point.fixed_zero:
            return None
        return dict(current)

    if refined_point.fixed_zero:
        image = AngleImage(None, 0)
    else:
        image = AngleImage(
            refined_point.class_id,
            general_point.sign * refined_point.sign,
        )
    existing = current.get(general_point.class_id)
    if existing is not None and existing != image:
        return None
    output = dict(current)
    output[general_point.class_id] = image
    return output


def _expanded_ref(
    reference: canonical.DirectedSegmentRef,
    cuts: Sequence[int],
) -> Tuple[canonical.DirectedSegmentRef, ...]:
    start = cuts[reference.position]
    end = cuts[reference.position + 1]
    if reference.orientation == 1:
        return tuple(
            canonical.DirectedSegmentRef(position, 1)
            for position in range(start, end)
        )
    if reference.orientation == -1:
        return tuple(
            canonical.DirectedSegmentRef(position, -1)
            for position in range(end - 1, start - 1, -1)
        )
    raise ValueError(f"Invalid directed reference orientation {reference.orientation}")


def _mapped_boundary(boundary: int, cuts: Sequence[int], segment_count: int) -> int:
    if not (0 <= boundary < segment_count):
        raise ValueError(f"Invalid boundary {boundary} for {segment_count} segments")
    return cuts[boundary]


def _expand_mapping(
    mapping: canonical.ContactMapping,
    cuts: Sequence[int],
    general_segment_count: int,
) -> Optional[canonical.ContactMapping]:
    expanded_pairs: List[
        Tuple[canonical.DirectedSegmentRef, canonical.DirectedSegmentRef]
    ] = []
    for source, target in mapping.pairs:
        expanded_source = _expanded_ref(source, cuts)
        expanded_target = _expanded_ref(target, cuts)
        if len(expanded_source) != len(expanded_target):
            return None
        expanded_pairs.extend(zip(expanded_source, expanded_target))
    return canonical.ContactMapping(
        source_start_boundary=_mapped_boundary(
            mapping.source_start_boundary, cuts, general_segment_count
        ),
        source_end_boundary=_mapped_boundary(
            mapping.source_end_boundary, cuts, general_segment_count
        ),
        target_start_boundary=_mapped_boundary(
            mapping.target_start_boundary, cuts, general_segment_count
        ),
        target_end_boundary=_mapped_boundary(
            mapping.target_end_boundary, cuts, general_segment_count
        ),
        mirror_sign=mapping.mirror_sign,
        pairs=tuple(expanded_pairs),
    )


def _mapping_signature(mapping: canonical.ContactMapping) -> Tuple[object, ...]:
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


def _mappings_match(
    general: canonical.NormalizedDecoratedSolution,
    refined: canonical.NormalizedDecoratedSolution,
    cuts: Sequence[int],
) -> bool:
    expanded: List[canonical.ContactMapping] = []
    for mapping in general.mappings:
        value = _expand_mapping(mapping, cuts, general.segment_count)
        if value is None:
            return False
        expanded.append(value)
    return sorted(_mapping_signature(item) for item in expanded) == sorted(
        _mapping_signature(item) for item in refined.mappings
    )


def _possible_end_boundaries(
    *,
    general_index: int,
    current_refined_boundary: int,
    general_p1: int,
    refined_p1: int,
    general_segment_count: int,
    refined_segment_count: int,
    assigned_length: Optional[int],
    required_boundaries: Mapping[int, int],
) -> Iterable[int]:
    next_general_boundary = general_index + 1
    normalized_boundary = 0 if next_general_boundary == general_segment_count else next_general_boundary
    required = required_boundaries.get(normalized_boundary)
    if required is not None:
        candidates: Iterable[int] = (
            refined_segment_count if next_general_boundary == general_segment_count and required == 0 else required,
        )
    elif next_general_boundary == general_p1:
        candidates = (refined_p1,)
    elif next_general_boundary == general_segment_count:
        candidates = (refined_segment_count,)
    elif next_general_boundary < general_p1:
        remaining_segments = general_p1 - next_general_boundary
        upper = refined_p1 - remaining_segments
        candidates = range(current_refined_boundary + 1, upper + 1)
    else:
        remaining_segments = general_segment_count - next_general_boundary
        upper = refined_segment_count - remaining_segments
        candidates = range(current_refined_boundary + 1, upper + 1)

    for candidate in candidates:
        if candidate <= current_refined_boundary:
            continue
        # No image of a single general curve occurrence may cross a pole.
        if general_index < general_p1 and candidate > refined_p1:
            continue
        if general_index >= general_p1 and current_refined_boundary < refined_p1:
            continue
        if assigned_length is not None and candidate - current_refined_boundary != assigned_length:
            continue
        yield candidate


def _mapping_correspondences(
    general: canonical.NormalizedDecoratedSolution,
    refined: canonical.NormalizedDecoratedSolution,
) -> Iterable[Tuple[Tuple[int, ...], Dict[int, int]]]:
    for permutation in permutations(range(len(refined.mappings))):
        boundary_map: Dict[int, int] = {0: 0}
        valid = True
        for general_index, refined_index in enumerate(permutation):
            left = general.mappings[general_index]
            right = refined.mappings[refined_index]
            if left.mirror_sign != right.mirror_sign:
                valid = False
                break
            if len(left.pairs) > len(right.pairs):
                valid = False
                break
            constraints = (
                (left.source_start_boundary, right.source_start_boundary),
                (left.source_end_boundary, right.source_end_boundary),
                (left.target_start_boundary, right.target_start_boundary),
                (left.target_end_boundary, right.target_end_boundary),
            )
            for general_boundary, refined_boundary in constraints:
                existing = boundary_map.get(general_boundary)
                if existing is not None and existing != refined_boundary:
                    valid = False
                    break
                boundary_map[general_boundary] = refined_boundary
            if not valid:
                break
        if valid:
            yield tuple(permutation), boundary_map


def _mapped_pairing_matches(
    general: canonical.NormalizedDecoratedSolution,
    refined: canonical.NormalizedDecoratedSolution,
    cuts: Sequence[int],
    mapping_permutation: Sequence[int],
) -> bool:
    for general_index, refined_index in enumerate(mapping_permutation):
        expanded = _expand_mapping(
            general.mappings[general_index], cuts, general.segment_count
        )
        if expanded is None:
            return False
        if _mapping_signature(expanded) != _mapping_signature(
            refined.mappings[refined_index]
        ):
            return False
    return True


def _match_variants(
    general: canonical.NormalizedDecoratedSolution,
    refined: canonical.NormalizedDecoratedSolution,
    *,
    require_mapping_refinement: bool,
) -> Optional[SubsumptionCertificate]:
    if general.segment_count > refined.segment_count:
        return None
    if require_mapping_refinement:
        if len(general.mappings) != len(refined.mappings):
            return None
        if sorted(mapping.mirror_sign for mapping in general.mappings) != sorted(
            mapping.mirror_sign for mapping in refined.mappings
        ):
            return None

    general_p0 = _pole_boundary(general, "P0")
    refined_p0 = _pole_boundary(refined, "P0")
    if general_p0 != 0 or refined_p0 != 0:
        raise ValueError("Normalized profiles must start at P0")
    general_p1 = _pole_boundary(general, "P1")
    refined_p1 = _pole_boundary(refined, "P1")
    if general_p1 > refined_p1:
        return None
    if general.segment_count - general_p1 > refined.segment_count - refined_p1:
        return None

    initial_angles = _angle_extension(general.points[0], refined.points[0], {})
    if initial_angles is None:
        return None

    correspondence_options: Iterable[Tuple[Tuple[int, ...], Dict[int, int]]]
    if require_mapping_refinement:
        correspondence_options = _mapping_correspondences(general, refined)
    else:
        correspondence_options = (((), {0: 0, general_p1: refined_p1}),)

    for mapping_permutation, required_boundaries in correspondence_options:
        if required_boundaries.get(general_p1, refined_p1) != refined_p1:
            continue
        cuts: List[int] = [0]
        variable_images: Dict[str, DecoratedPath] = {}

        def search(
            general_index: int,
            refined_boundary: int,
            angle_images: Mapping[str, AngleImage],
        ) -> Optional[SubsumptionCertificate]:
            if general_index == general.segment_count:
                if refined_boundary != refined.segment_count:
                    return None
                if require_mapping_refinement and not _mapped_pairing_matches(
                    general, refined, cuts, mapping_permutation
                ):
                    return None
                return SubsumptionCertificate(
                    general_variant_label=general.transform_label,
                    refined_variant_label=refined.transform_label,
                    variable_substitution=dict(variable_images),
                    angle_substitution=dict(angle_images),
                    boundary_map=tuple(cuts),
                    scope=(
                        "configuration_mapping_refinement"
                        if require_mapping_refinement
                        else "contour_shape_family"
                    ),
                    copy_mappings_checked_exactly=require_mapping_refinement,
                )

            literal = general.segments[general_index]
            assigned = variable_images.get(literal.variable)
            assigned_length = None if assigned is None else len(assigned.segments)
            for refined_end in _possible_end_boundaries(
                general_index=general_index,
                current_refined_boundary=refined_boundary,
                general_p1=general_p1,
                refined_p1=refined_p1,
                general_segment_count=general.segment_count,
                refined_segment_count=refined.segment_count,
                assigned_length=assigned_length,
                required_boundaries=required_boundaries,
            ):
                block = _extract_path(refined, refined_boundary, refined_end)
                base_image = block.inverted() if literal.inverse else block
                if assigned is not None and assigned != base_image:
                    continue

                next_boundary = general_index + 1
                refined_point_index = (
                    0 if refined_end == refined.segment_count else refined_end
                )
                general_point_index = (
                    0 if next_boundary == general.segment_count else next_boundary
                )
                extended_angles = _angle_extension(
                    general.points[general_point_index],
                    refined.points[refined_point_index],
                    angle_images,
                )
                if extended_angles is None:
                    continue

                newly_assigned = assigned is None
                if newly_assigned:
                    variable_images[literal.variable] = base_image
                cuts.append(refined_end)
                result = search(general_index + 1, refined_end, extended_angles)
                cuts.pop()
                if newly_assigned:
                    del variable_images[literal.variable]
                if result is not None:
                    return result
            return None

        result = search(0, 0, initial_angles)
        if result is not None:
            return result
    return None


def find_subsumption_normalized(
    general_variants: Sequence[canonical.NormalizedDecoratedSolution],
    refined_canonical: canonical.NormalizedDecoratedSolution,
    *,
    require_mapping_refinement: bool = False,
) -> Optional[SubsumptionCertificate]:
    for general_variant in sorted(general_variants, key=lambda item: item.serialized):
        certificate = _match_variants(
            general_variant,
            refined_canonical,
            require_mapping_refinement=require_mapping_refinement,
        )
        if certificate is not None:
            return certificate
    return None


def find_shape_subsumption(
    general: canonical.DecoratedSolution,
    refined: canonical.DecoratedSolution,
) -> Optional[SubsumptionCertificate]:
    """Return a certificate for inclusion of contour-shape families.

    The refined profile's own contact mapping need not refine the general one:
    once its contour is an instance of the general solved profile, the general
    profile already supplies a valid mapping for the same contour.
    """
    return find_subsumption_normalized(
        canonical.normalized_variants(general),
        canonical.canonical_normalized_solution(refined),
        require_mapping_refinement=False,
    )


def find_configuration_subsumption(
    general: canonical.DecoratedSolution,
    refined: canonical.DecoratedSolution,
) -> Optional[SubsumptionCertificate]:
    """Return a stronger certificate requiring exact mapping refinement."""
    return find_subsumption_normalized(
        canonical.normalized_variants(general),
        canonical.canonical_normalized_solution(refined),
        require_mapping_refinement=True,
    )


def find_subsumption(
    general: canonical.DecoratedSolution,
    refined: canonical.DecoratedSolution,
) -> Optional[SubsumptionCertificate]:
    """Compatibility alias for shape-family subsumption."""
    return find_shape_subsumption(general, refined)


def _generality_key(entry: ProfileReductionEntry) -> Tuple[object, ...]:
    normalized = canonical.canonical_normalized_solution(entry.decorated_solution)
    return (
        normalized.segment_count,
        normalized.fixed_zero_count,
        -normalized.free_angle_count,
        normalized.variable_count,
        entry.canonical_json,
        entry.profile_id,
    )


def reduce_profiles(
    entries: Sequence[ProfileReductionEntry],
) -> Tuple[Tuple[int, ...], Tuple[AbsorptionRecord, ...]]:
    """Return retained profile ids and exact absorption records.

    Inputs are expected to be one representative per decorated canonical class.
    The deterministic generality order makes every accepted parent no more
    detailed than its child, so the resulting relation is acyclic.
    """
    normalized = {
        entry.profile_id: canonical.canonical_normalized_solution(
            entry.decorated_solution
        )
        for entry in entries
    }
    variants = {
        entry.profile_id: canonical.normalized_variants(entry.decorated_solution)
        for entry in entries
    }

    def cached_key(entry: ProfileReductionEntry) -> Tuple[object, ...]:
        value = normalized[entry.profile_id]
        return (
            value.segment_count,
            value.fixed_zero_count,
            -value.free_angle_count,
            value.variable_count,
            entry.canonical_json,
            entry.profile_id,
        )

    ordered = sorted(entries, key=cached_key)
    parent_records: Dict[int, AbsorptionRecord] = {}
    active_generals: List[ProfileReductionEntry] = []

    for refined_entry in ordered:
        refined_value = normalized[refined_entry.profile_id]
        selected: Optional[AbsorptionRecord] = None
        for general_entry in active_generals:
            if not general_entry.all_curve_components_free:
                continue
            general_value = normalized[general_entry.profile_id]
            if general_value.segment_count > refined_value.segment_count:
                continue
            certificate = find_subsumption_normalized(
                variants[general_entry.profile_id],
                refined_value,
                require_mapping_refinement=False,
            )
            if certificate is None:
                continue
            selected = AbsorptionRecord(
                absorbed_profile_id=refined_entry.profile_id,
                representative_profile_id=general_entry.profile_id,
                certificate=certificate,
            )
            break
        if selected is None:
            active_generals.append(refined_entry)
        else:
            parent_records[refined_entry.profile_id] = selected

    retained = tuple(entry.profile_id for entry in active_generals)
    return retained, tuple(
        parent_records[profile_id] for profile_id in sorted(parent_records)
    )

