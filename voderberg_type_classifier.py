#!/usr/bin/env python3
"""Formal contact-topology classifier for Voderberg types 1 and 2.

The classifier uses only a placement case: the two contact arcs, the two target
factors, their traversal directions, and the direct/reflected parity fixed by
opposite-side contact.  It therefore classifies *formal compatibility* with
Voderberg's two coarse schemes; it is not a geometric realizability test.

Type 1 compatibility
--------------------
One contact (the principal contact) is mapped directly onto the same undirected
arc with reversed traversal.  The other contact is mapped directly onto a
nonempty proper subarc of the principal contact.  An orientation-preserving
isometry that swaps the two distinct poles is necessarily a half-turn, once a
geometric realization exists.

Type 2 compatibility
--------------------
The principal contact is mapped by an orientation-reversing isometry onto an
arc whose interior contains exactly one pole.  The other contact is mapped
directly onto a nonempty proper subarc of the principal contact.  The formal
model cannot by itself distinguish a genuine glide reflection from every
possible reflected geometric degeneration, hence the deliberately cautious
"compatible" wording.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import symbolic_enumerator as base


SCHEMA_VERSION = "voderberg-formal-type-v1"
TYPE_1 = "type1"
TYPE_2 = "type2"
KNOWN_TYPES = (TYPE_1, TYPE_2)
SELECTION_ALL = "all"
SELECTION_BOTH = "type1+type2"
SELECTION_CHOICES = (SELECTION_ALL, TYPE_1, TYPE_2, SELECTION_BOTH)


@dataclass(frozen=True)
class DirectedSegmentRef:
    position: int
    orientation: int


@dataclass(frozen=True)
class ContactTopology:
    name: str
    source_start_boundary: int
    source_end_boundary: int
    target_start_boundary: int
    target_end_boundary: int
    target_direction: int
    mirror_sign: int
    source: Tuple[DirectedSegmentRef, ...]
    target: Tuple[DirectedSegmentRef, ...]
    target_interior_poles: Tuple[str, ...]

    @property
    def is_direct(self) -> bool:
        return self.mirror_sign == base.DIRECT

    @property
    def is_reflected(self) -> bool:
        return self.mirror_sign == base.REFLECTED


@dataclass(frozen=True)
class TypeWitness:
    type_name: str
    principal_contact: str
    secondary_contact: str
    principal_relation: str
    secondary_relation: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": self.type_name,
            "principal_contact": self.principal_contact,
            "secondary_contact": self.secondary_contact,
            "principal_relation": self.principal_relation,
            "secondary_relation": self.secondary_relation,
        }


@dataclass(frozen=True)
class VoderbergTypeClassification:
    compatible_types: Tuple[str, ...]
    witnesses: Tuple[TypeWitness, ...]

    def is_compatible_with(self, type_name: str) -> bool:
        return type_name in self.compatible_types

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "classification_level": "formal_contact_topology",
            "compatible_types": list(self.compatible_types),
            "type1_compatible": TYPE_1 in self.compatible_types,
            "type2_compatible": TYPE_2 in self.compatible_types,
            "witnesses": [witness.to_dict() for witness in self.witnesses],
            "geometric_realizability_proved": False,
            "interpretation": (
                "Compatibility with Voderberg's coarse contact/isometry scheme; "
                "not a proof that a simple non-overlapping three-copy realization exists."
            ),
        }


def normalize_selection(value: str) -> str:
    """Normalize CLI spelling while retaining four distinct selection modes."""
    compact = value.strip().lower().replace(" ", "")
    aliases = {
        "all": SELECTION_ALL,
        "type1": TYPE_1,
        "1": TYPE_1,
        "type2": TYPE_2,
        "2": TYPE_2,
        "type1+type2": SELECTION_BOTH,
        "type2+type1": SELECTION_BOTH,
        "type1,type2": SELECTION_BOTH,
        "type2,type1": SELECTION_BOTH,
        "both": SELECTION_BOTH,
        "voderberg": SELECTION_BOTH,
    }
    try:
        return aliases[compact]
    except KeyError as exc:
        choices = ", ".join(SELECTION_CHOICES)
        raise ValueError(
            f"Unknown Voderberg type selection {value!r}; choose one of: {choices}"
        ) from exc


def selection_types(selection: str) -> Tuple[str, ...]:
    normalized = normalize_selection(selection)
    if normalized == SELECTION_ALL:
        return ()
    if normalized == SELECTION_BOTH:
        return KNOWN_TYPES
    return (normalized,)


def matches_selection(
    classification: VoderbergTypeClassification,
    selection: str,
) -> bool:
    requested = selection_types(selection)
    if not requested:
        return True
    return any(type_name in classification.compatible_types for type_name in requested)


def record_matches_selection(record: Mapping[str, object], selection: str) -> bool:
    """Apply a selection to an exported profile record.

    A non-``all`` selection requires a recent audit export carrying the
    ``voderberg_type`` annotation.  Silently guessing from display text would
    make the geometry filter fragile, so missing annotations are reported.
    """
    requested = selection_types(selection)
    if not requested:
        return True
    payload = record.get("voderberg_type")
    if not isinstance(payload, Mapping):
        raise ValueError(
            "The input profile has no voderberg_type annotation. Rerun the audit "
            "with the type-classifier patch before selecting type1 or type2."
        )
    compatible = payload.get("compatible_types", ())
    if not isinstance(compatible, Sequence) or isinstance(compatible, (str, bytes)):
        raise ValueError("Invalid voderberg_type.compatible_types field")
    labels = {str(item) for item in compatible}
    return any(type_name in labels for type_name in requested)


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


def _interior_boundaries(
    start: int,
    end: int,
    direction: int,
    segment_count: int,
) -> Tuple[int, ...]:
    current = start
    output: List[int] = []
    if direction == base.FORWARD:
        while True:
            current = (current + 1) % segment_count
            if current == end:
                break
            output.append(current)
    else:
        while True:
            current = (current - 1) % segment_count
            if current == end:
                break
            output.append(current)
    return tuple(output)


def _build_contacts(case: base.PlacementCase) -> Tuple[ContactTopology, ContactTopology]:
    segment_count = len(case.cycle_word)
    p0 = 0
    p1 = len(case.a_word)
    marker_boundaries = case.marker_boundary_map()
    poles = {p0: "P0", p1: "P1"}

    def contact(
        name: str,
        source_start: int,
        source_end: int,
        start_marker: str,
        end_marker: str,
        target_direction: int,
        mirror_sign: int,
    ) -> ContactTopology:
        target_start = marker_boundaries[start_marker]
        target_end = marker_boundaries[end_marker]
        interior = _interior_boundaries(
            target_start, target_end, target_direction, segment_count
        )
        return ContactTopology(
            name=name,
            source_start_boundary=source_start,
            source_end_boundary=source_end,
            target_start_boundary=target_start,
            target_end_boundary=target_end,
            target_direction=target_direction,
            mirror_sign=mirror_sign,
            source=_directed_factor_refs(
                source_start, source_end, base.FORWARD, segment_count
            ),
            target=_directed_factor_refs(
                target_start, target_end, target_direction, segment_count
            ),
            target_interior_poles=tuple(
                poles[boundary] for boundary in interior if boundary in poles
            ),
        )

    return (
        contact(
            "A",
            p0,
            p1,
            "A_start",
            "A_end",
            case.a_direction,
            case.a_mirror_sign,
        ),
        contact(
            "B",
            p1,
            p0,
            "B_start",
            "B_end",
            case.b_direction,
            case.b_mirror_sign,
        ),
    )


def _positions(refs: Iterable[DirectedSegmentRef]) -> Tuple[int, ...]:
    return tuple(reference.position for reference in refs)


def _is_reverse_of_same_arc(
    source: Sequence[DirectedSegmentRef],
    target: Sequence[DirectedSegmentRef],
) -> bool:
    expected = tuple(
        DirectedSegmentRef(reference.position, -reference.orientation)
        for reference in reversed(source)
    )
    return tuple(target) == expected


def _is_nonempty_proper_subarc(
    whole: Sequence[DirectedSegmentRef],
    candidate: Sequence[DirectedSegmentRef],
) -> bool:
    whole_positions = _positions(whole)
    candidate_positions = _positions(candidate)
    if not candidate_positions or len(candidate_positions) >= len(whole_positions):
        return False
    width = len(candidate_positions)
    for start in range(len(whole_positions) - width + 1):
        block = whole_positions[start : start + width]
        if candidate_positions == block or candidate_positions == tuple(reversed(block)):
            return True
    return False


def classify_placement(case: base.PlacementCase) -> VoderbergTypeClassification:
    contacts = _build_contacts(case)
    witnesses: List[TypeWitness] = []

    for principal_index, principal in enumerate(contacts):
        secondary = contacts[1 - principal_index]
        secondary_is_rotational_subarc = bool(
            secondary.is_direct
            and _is_nonempty_proper_subarc(principal.source, secondary.target)
        )
        if not secondary_is_rotational_subarc:
            continue

        if principal.is_direct and _is_reverse_of_same_arc(
            principal.source, principal.target
        ):
            witnesses.append(
                TypeWitness(
                    type_name=TYPE_1,
                    principal_contact=principal.name,
                    secondary_contact=secondary.name,
                    principal_relation="direct self-contact with reversed traversal (half-turn compatible)",
                    secondary_relation="direct image of a nonempty proper subarc of the principal contact",
                )
            )

        if principal.is_reflected and len(principal.target_interior_poles) == 1:
            witnesses.append(
                TypeWitness(
                    type_name=TYPE_2,
                    principal_contact=principal.name,
                    secondary_contact=secondary.name,
                    principal_relation=(
                        "reflected target arc contains exactly one pole in its interior "
                        "(glide-reflection compatible)"
                    ),
                    secondary_relation="direct image of a nonempty proper subarc of the principal contact",
                )
            )

    compatible_types = tuple(
        type_name for type_name in KNOWN_TYPES
        if any(witness.type_name == type_name for witness in witnesses)
    )
    return VoderbergTypeClassification(
        compatible_types=compatible_types,
        witnesses=tuple(witnesses),
    )


def summarize_classifications(
    classifications: Sequence[Tuple[int, int, VoderbergTypeClassification]],
    *,
    selection: str,
) -> Dict[str, object]:
    """Summarize ``(profile_id, case_id, classification)`` triples."""
    normalized_selection = normalize_selection(selection)
    type1_profiles = {
        profile_id for profile_id, _case_id, item in classifications
        if item.is_compatible_with(TYPE_1)
    }
    type2_profiles = {
        profile_id for profile_id, _case_id, item in classifications
        if item.is_compatible_with(TYPE_2)
    }
    type1_cases = {
        case_id for _profile_id, case_id, item in classifications
        if item.is_compatible_with(TYPE_1)
    }
    type2_cases = {
        case_id for _profile_id, case_id, item in classifications
        if item.is_compatible_with(TYPE_2)
    }
    selected_profiles = {
        profile_id for profile_id, _case_id, item in classifications
        if matches_selection(item, normalized_selection)
    }
    all_profiles = {profile_id for profile_id, _case_id, _item in classifications}
    return {
        "schema_version": SCHEMA_VERSION,
        "classification_level": "formal_contact_topology",
        "selection": normalized_selection,
        "terminal_profile_count_before_type_selection": len(all_profiles),
        "terminal_profile_count_selected_for_downstream_pipeline": len(selected_profiles),
        "terminal_profile_count_excluded_by_type_selection": len(all_profiles - selected_profiles),
        "type1_compatible_profile_count": len(type1_profiles),
        "type2_compatible_profile_count": len(type2_profiles),
        "compatible_with_both_profile_count": len(type1_profiles & type2_profiles),
        "compatible_with_neither_profile_count": len(
            all_profiles - (type1_profiles | type2_profiles)
        ),
        "type1_compatible_placement_case_count_with_terminal_profiles": len(type1_cases),
        "type2_compatible_placement_case_count_with_terminal_profiles": len(type2_cases),
        "filtering_occurs_after_formal_terminal_profiles": True,
        "geometric_realizability_proved": False,
    }
