#!/usr/bin/env python3
"""Detailed JSON export for the local results viewer.

This module intentionally contains no filtering mathematics. It converts one
already-computed terminal profile analysis into a stable, user-facing record.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import analysis_pipeline as pipeline
import angle_constraints as angles
import symbolic_enumerator as base
import profile_formatter


def _literal_variables(text: str) -> Tuple[str, ...]:
    variables: List[str] = []
    seen = set()
    for token in text.split():
        variable = token.removesuffix("^-1")
        if variable != "1" and variable not in seen:
            seen.add(variable)
            variables.append(variable)
    return tuple(variables)


def _point_description(case: base.PlacementCase, marker: str) -> str:
    locus = case.marker_locus_map()[marker]
    if locus in ("P0", "P1"):
        return locus
    boundary = case.marker_boundary_map()[marker]
    return f"inside {locus} (boundary {boundary})"


def mapping_summary(case: base.PlacementCase) -> Dict[str, object]:
    """Return an explicit human-readable description of both copy mappings."""
    case_data = case.to_dict()
    return {
        "A": {
            "central_part": "P0 -- A -- P1",
            "copy_factor": case_data["a_target"],
            "copy_start_point": _point_description(case, "A_start"),
            "copy_start_maps_to": "P0",
            "copy_end_point": _point_description(case, "A_end"),
            "copy_end_maps_to": "P1",
            "direction": case_data["a_direction"],
            "isometry": case_data["a_isometry"],
            "flipped": case.a_mirror_sign == base.REFLECTED,
            "equation": case_data["equations"][0],
            "display": (
                f"A <= {case_data['a_target']}; "
                f"{_point_description(case, 'A_start')} -> P0, "
                f"{_point_description(case, 'A_end')} -> P1; "
                f"{case_data['a_direction']}, {case_data['a_isometry']}"
            ),
        },
        "B": {
            "central_part": "P1 -- B -- P0",
            "copy_factor": case_data["b_target"],
            "copy_start_point": _point_description(case, "B_start"),
            "copy_start_maps_to": "P1",
            "copy_end_point": _point_description(case, "B_end"),
            "copy_end_maps_to": "P0",
            "direction": case_data["b_direction"],
            "isometry": case_data["b_isometry"],
            "flipped": case.b_mirror_sign == base.REFLECTED,
            "equation": case_data["equations"][1],
            "display": (
                f"B <= {case_data['b_target']}; "
                f"{_point_description(case, 'B_start')} -> P1, "
                f"{_point_description(case, 'B_end')} -> P0; "
                f"{case_data['b_direction']}, {case_data['b_isometry']}"
            ),
        },
    }


def rejection_stage(analysis: pipeline.ProfileAnalysis) -> str:
    turn_ok = analysis.total_turn.feasible
    poles_ok = analysis.pole_angles.feasible
    translation_ok = analysis.translation_pass
    if turn_ok and poles_ok and translation_ok:
        return "retained"
    if not turn_ok and not poles_ok:
        return "total_turn_and_poles"
    if not turn_ok:
        return "total_turn"
    if not poles_ok:
        return "pole_angles"
    return "translation"


def rejection_reasons(analysis: pipeline.ProfileAnalysis) -> List[str]:
    reasons: List[str] = []
    if not analysis.total_turn.feasible:
        reasons.append(analysis.total_turn.discard_reason or "Total-turn constraint failed")
    if not analysis.pole_angles.feasible:
        reasons.append(analysis.pole_angles.discard_reason or "Pole-angle constraint failed")
    if not analysis.translation_pass:
        reasons.append(
            analysis.se2_holonomy.translation.discard_reason
            or "Translation-holonomy constraint failed"
        )
    return reasons


def detailed_profile_record(
    profile_id: int,
    case: base.PlacementCase,
    state: base.SolverState,
    derivation: Sequence[str],
    analysis: pipeline.ProfileAnalysis,
    experimental: Dict[str, object] | None = None,
) -> Dict[str, object]:
    a_text, b_text = angles.state_profile_text(case, state)
    formal_profile = profile_formatter.build_formal_profile(
        case, state, analysis.angle_solution
    )
    word_contour = formal_profile.word_contour
    curve_variables = _literal_variables(f"{a_text} {b_text}")
    all_parameters = curve_variables + formal_profile.free_angle_parameters
    stage = rejection_stage(analysis)
    retained = stage == "retained"
    reflection_count = int(case.a_mirror_sign == base.REFLECTED) + int(
        case.b_mirror_sign == base.REFLECTED
    )

    return {
        "profile_id": profile_id,
        "case_id": case.case_id,
        "placement": case.to_dict(),
        "mapping": mapping_summary(case),
        "solution": {
            "profile": formal_profile.text,
            "formal_profile": formal_profile.to_dict(),
            "word_contour": word_contour,
            "contour": formal_profile.text,
            "derivation": list(derivation),
            "solver_depth": state.depth,
            "parameters": list(all_parameters),
            "curve_parameters": list(curve_variables),
            "angle_parameters": list(formal_profile.free_angle_parameters),
            "fixed_zero_angle_classes": list(
                formal_profile.fixed_zero_angle_classes
            ),
            "curve_parameter_count": len(curve_variables),
            "angle_parameter_count": formal_profile.angle_parameter_count,
            "parameter_count": len(all_parameters),
            "word_token_count": len(a_text.split()) + len(b_text.split()),
        },
        "status": {
            "retained": retained,
            "stage": stage,
            "reasons": rejection_reasons(analysis),
        },
        "filters": {
            "point_angles": analysis.angle_solution.to_dict(),
            "total_turn": analysis.total_turn.to_dict(),
            "pole_angles": analysis.pole_angles.to_dict(),
            "translation_holonomy": analysis.se2_holonomy.translation.to_dict(),
            "angular_pass": analysis.angular_pass,
            "translation_pass": analysis.translation_pass,
            "passes_all": analysis.passes_all,
        },
        "experimental": experimental or {},
        "sort": {
            "retained_priority": 0 if retained else 1,
            "stage_priority": {
                "retained": 0,
                "translation": 1,
                "pole_angles": 2,
                "total_turn": 3,
                "total_turn_and_poles": 4,
            }[stage],
            "reflection_count": reflection_count,
            "word_token_count": len(a_text.split()) + len(b_text.split()),
            "parameter_count": len(all_parameters),
        },
    }
