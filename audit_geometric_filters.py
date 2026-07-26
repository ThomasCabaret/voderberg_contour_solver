#!/usr/bin/env python3
"""Sequential bounded audit of formal Voderberg contour profiles.

Every independent filter has its own pass over the profile collection.  Fast
formal and local filters are evaluated before the geometric cascade.  Expensive
shared-boundary and Z3 stages receive only primitive canonical profiles that
survived every preceding filter; symmetry-equivalent and subsumed profiles keep
an audit link to their representative instead of being recomputed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import analysis_pipeline as pipeline
import angle_constraints as angles
import experimental_pipeline
import external_boundary_constraints as external
import exact_partial_word_solver as exact_word_solver
import family_representative_expansion as family_expansion
import curve_term_solver
import forced_point_coincidence as forced_points
import formal_equation_audit as formal_audit
import joint_translation_z3 as z3_backend
import global_linear_contour_filter as global_linear
import placed_copy_geometry as placed_geometry
import method_status
import positive_length_filter
import profile_formatter
import results_export
import solution_canonicalization
import profile_subsumption
import voderberg_type_classifier as voderberg_types
import settings
import symbolic_enumerator as base


MIRROR_VARIANTS = (
    (angles.SAME, angles.SAME),
    (angles.SAME, angles.OPPOSITE),
    (angles.OPPOSITE, angles.SAME),
    (angles.OPPOSITE, angles.OPPOSITE),
)


class AuditProgress:
    def __init__(self, enabled: bool, interval: int) -> None:
        self.enabled = enabled
        self.interval = max(1, interval)

    def stage(self, index: int, total: int, message: str) -> None:
        if self.enabled:
            print(f"[{index}/{total}] {message}", flush=True)

    def update(self, current: int, total: int, message: str) -> None:
        if self.enabled and (current % self.interval == 0 or current == total):
            print(f"      {message}", flush=True)

    def done(self, message: str) -> None:
        if self.enabled:
            print(f"      -> {message}", flush=True)


@dataclass(frozen=True)
class TerminalProfile:
    profile_id: int
    case: base.PlacementCase
    state: base.SolverState
    derivation: Tuple[str, ...]
    a_text: str
    b_text: str
    exact_family: Optional[exact_word_solver.ExactFormalFamily] = None
    family_exponent_assignment: Tuple[Tuple[str, int], ...] = ()
    formal_solver_status: str = "legacy_bounded"


@dataclass
class ProfileWork:
    terminal: TerminalProfile
    angle_solution: Optional[angles.AngleSolution] = None
    angle_error: Optional[str] = None
    total_turn: object | None = None
    pole_angles: object | None = None
    holonomy: object | None = None
    external_system: Optional[external.JointBoundarySystem] = None
    external_error: Optional[str] = None
    global_linear_analysis: Optional[global_linear.GlobalLinearContourAnalysis] = None
    global_linear_error: Optional[str] = None
    inner_points: Optional[forced_points.ForcedPointCoincidenceAnalysis] = None
    outer_points: Optional[forced_points.ForcedPointCoincidenceAnalysis] = None
    forced_point_error: Optional[str] = None
    placed_copy_analysis: Optional[placed_geometry.PlacedCopyGeometryAnalysis] = None
    placed_copy_error: Optional[str] = None
    z3_problem: Optional[z3_backend.Z3Problem] = None
    z3_result: Optional[z3_backend.Z3Result] = None
    canonical_solution: Optional[solution_canonicalization.CanonicalSolution] = None
    canonicalization_error: Optional[str] = None
    voderberg_type: Optional[voderberg_types.VoderbergTypeClassification] = None
    terminal_mapping: Optional[Dict[str, object]] = None
    terminal_mapping_error: Optional[str] = None
    curve_term_solution: Optional[curve_term_solver.CurveTermSolution] = None
    curve_term_error: Optional[str] = None
    decorated_solution: Optional[solution_canonicalization.DecoratedSolution] = None
    canonical_representative_profile_id: Optional[int] = None
    canonical_class_size: int = 1
    subsumption_record: Optional[profile_subsumption.AbsorptionRecord] = None

    @property
    def formal_reduction_pass(self) -> bool:
        profile_id = self.terminal.profile_id
        canonical_ok = (
            self.canonical_representative_profile_id is None
            or self.canonical_representative_profile_id == profile_id
        )
        return canonical_ok and self.subsumption_record is None

    @property
    def core_pass(self) -> bool:
        return bool(
            self.angle_solution is not None
            and self.total_turn is not None
            and self.total_turn.feasible
            and self.pole_angles is not None
            and self.pole_angles.feasible
            and self.holonomy is not None
            and not self.holonomy.translation.exact_obstruction
        )

    @property
    def joint_rotation_pass(self) -> bool:
        return bool(
            self.external_system is not None
            and self.external_system.rotation_analysis.feasible
        )

    @property
    def global_linear_pass(self) -> bool:
        return bool(
            self.global_linear_analysis is not None
            and self.global_linear_analysis.feasible
        )

    @property
    def joint_translation_pass(self) -> bool:
        return bool(
            self.external_system is not None
            and not self.external_system.translation_analysis.exact_obstruction
        )

    @property
    def forced_point_pass(self) -> bool:
        return bool(
            self.inner_points is not None
            and self.outer_points is not None
            and self.inner_points.passes_filter
            and self.outer_points.passes_filter
        )

    @property
    def placed_copy_pass(self) -> bool:
        return bool(
            self.placed_copy_analysis is not None
            and self.placed_copy_analysis.passes_filter
        )

    @property
    def pre_z3_pass(self) -> bool:
        return bool(
            self.core_pass
            and self.external_error is None
            and self.global_linear_error is None
            and self.global_linear_pass
            and self.joint_translation_pass
            and self.forced_point_error is None
            and self.forced_point_pass
            and self.placed_copy_error is None
            and self.placed_copy_pass
        )

    @property
    def final_pass(self) -> bool:
        if not self.formal_reduction_pass:
            return False
        if not self.pre_z3_pass:
            return False
        return not bool(self.z3_result and self.z3_result.exact_unsat)

    def final_stage(self) -> str:
        if (
            self.canonical_representative_profile_id is not None
            and self.canonical_representative_profile_id != self.terminal.profile_id
        ):
            return "canonical_equivalent"
        if self.subsumption_record is not None:
            return "profile_subsumed"
        if self.angle_error:
            return "point_angle_error"
        if self.total_turn is not None and not self.total_turn.feasible:
            return "total_turn"
        if self.pole_angles is not None and not self.pole_angles.feasible:
            return "pole_angles"
        if self.holonomy is not None and self.holonomy.translation.exact_obstruction:
            return "core_translation"
        if self.external_error:
            return "external_boundary_error"
        if self.global_linear_error:
            return "global_linear_contour_error"
        if self.global_linear_analysis is not None and not self.global_linear_analysis.feasible:
            return "global_linear_contours"
        if self.external_system is not None and self.external_system.translation_analysis.exact_obstruction:
            return "joint_translation"
        if self.forced_point_error:
            return "forced_point_error"
        if not self.forced_point_pass:
            return "forced_point_coincidence"
        if self.placed_copy_error:
            return "placed_copy_error"
        if self.placed_copy_analysis is not None and not self.placed_copy_analysis.passes_filter:
            return "placed_copy_geometry"
        if self.z3_result is not None and self.z3_result.exact_unsat:
            return "z3_unsat"
        return "retained"

    def final_reason(self) -> Optional[str]:
        stage = self.final_stage()
        if stage == "canonical_equivalent":
            return (
                "Equivalent under pole exchange, contour reversal/global mirror, "
                "copy permutation, and signed renaming; represented by profile "
                f"{self.canonical_representative_profile_id}."
            )
        if stage == "profile_subsumed":
            assert self.subsumption_record is not None
            return (
                "Contour-shape family is an instance of profile "
                f"{self.subsumption_record.representative_profile_id}."
            )
        if stage == "point_angle_error":
            return self.angle_error
        if stage == "total_turn":
            return self.total_turn.discard_reason
        if stage == "pole_angles":
            return self.pole_angles.discard_reason
        if stage == "core_translation":
            return self.holonomy.translation.discard_reason
        if stage == "external_boundary_error":
            return self.external_error
        if stage == "global_linear_contour_error":
            return self.global_linear_error
        if stage == "global_linear_contours":
            return self.global_linear_analysis.discard_reason
        if stage == "joint_translation":
            return self.external_system.translation_analysis.reason
        if stage == "forced_point_error":
            return self.forced_point_error
        if stage == "forced_point_coincidence":
            reasons = []
            if self.inner_points is not None and not self.inner_points.passes_filter:
                reasons.append("Inner boundary: " + (self.inner_points.discard_reason or "forced coincidence"))
            if self.outer_points is not None and not self.outer_points.passes_filter:
                reasons.append("Outer boundary: " + (self.outer_points.discard_reason or "forced coincidence"))
            return "; ".join(reasons) or "Forced point coincidence"
        if stage == "placed_copy_error":
            return self.placed_copy_error
        if stage == "placed_copy_geometry":
            return self.placed_copy_analysis.discard_reason
        if stage == "z3_unsat":
            return "Z3/NLSAT proved the polynomial relaxation unsatisfiable."
        return None


def _profile_text(case: base.PlacementCase, state: base.SolverState) -> Tuple[str, str]:
    return angles.state_profile_text(case, state)


def _variant_passes(case: base.PlacementCase, state: base.SolverState, signs: Tuple[int, int]) -> Tuple[bool, bool]:
    result = pipeline.analyze_terminal_profile(
        case,
        state,
        mirror_sign_a=signs[0],
        mirror_sign_b=signs[1],
    )
    return result.angular_pass, result.passes_all


def _experimental_record(work: ProfileWork) -> Dict[str, object]:
    if work.external_system is None:
        if not work.formal_reduction_pass:
            status = "not_run_after_formal_profile_reduction"
            reason = work.final_reason()
        elif not work.core_pass:
            status = "not_run_after_core_rejection"
            reason = work.final_reason()
        else:
            status = "external_boundary_error"
            reason = work.external_error
        return {
            "status": status,
            "reason": reason,
            "affects_core_status": False,
            "exact_encoded_model_rejection": False,
            "external_boundary": None,
            "z3_problem": None,
            "z3_result": None,
        }

    if work.global_linear_error:
        status = "global_linear_contour_filter_error"
        reason = work.global_linear_error
    elif work.global_linear_analysis is not None and not work.global_linear_analysis.feasible:
        status = "exact_global_linear_contour_reject"
        reason = work.global_linear_analysis.discard_reason
    elif work.external_system.translation_analysis.exact_obstruction:
        status = "exact_joint_translation_reject"
        reason = work.external_system.translation_analysis.reason
    elif work.forced_point_error:
        status = "forced_point_filter_error"
        reason = work.forced_point_error
    elif not work.forced_point_pass:
        status = "exact_forced_point_reject"
        reason = work.final_reason()
    elif work.placed_copy_error:
        status = "placed_copy_filter_error"
        reason = work.placed_copy_error
    elif work.placed_copy_analysis is not None and not work.placed_copy_analysis.passes_filter:
        status = "exact_placed_copy_reject"
        reason = work.placed_copy_analysis.discard_reason
    elif work.z3_result is not None:
        status = work.z3_result.status
        reason = work.z3_result.reason
    elif work.z3_problem is not None:
        status = "z3_problem_ready"
        reason = None
    else:
        status = "shared_system_built"
        reason = None

    analysis = experimental_pipeline.ExperimentalProfileAnalysis(
        status=status,
        reason=reason,
        external_system=work.external_system,
        inner_point_coincidence=work.inner_points,
        outer_point_coincidence=work.outer_points,
        global_linear_analysis=work.global_linear_analysis,
        placed_copy_analysis=work.placed_copy_analysis,
        z3_problem=work.z3_problem,
        z3_result=work.z3_result,
    )
    return analysis.to_dict()


def _pipeline_flags(work: ProfileWork) -> Dict[str, object]:
    return {
        "formal_reduction_pass": work.formal_reduction_pass,
        "canonical_representative_profile_id": work.canonical_representative_profile_id,
        "subsumed_by_profile_id": (
            None if work.subsumption_record is None
            else work.subsumption_record.representative_profile_id
        ),
        "point_angles_resolved": work.angle_solution is not None,
        "total_turn_pass": bool(work.total_turn and work.total_turn.feasible),
        "pole_angles_pass": bool(work.pole_angles and work.pole_angles.feasible),
        "core_translation_pass": bool(
            work.holonomy and not work.holonomy.translation.exact_obstruction
        ),
        "core_pass": work.core_pass,
        "external_boundary_built": work.external_system is not None,
        "joint_rotation_diagnostic_pass": work.joint_rotation_pass,
        "global_linear_contour_pass": work.global_linear_pass,
        "joint_translation_pass": work.joint_translation_pass,
        "forced_point_pass": work.forced_point_pass,
        "placed_copy_pass": work.placed_copy_pass,
        "z3_status": None if work.z3_result is None else work.z3_result.status,
        "final_retained": work.final_pass,
        "final_stage": work.final_stage(),
        "final_reason": work.final_reason(),
    }


def audit(
    max_depth: Optional[int],
    max_states: Optional[int],
    *,
    formal_solver_mode: str = settings.DEFAULT_FORMAL_SOLVER_MODE,
    exact_graph_max_nodes: Optional[int] = settings.DEFAULT_EXACT_GRAPH_MAX_NODES,
    exact_graph_max_edges: Optional[int] = settings.DEFAULT_EXACT_GRAPH_MAX_EDGES,
    exact_max_families: int = settings.DEFAULT_EXACT_MAX_FAMILIES_PER_CASE,
    family_expansion_policy: str = settings.DEFAULT_FAMILY_EXPANSION_POLICY,
    family_expansion_max_exponent: int = settings.DEFAULT_FAMILY_EXPANSION_MAX_EXPONENT,
    family_expansion_max_specializations: Optional[int] = (
        settings.DEFAULT_FAMILY_EXPANSION_MAX_SPECIALIZATIONS
    ),
    representative_exponent_value: int = settings.DEFAULT_FAMILY_REPRESENTATIVE_EXPONENT,
    expand_parametric_representatives: Optional[bool] = None,
    specialize_curve_terms: bool = settings.DEFAULT_ENABLE_CURVE_TERM_SPECIALIZATION,
    max_cycle_unrolls: Optional[int] = settings.DEFAULT_FORMAL_MAX_CYCLE_UNROLLS,
    apply_positive_length_filter: bool = settings.DEFAULT_ENABLE_POSITIVE_LENGTH_FILTER,
    canonicalize_solutions: bool = settings.DEFAULT_ENABLE_SOLUTION_CANONICALIZATION,
    reduce_equivalent_profiles: bool = settings.DEFAULT_ENABLE_CANONICAL_PROFILE_REDUCTION,
    reduce_subsumed_profiles: bool = settings.DEFAULT_ENABLE_PROFILE_SUBSUMPTION_REDUCTION,
    enable_global_linear_angle_filter: bool = settings.DEFAULT_ENABLE_GLOBAL_LINEAR_ANGLE_FILTER,
    enable_global_linear_length_filter: bool = settings.DEFAULT_ENABLE_GLOBAL_LINEAR_LENGTH_FILTER,
    enable_chord_length_layer: bool = settings.DEFAULT_ENABLE_CHORD_LENGTH_LAYER,
    enable_signed_area_layer: bool = settings.DEFAULT_ENABLE_SIGNED_AREA_LAYER,
    voderberg_type_selection: str = settings.DEFAULT_VODERBERG_TYPE_SELECTION,
    collect_profiles: bool = False,
    collect_survivors: bool = False,
    run_z3: bool = settings.DEFAULT_RUN_Z3,
    z3_max_profiles: int = settings.DEFAULT_Z3_MAX_PROFILES,
    z3_timeout_ms: int = settings.Z3_DEFAULT_TIMEOUT_MS,
    show_progress: bool = settings.DEFAULT_SHOW_AUDIT_PROGRESS,
    progress_interval: int = settings.DEFAULT_AUDIT_PROGRESS_INTERVAL,
    parity_diagnostics: bool = settings.DEFAULT_RUN_PARITY_DIAGNOSTICS,
) -> Dict[str, object]:
    if expand_parametric_representatives is not None:
        # Compatibility with the v1 API.  New callers should select a named
        # policy instead of toggling implicit exponent-one representatives.
        family_expansion_policy = (
            family_expansion.POLICY_FIXED
            if expand_parametric_representatives
            else family_expansion.POLICY_NONE
        )
    expansion_policy = family_expansion.ExpansionPolicy(
        kind=family_expansion_policy,
        fixed_exponent=representative_exponent_value,
        maximum_exponent=family_expansion_max_exponent,
        max_specializations=family_expansion_max_specializations,
    )
    progress = AuditProgress(show_progress, progress_interval)
    normalized_type_selection = voderberg_types.normalize_selection(voderberg_type_selection)
    stage_total = (18 if parity_diagnostics else 17) + int(canonicalize_solutions) + int(canonicalize_solutions and reduce_subsumed_profiles)
    stage = 0

    stage += 1
    progress.stage(stage, stage_total, "Generating placement cases and fixing contact parity")
    cases = list(base.enumerate_placement_cases())
    direct_only_cases = sum(1 for _ in base.enumerate_placement_cases(allow_reflections=False))
    progress.done(f"{len(cases)} placements; {direct_only_cases} use direct copies only")

    stage += 1
    progress.stage(stage, stage_total, "Auditing the structure of generated formal word systems")
    case_audit_records: Dict[int, Dict[str, object]] = {}
    case_type_classifications: Dict[int, voderberg_types.VoderbergTypeClassification] = {}
    for case_index, case in enumerate(cases, start=1):
        type_classification = voderberg_types.classify_placement(case)
        length_analysis = positive_length_filter.analyze_case(case)
        case_type_classifications[case.case_id] = type_classification
        case_audit_records[case.case_id] = {
            "case_id": case.case_id,
            "placement": case.to_dict(),
            "structure": formal_audit.analyze_case_structure(case),
            "positive_length_filter": {
                **length_analysis.to_dict(),
                "enabled_for_rejection": apply_positive_length_filter,
            },
            "voderberg_type": type_classification.to_dict(),
        }
        progress.update(
            case_index,
            len(cases),
            f"systems characterized: {case_index}/{len(cases)}",
        )
    quadratic_count = sum(
        1 for record in case_audit_records.values()
        if record["structure"]["is_quadratic_system"]
    )
    progress.done(
        f"{quadratic_count} quadratic systems; "
        f"{len(cases) - quadratic_count} nonquadratic systems"
    )

    stage += 1
    progress.stage(
        stage,
        stage_total,
        "Checking exact positive word-length feasibility before branching",
    )
    length_infeasible_case_ids = {
        case.case_id
        for case in cases
        if not case_audit_records[case.case_id]["positive_length_filter"]["feasible"]
    }
    initially_inconsistent_case_ids = {
        case.case_id
        for case in cases
        if base.initial_solver_state(case) is None
    }
    additional_length_rejections = (
        length_infeasible_case_ids - initially_inconsistent_case_ids
    )
    if apply_positive_length_filter:
        progress.done(
            f"{len(length_infeasible_case_ids)} systems reject strict positive word lengths: "
            f"{len(initially_inconsistent_case_ids)} already fail initial simplification; "
            f"{len(additional_length_rejections)} additional systems will be skipped"
        )
        progress.done(
            f"{len(cases) - len(length_infeasible_case_ids)} systems remain for the branching solver"
        )
    else:
        progress.done(
            f"{len(length_infeasible_case_ids)} positive-length contradictions detected, "
            "but rejection is disabled for this run"
        )

    stage += 1
    works: List[ProfileWork] = []
    cases_with_terminals = set()
    exact_case_results: List[exact_word_solver.ExactCaseResult] = []

    if formal_solver_mode == "exact-partial":
        progress.stage(
            stage,
            stage_total,
            "Building exact residual graphs and compiling formal families",
        )
        for case_index, case in enumerate(cases, start=1):
            length_feasible = bool(
                case_audit_records[case.case_id]["positive_length_filter"]["feasible"]
            )
            if apply_positive_length_filter and not length_feasible:
                exact_result = exact_word_solver.ExactCaseResult(
                    case_id=case.case_id,
                    status=exact_word_solver.EXACT_UNSAT,
                    graph_complete=True,
                    graph_summary={
                        "complete": True,
                        "skipped_by_positive_length_filter": True,
                        "reason": "strictly positive word lengths are inconsistent",
                    },
                    families=(),
                    unsupported_families=(),
                    unsupported_reasons=(),
                )
            else:
                exact_result = exact_word_solver.solve_case(
                    case,
                    max_nodes=exact_graph_max_nodes,
                    max_edges=exact_graph_max_edges,
                    max_families=exact_max_families,
                )
            exact_case_results.append(exact_result)
            case_audit_records[case.case_id]["exact_formal_solver"] = exact_result.to_dict()
            case_audit_records[case.case_id]["bounded_search"] = {
                "status": "not_run_exact_partial_mode",
                "search_outcome_class": exact_result.status,
                "terminal_count": len(exact_result.families),
                "search_truncated": not exact_result.graph_complete,
                "state_limit_hit": exact_result.status == exact_word_solver.UNRESOLVED_GRAPH_LIMIT,
                "depth_frontier_cut_count": 0,
                "cycle_unroll_cap_hit": False,
                "cycle_unroll_pruned_state_count": 0,
                "search_exhausted_within_current_graph": exact_result.graph_complete,
                "initial_inconsistent": case.case_id in initially_inconsistent_case_ids,
                "skipped_by_positive_length_filter": (
                    apply_positive_length_filter and not length_feasible
                ),
                "submitted_to_branching_solver": not (
                    apply_positive_length_filter and not length_feasible
                ),
            }
            profile_ids: List[int] = []
            for family in exact_result.families:
                expanded_states = family_expansion.expand_family(
                    dict(family.environment),
                    dict(family.exponent_minimums),
                    policy=expansion_policy,
                    depth=len(family.trace),
                )
                for expanded in expanded_states:
                    state = expanded.state
                    if not base.terminal_state_satisfies_case(case, state):
                        raise AssertionError(
                            f"Expansion policy produced an invalid terminal state for "
                            f"case {case.case_id}, family {family.family_id}, "
                            f"assignment {dict(expanded.assignment)}"
                        )
                    a_text, b_text = _profile_text(case, state)
                    assignment_text = (
                        "finite"
                        if not expanded.assignment
                        else ",".join(
                            f"{name}={value}" for name, value in expanded.assignment
                        )
                    )
                    terminal = TerminalProfile(
                        profile_id=len(works),
                        case=case,
                        state=state,
                        derivation=(
                            f"exact_family:{family.kind}",
                            *family.trace,
                            f"family_expansion:{expansion_policy.kind}:{assignment_text}",
                        ),
                        a_text=a_text,
                        b_text=b_text,
                        exact_family=family,
                        family_exponent_assignment=expanded.assignment,
                        formal_solver_status=exact_result.status,
                    )
                    profile_ids.append(terminal.profile_id)
                    works.append(ProfileWork(terminal=terminal))
                    cases_with_terminals.add(case.case_id)
            case_audit_records[case.case_id]["terminal_profile_ids"] = profile_ids
            progress.update(
                case_index,
                len(cases),
                f"systems processed: {case_index}/{len(cases)}; downstream expansions: {len(works)}",
            )

        exact_summary = exact_word_solver.summarize(exact_case_results)
        formal_audit_summary = {
            "formal_solver_mode": formal_solver_mode,
            "placement_system_count": len(cases),
            "quadratic_system_count": quadratic_count,
            "nonquadratic_system_count": len(cases) - quadratic_count,
            "positive_length_filter_enabled": apply_positive_length_filter,
            "positive_length_filter_rejected_case_count": (
                len(length_infeasible_case_ids) if apply_positive_length_filter else 0
            ),
            "exact_partial_solver": exact_summary,
            "downstream_expansion_count": len(works),
            "family_expansion_policy": expansion_policy.to_dict(),
            "interpretation": {
                "exact_supported": (
                    "The residual graph was completely constructed and every useful cycle "
                    "was compiled into the supported finite/nested-power language."
                ),
                "exact_unsupported_family_language": (
                    "The complete residual graph is known, but at least one useful SCC mixes "
                    "or branches evolving variables beyond the supported nested-power compiler."
                ),
                "unresolved_graph_limit": (
                    "Graph construction reached its node/edge budget; no completeness claim is made."
                ),
                "expansion": (
                    "Finite formal families are always sent downstream. Parametric families "
                    "remain symbolic unless the explicit family expansion policy selects "
                    "one or more exponent assignments."
                ),
            },
        }
        progress.done(
            f"{exact_summary['status_counts']}; {exact_summary['family_count']} exact families; "
            f"{len(works)} downstream expansions"
        )
    else:
        progress.stage(stage, stage_total, "Solving bounded formal word systems and recording search truncation")
        for case_index, case in enumerate(cases, start=1):
            length_feasible = bool(
                case_audit_records[case.case_id]["positive_length_filter"]["feasible"]
            )
            if apply_positive_length_filter and not length_feasible:
                search_audit = formal_audit.positive_length_rejection_audit(
                    initial_inconsistent=case.case_id in initially_inconsistent_case_ids
                )
            else:
                search_audit = formal_audit.explore_case_with_audit(
                    case,
                    max_depth=max_depth,
                    max_states=max_states,
                    max_cycle_unrolls=max_cycle_unrolls,
                )
            case_audit_records[case.case_id]["bounded_search"] = search_audit.to_dict()
            profile_ids: List[int] = []
            for state, derivation in search_audit.terminal_states:
                a_text, b_text = _profile_text(case, state)
                terminal = TerminalProfile(
                    profile_id=len(works),
                    case=case,
                    state=state,
                    derivation=derivation,
                    a_text=a_text,
                    b_text=b_text,
                )
                profile_ids.append(terminal.profile_id)
                works.append(ProfileWork(terminal=terminal))
                cases_with_terminals.add(case.case_id)
            case_audit_records[case.case_id]["terminal_profile_ids"] = profile_ids
            progress.update(
                case_index,
                len(cases),
                f"placements processed: {case_index}/{len(cases)}; terminal profiles: {len(works)}",
            )
        case_audit_list = [case_audit_records[case.case_id] for case in cases]
        formal_audit_summary = formal_audit.summarize_case_audits(case_audit_list)
        formal_audit_summary["formal_solver_mode"] = formal_solver_mode
        progress.done(
            f"{len(works)} terminal profiles from {len(cases_with_terminals)} placements"
        )
        progress.done(
            "search outcome cross-table: "
            f"{formal_audit_summary['exhausted_with_terminal_profiles_case_count']} exhausted with terminals; "
            f"{formal_audit_summary['truncated_with_terminal_profiles_case_count']} truncated with terminals; "
            f"{formal_audit_summary['exhausted_without_terminal_profiles_case_count']} exhausted without terminals; "
            f"{formal_audit_summary['truncated_without_terminal_profiles_case_count']} truncated without terminals"
        )

    case_audit_list = [case_audit_records[case.case_id] for case in cases]

    stage += 1
    progress.stage(
        stage,
        stage_total,
        "Classifying formal profiles by Voderberg contact type and applying the optional selector",
    )
    classification_triples = []
    for index, work in enumerate(works, start=1):
        case_id = work.terminal.case.case_id
        classification = case_type_classifications[case_id]
        work.voderberg_type = classification
        classification_triples.append(
            (work.terminal.profile_id, case_id, classification)
        )
        progress.update(
            index,
            len(works),
            f"profiles classified: {index}/{len(works)}",
        )
    voderberg_type_summary = voderberg_types.summarize_classifications(
        classification_triples, selection=normalized_type_selection
    )
    voderberg_type_summary.update(
        {
            "type1_compatible_placement_case_count_generated": sum(
                classification.is_compatible_with(voderberg_types.TYPE_1)
                for classification in case_type_classifications.values()
            ),
            "type2_compatible_placement_case_count_generated": sum(
                classification.is_compatible_with(voderberg_types.TYPE_2)
                for classification in case_type_classifications.values()
            ),
        }
    )
    pipeline_works = [
        work for work in works
        if work.voderberg_type is not None
        and voderberg_types.matches_selection(
            work.voderberg_type, normalized_type_selection
        )
    ]
    selected_ids_by_case: Dict[int, List[int]] = {}
    for work in pipeline_works:
        selected_ids_by_case.setdefault(work.terminal.case.case_id, []).append(
            work.terminal.profile_id
        )
    for case in cases:
        case_audit_records[case.case_id][
            "terminal_profile_ids_selected_for_downstream_pipeline"
        ] = selected_ids_by_case.get(case.case_id, [])
    progress.done(
        f"type1-compatible: {voderberg_type_summary['type1_compatible_profile_count']}; "
        f"type2-compatible: {voderberg_type_summary['type2_compatible_profile_count']}; "
        f"selected for downstream pipeline: {len(pipeline_works)}/{len(works)} "
        f"(selection={normalized_type_selection})"
    )

    stage += 1
    progress.stage(stage, stage_total, "Resolving point-angle equivalence classes")
    angle_errors = 0
    for index, work in enumerate(pipeline_works, start=1):
        try:
            work.angle_solution = pipeline.solve_point_angles(work.terminal.case, work.terminal.state)
            formal_profile = profile_formatter.build_formal_profile(
                work.terminal.case,
                work.terminal.state,
                work.angle_solution,
            )
            work.terminal_mapping = solution_canonicalization.terminal_mapping_record(
                work.terminal.case,
                work.terminal.state,
                formal_profile,
            )
        except Exception as exc:
            work.angle_error = f"{type(exc).__name__}: {exc}"
            if work.angle_solution is not None:
                work.terminal_mapping_error = work.angle_error
            angle_errors += 1
        progress.update(index, len(pipeline_works), f"profiles processed: {index}/{len(pipeline_works)}; errors: {angle_errors}")
    progress.done(f"{len(pipeline_works) - angle_errors} angle systems resolved; {angle_errors} errors")

    angle_ready = [work for work in pipeline_works if work.angle_solution is not None]

    stage += 1
    progress.stage(
        stage,
        stage_total,
        "Interpreting terminal mappings as formal Straight/Mirror curve terms",
    )
    curve_term_errors = 0
    for index, work in enumerate(angle_ready, start=1):
        try:
            formal_profile = profile_formatter.build_formal_profile(
                work.terminal.case,
                work.terminal.state,
                work.angle_solution,
            )
            expanded_segments = base.substitute_word(
                work.terminal.case.cycle_word,
                work.terminal.state.environment_map(),
            )
            work.curve_term_solution = curve_term_solver.solve_curve_terms(
                curve_variables=formal_profile.curve_parameters,
                occurrences=expanded_segments,
                terminal_mapping=work.terminal_mapping,
                enabled=specialize_curve_terms,
            )
        except Exception as exc:
            work.curve_term_error = f"{type(exc).__name__}: {exc}"
            curve_term_errors += 1
        progress.update(
            index,
            len(angle_ready),
            f"profiles processed: {index}/{len(angle_ready)}; curve-term errors: {curve_term_errors}",
        )
    progress.done(
        f"{len(angle_ready) - curve_term_errors} curve-term solutions; "
        f"{curve_term_errors} errors; enabled={specialize_curve_terms}"
    )

    canonical_class_members: Dict[str, List[int]] = {}
    if canonicalize_solutions:
        stage += 1
        progress.stage(
            stage,
            stage_total,
            "Canonicalizing decorated solutions (contour plus both copy mappings)",
        )
        canonical_errors = 0
        for index, work in enumerate(angle_ready, start=1):
            try:
                formal_profile = profile_formatter.build_formal_profile(
                    work.terminal.case,
                    work.terminal.state,
                    work.angle_solution,
                )
                work.decorated_solution = solution_canonicalization.build_decorated_solution(
                    work.terminal.case,
                    work.terminal.state,
                    formal_profile,
                )
                work.canonical_solution = solution_canonicalization.canonicalize_decorated_data(
                    work.decorated_solution
                )
                canonical_class_members.setdefault(
                    work.canonical_solution.key, []
                ).append(work.terminal.profile_id)
            except Exception as exc:
                work.canonicalization_error = f"{type(exc).__name__}: {exc}"
                canonical_errors += 1
            progress.update(
                index,
                len(angle_ready),
                f"profiles processed: {index}/{len(angle_ready)}; canonicalization errors: {canonical_errors}",
            )
        duplicate_instances = sum(
            max(0, len(profile_ids) - 1)
            for profile_ids in canonical_class_members.values()
        )
        progress.done(
            f"{len(canonical_class_members)} decorated solution classes; "
            f"{duplicate_instances} additional equivalent profile instances; "
            f"{canonical_errors} errors"
        )

        work_by_id = {work.terminal.profile_id: work for work in angle_ready}
        for member_ids in canonical_class_members.values():
            members = [work_by_id[profile_id] for profile_id in member_ids]
            representative = min(
                members,
                key=lambda item: (
                    item.canonical_solution.transform_label != "direct_pole_choice_0",
                    item.canonical_solution.transform_label,
                    item.canonical_solution.canonical_json,
                    item.terminal.profile_id,
                ),
            )
            for member in members:
                member.canonical_representative_profile_id = (
                    representative.terminal.profile_id
                    if reduce_equivalent_profiles
                    else member.terminal.profile_id
                )
                member.canonical_class_size = len(members)

    subsumption_records: Tuple[profile_subsumption.AbsorptionRecord, ...] = ()
    if canonicalize_solutions and reduce_subsumed_profiles:
        stage += 1
        progress.stage(
            stage,
            stage_total,
            "Removing contour-shape profiles absorbed by nonerasing curve substitutions",
        )
        representatives = [
            work for work in angle_ready
            if work.canonical_representative_profile_id == work.terminal.profile_id
            and work.decorated_solution is not None
            and work.canonical_solution is not None
        ]
        entries = []
        for work in representatives:
            all_free = bool(
                work.curve_term_solution is not None
                and all(
                    component.mode == "free"
                    for component in work.curve_term_solution.relation_analysis.components
                )
            )
            entries.append(
                profile_subsumption.ProfileReductionEntry(
                    profile_id=work.terminal.profile_id,
                    decorated_solution=work.decorated_solution,
                    canonical_key=work.canonical_solution.key,
                    canonical_json=work.canonical_solution.canonical_json,
                    all_curve_components_free=all_free,
                )
            )
        retained_ids, subsumption_records = profile_subsumption.reduce_profiles(entries)
        records_by_absorbed = {
            record.absorbed_profile_id: record for record in subsumption_records
        }
        for work in representatives:
            work.subsumption_record = records_by_absorbed.get(work.terminal.profile_id)
        progress.done(
            f"{len(representatives)} canonical profiles; "
            f"{len(subsumption_records)} absorbed; {len(retained_ids)} primitive profiles"
        )

    stage += 1
    progress.stage(stage, stage_total, "Checking total turning of the prototype contour")
    turn_rejects = 0
    for index, work in enumerate(angle_ready, start=1):
        work.total_turn = pipeline.check_total_turn(
            work.terminal.case,
            work.terminal.state,
            angle_solution=work.angle_solution,
        )
        if not work.total_turn.feasible:
            turn_rejects += 1
        progress.update(index, len(angle_ready), f"profiles processed: {index}/{len(angle_ready)}; rejected: {turn_rejects}")
    progress.done(f"{turn_rejects} total-turn rejects")

    stage += 1
    progress.stage(stage, stage_total, "Checking the two three-tile pole-angle constraints")
    pole_rejects = 0
    for index, work in enumerate(angle_ready, start=1):
        work.pole_angles = pipeline.check_pole_angles(
            work.terminal.case,
            work.terminal.state,
            work.angle_solution,
        )
        if not work.pole_angles.feasible:
            pole_rejects += 1
        progress.update(index, len(angle_ready), f"profiles processed: {index}/{len(angle_ready)}; rejected: {pole_rejects}")
    angular_unique_rejects = sum(
        1 for work in pipeline_works
        if work.angle_solution is None
        or not work.total_turn.feasible
        or not work.pole_angles.feasible
    )
    progress.done(f"{pole_rejects} pole rejects; {angular_unique_rejects} unique angular rejects")

    stage += 1
    progress.stage(stage, stage_total, "Checking the prototype translation-holonomy obstruction")
    translation_rejects_all = 0
    additional_translation_rejects = 0
    for index, work in enumerate(angle_ready, start=1):
        work.holonomy = pipeline.check_translation_holonomy(
            work.terminal.case,
            work.terminal.state,
            work.angle_solution,
            work.total_turn,
        )
        rejected = work.holonomy.translation.exact_obstruction
        translation_rejects_all += int(rejected)
        if rejected and work.total_turn.feasible and work.pole_angles.feasible:
            additional_translation_rejects += 1
        progress.update(
            index,
            len(angle_ready),
            f"profiles processed: {index}/{len(angle_ready)}; exact obstructions: {translation_rejects_all}",
        )
    core_survivors = [
        work for work in pipeline_works
        if work.formal_reduction_pass and work.core_pass
    ]
    progress.done(
        f"{translation_rejects_all} translation obstructions total; "
        f"{additional_translation_rejects} additional; {len(core_survivors)} core survivors"
    )

    comparative_counts = {
        "rejected_by_angular_filters_for_all_four_parity_choices": 0,
        "rejected_by_SE2_filters_for_all_four_parity_choices": 0,
        "additionally_rejected_by_placement_time_contact_parity": 0,
    }
    if parity_diagnostics:
        stage += 1
        progress.stage(stage, stage_total, "Diagnostic only: comparing all four parity choices")
        for index, work in enumerate(pipeline_works, start=1):
            outcomes = [_variant_passes(work.terminal.case, work.terminal.state, signs) for signs in MIRROR_VARIANTS]
            angular_any = any(item[0] for item in outcomes)
            se2_any = any(item[1] for item in outcomes)
            if not angular_any:
                comparative_counts["rejected_by_angular_filters_for_all_four_parity_choices"] += 1
            if not se2_any:
                comparative_counts["rejected_by_SE2_filters_for_all_four_parity_choices"] += 1
            if se2_any and not work.core_pass:
                comparative_counts["additionally_rejected_by_placement_time_contact_parity"] += 1
            progress.update(index, len(pipeline_works), f"profiles processed: {index}/{len(pipeline_works)}")
        progress.done("parity comparison complete; diagnostic counts do not change survivors")

    stage += 1
    progress.stage(stage, stage_total, "Building shared inner and outer boundary systems for core survivors")
    external_errors = 0
    for index, work in enumerate(core_survivors, start=1):
        try:
            work.external_system = external.build_joint_boundary_system(
                work.terminal.case, work.terminal.state
            )
        except Exception as exc:
            work.external_error = f"{type(exc).__name__}: {exc}"
            external_errors += 1
        progress.update(index, len(core_survivors), f"profiles processed: {index}/{len(core_survivors)}; build errors: {external_errors}")
    built = [work for work in core_survivors if work.external_system is not None]
    progress.done(f"{len(built)} shared systems built; {external_errors} errors")

    stage += 1
    progress.stage(
        stage, stage_total,
        "Solving the exact global linear contour filter (inner/outer turns, poles, principal boundary turns, and perimeters)",
    )
    global_linear_rejects = 0
    global_linear_errors = 0
    global_linear_angular_rejects = 0
    global_linear_length_rejects = 0
    for index, work in enumerate(built, start=1):
        try:
            work.global_linear_analysis = global_linear.analyze_global_linear_contours(
                work.external_system,
                work.pole_angles,
                enable_angle_block=enable_global_linear_angle_filter,
                enable_length_block=enable_global_linear_length_filter,
            )
            if not work.global_linear_analysis.feasible:
                global_linear_rejects += 1
                global_linear_angular_rejects += int(
                    not work.global_linear_analysis.angle_block.feasible
                )
                global_linear_length_rejects += int(
                    work.global_linear_analysis.angle_block.feasible
                    and not work.global_linear_analysis.length_block.feasible
                )
        except Exception as exc:
            work.global_linear_error = f"{type(exc).__name__}: {exc}"
            global_linear_errors += 1
        progress.update(
            index, len(built),
            f"profiles processed: {index}/{len(built)}; rejected: {global_linear_rejects}; errors: {global_linear_errors}",
        )
    after_global_linear = [
        work for work in built
        if work.global_linear_error is None and work.global_linear_pass
    ]
    progress.done(
        f"{global_linear_rejects} rejected "
        f"({global_linear_angular_rejects} angular, {global_linear_length_rejects} length); "
        f"{global_linear_errors} errors; {len(after_global_linear)} survivors"
    )

    stage += 1
    progress.stage(stage, stage_total, "Checking exact elementary joint translation obstructions")
    joint_translation_rejects = 0
    for index, work in enumerate(after_global_linear, start=1):
        if not work.joint_translation_pass:
            joint_translation_rejects += 1
        progress.update(index, len(after_global_linear), f"profiles processed: {index}/{len(after_global_linear)}; rejected: {joint_translation_rejects}")
    after_joint_translation = [work for work in after_global_linear if work.joint_translation_pass]
    progress.done(f"{joint_translation_rejects} additional rejects; {len(after_joint_translation)} survivors")

    stage += 1
    progress.stage(stage, stage_total, "Checking exact forced point coincidences on both boundaries")
    forced_rejects = 0
    inner_forced_rejects = 0
    outer_forced_rejects = 0
    forced_errors = 0
    for index, work in enumerate(after_joint_translation, start=1):
        try:
            work.inner_points = forced_points.analyze_boundary_path_forced_coincidences(
                work.external_system.inner_boundary,
                work.external_system.curve_turn_solution,
            )
            work.outer_points = forced_points.analyze_boundary_path_forced_coincidences(
                work.external_system.outer_boundary,
                work.external_system.curve_turn_solution,
            )
            inner_reject = not work.inner_points.passes_filter
            outer_reject = not work.outer_points.passes_filter
            inner_forced_rejects += int(inner_reject)
            outer_forced_rejects += int(outer_reject)
            forced_rejects += int(inner_reject or outer_reject)
        except Exception as exc:
            work.forced_point_error = f"{type(exc).__name__}: {exc}"
            forced_errors += 1
        progress.update(index, len(after_joint_translation), f"profiles processed: {index}/{len(after_joint_translation)}; rejected: {forced_rejects}; errors: {forced_errors}")
    after_boundary_points = [
        work for work in after_joint_translation
        if work.forced_point_error is None and work.forced_point_pass
    ]
    progress.done(
        f"{forced_rejects} rejects ({inner_forced_rejects} inner, "
        f"{outer_forced_rejects} outer); {len(after_boundary_points)} survivors"
    )

    stage += 1
    progress.stage(
        stage, stage_total,
        "Placing all three copies in one frame and checking forced global coincidences/overlaps",
    )
    placed_copy_rejects = 0
    placed_copy_errors = 0
    for index, work in enumerate(after_boundary_points, start=1):
        try:
            work.placed_copy_analysis = placed_geometry.analyze_placed_copy_geometry(
                work.terminal.case, work.terminal.state, work.external_system
            )
            placed_copy_rejects += int(not work.placed_copy_analysis.passes_filter)
        except Exception as exc:
            work.placed_copy_error = f"{type(exc).__name__}: {exc}"
            placed_copy_errors += 1
        progress.update(
            index, len(after_boundary_points),
            f"profiles processed: {index}/{len(after_boundary_points)}; "
            f"rejected: {placed_copy_rejects}; errors: {placed_copy_errors}",
        )
    pre_z3_survivors = [
        work for work in after_boundary_points
        if work.placed_copy_error is None and work.placed_copy_pass
    ]
    progress.done(
        f"{placed_copy_rejects} exact global-placement rejects; "
        f"{placed_copy_errors} errors; {len(pre_z3_survivors)} survivors"
    )

    stage += 1
    z3_message = "Running optional Z3/NLSAT checks on final exact-filter survivors" if run_z3 else "Preparing optional Z3/NLSAT problems without running the solver"
    progress.stage(stage, stage_total, z3_message)
    z3_candidates = pre_z3_survivors
    if z3_max_profiles > 0:
        z3_candidates = z3_candidates[:z3_max_profiles]
    z3_counts = {
        "z3_problems_generated_for_core_survivors": 0,
        "z3_encoding_unsupported": 0,
        "z3_encoding_errors": 0,
        "z3_checks_requested": len(z3_candidates) if run_z3 else 0,
        "z3_solver_invocations": 0,
        "z3_exact_unsat": 0,
        "z3_sat_candidates": 0,
        "z3_timeouts": 0,
        "z3_not_installed": 0,
        "z3_solver_errors_or_unknown": 0,
    }
    for index, work in enumerate(z3_candidates, start=1):
        try:
            work.z3_problem = z3_backend.build_z3_problem(
                work.external_system,
                placed_geometry_analysis=work.placed_copy_analysis,
                require_all_chords_nonzero=settings.Z3_REQUIRE_ALL_CHORDS_NONZERO,
                enable_metric_lengths=enable_chord_length_layer,
                enable_signed_areas=(
                    enable_chord_length_layer and enable_signed_area_layer
                ),
            )
            z3_counts["z3_problems_generated_for_core_survivors"] += 1
        except NotImplementedError:
            z3_counts["z3_encoding_unsupported"] += 1
            continue
        except Exception:
            z3_counts["z3_encoding_errors"] += 1
            continue

        if run_z3:
            if progress.enabled:
                print(
                    f"      Z3 {index}/{len(z3_candidates)}: profile "
                    f"{work.terminal.profile_id}, case {work.terminal.case.case_id}...",
                    flush=True,
                )
            work.z3_result = z3_backend.run_z3_problem(
                work.z3_problem, timeout_ms=z3_timeout_ms
            )
            z3_counts["z3_solver_invocations"] += 1
            status = work.z3_result.status
            if progress.enabled:
                print(
                    f"          {status} in {work.z3_result.elapsed_seconds:.3f}s",
                    flush=True,
                )
            if work.z3_result.exact_unsat:
                z3_counts["z3_exact_unsat"] += 1
            elif work.z3_result.sat_candidate:
                z3_counts["z3_sat_candidates"] += 1
            elif status == "timeout":
                z3_counts["z3_timeouts"] += 1
            elif status == "z3_not_installed":
                z3_counts["z3_not_installed"] += 1
            else:
                z3_counts["z3_solver_errors_or_unknown"] += 1
        progress.update(
            index,
            len(z3_candidates),
            f"profiles processed: {index}/{len(z3_candidates)}; unsat: {z3_counts['z3_exact_unsat']}; sat candidates: {z3_counts['z3_sat_candidates']}; timeouts: {z3_counts['z3_timeouts']}",
        )
    final_survivors = [work for work in pre_z3_survivors if work.final_pass]
    progress.done(
        f"{z3_counts['z3_exact_unsat']} exact UNSAT rejects; "
        f"{z3_counts['z3_timeouts']} timeouts; {len(final_survivors)} final survivors"
    )

    stage += 1
    progress.stage(stage, stage_total, "Preparing summary, detailed profiles and survivor file")

    physical_counts = {
        "formal_terminal_profile_instances_before_type_selection": len(works),
        "formal_terminal_profile_instances_selected_for_downstream_pipeline": len(pipeline_works),
        "formal_terminal_profile_instances": len(pipeline_works),
        "rejected_by_angular_filters_under_placement_parity": angular_unique_rejects,
        "additionally_rejected_by_translation_holonomy": additional_translation_rejects,
        "primitive_profiles_reaching_shared_boundary_filters": len(core_survivors),
        "remaining_after_all_current_filters": len(core_survivors),
        "would_be_pruned_at_placement_if_reflections_disabled": sum(
            1 for work in pipeline_works if work.terminal.case.requires_reflection
        ),
    }
    exact_model_rejects = (
        external_errors + global_linear_rejects + global_linear_errors
        + joint_translation_rejects + forced_rejects + forced_errors
        + placed_copy_rejects + placed_copy_errors
    )
    experimental_counts = {
        "profiles_examined": len(core_survivors),
        "formal_reduction_skipped_profile_count": sum(
            1 for work in pipeline_works if not work.formal_reduction_pass
        ),
        "external_boundary_build_errors": external_errors,
        "legacy_joint_rotation_diagnostic_rejections": sum(
            1 for work in built if not work.joint_rotation_pass
        ),
        "exact_global_linear_contour_rejections": global_linear_rejects,
        "global_linear_angular_rejections": global_linear_angular_rejects,
        "global_linear_length_rejections": global_linear_length_rejects,
        "global_linear_contour_filter_errors": global_linear_errors,
        "exact_elementary_joint_translation_rejections": joint_translation_rejects,
        "exact_inner_forced_point_coincidence_rejections": inner_forced_rejects,
        "exact_outer_forced_point_coincidence_rejections": outer_forced_rejects,
        "exact_forced_point_coincidence_rejections_total": forced_rejects,
        "forced_point_filter_errors": forced_errors,
        "exact_placed_copy_geometry_rejections": placed_copy_rejects,
        "placed_copy_geometry_errors": placed_copy_errors,
        "exact_encoded_model_rejections_total": exact_model_rejects,
        "additional_exact_rejections_among_core_survivors": exact_model_rejects,
        "core_survivors_after_experimental_exact_rejections": len(pre_z3_survivors),
        **z3_counts,
        "final_survivors_after_z3_unsat_rejections": len(final_survivors),
    }

    parity_counts = {"DD": 0, "DR": 0, "RD": 0, "RR": 0}
    for work in pipeline_works:
        parity_counts[work.terminal.case.parity_label] += 1

    detailed_profiles: List[Dict[str, object]] = []
    survivor_profiles: List[Dict[str, object]] = []
    for work in pipeline_works:
        if not collect_profiles and not (collect_survivors and work.final_pass):
            continue
        if work.angle_solution is None or work.total_turn is None or work.pole_angles is None or work.holonomy is None:
            continue
        core_analysis = pipeline.ProfileAnalysis(
            angle_solution=work.angle_solution,
            total_turn=work.total_turn,
            pole_angles=work.pole_angles,
            se2_holonomy=work.holonomy,
        )
        record = results_export.detailed_profile_record(
            profile_id=work.terminal.profile_id,
            case=work.terminal.case,
            state=work.terminal.state,
            derivation=work.terminal.derivation,
            analysis=core_analysis,
            experimental=_experimental_record(work),
        )
        record["voderberg_type"] = (
            work.voderberg_type.to_dict()
            if work.voderberg_type is not None
            else {"schema_version": voderberg_types.SCHEMA_VERSION, "compatible_types": []}
        )
        record["pipeline_status"] = _pipeline_flags(work)
        case_audit = case_audit_records[work.terminal.case.case_id]
        record["formal_equation_audit"] = {
            "case_id": work.terminal.case.case_id,
            "structure": case_audit["structure"],
            "positive_length_filter": case_audit["positive_length_filter"],
            "formal_solver_mode": formal_solver_mode,
            "bounded_search": case_audit.get("bounded_search"),
            "exact_formal_solver": case_audit.get("exact_formal_solver"),
        }
        if work.terminal.exact_family is not None:
            family_record = work.terminal.exact_family.to_profile_dict()
            family_record["expanded_for_downstream"] = True
            family_record["expansion_assignment"] = dict(
                work.terminal.family_exponent_assignment
            )
            family_record["complete_family_geometrically_tested"] = (
                not work.terminal.exact_family.parametric
            )
            record["exact_formal_family"] = family_record
        else:
            record["exact_formal_family"] = None
        if work.curve_term_solution is not None:
            record["curve_term_solution"] = work.curve_term_solution.to_dict()
        else:
            record["curve_term_solution"] = {
                "schema_version": curve_term_solver.SCHEMA_VERSION,
                "status": "error" if work.curve_term_error else "not_run",
                "error": work.curve_term_error,
            }
        if work.canonical_solution is not None:
            members = canonical_class_members.get(work.canonical_solution.key, [])
            equivalence = work.canonical_solution.to_record(
                include_canonical_json=True
            )
            equivalence.update(
                {
                    "class_size_within_bounded_terminal_output": len(members),
                    "representative_profile_id": (
                        work.canonical_representative_profile_id
                        if work.canonical_representative_profile_id is not None
                        else work.terminal.profile_id
                    ),
                }
            )
            record["solution_equivalence"] = equivalence
        elif canonicalize_solutions:
            record["solution_equivalence"] = {
                "schema_version": solution_canonicalization.SCHEMA_VERSION,
                "status": "error",
                "error": work.canonicalization_error,
            }
        record["formal_profile_reduction"] = {
            "canonical_representative_profile_id": work.canonical_representative_profile_id,
            "canonical_class_size": work.canonical_class_size,
            "canonical_equivalent_removed": bool(
                work.canonical_representative_profile_id is not None
                and work.canonical_representative_profile_id != work.terminal.profile_id
            ),
            "subsumption": (
                None if work.subsumption_record is None
                else work.subsumption_record.to_dict()
            ),
            "retained_as_primitive_profile": work.formal_reduction_pass,
        }
        if work.terminal_mapping is not None:
            record["terminal_mapping"] = work.terminal_mapping
        else:
            record["terminal_mapping"] = {
                "schema_version": "terminal-contact-mapping-v1",
                "status": "error",
                "error": work.terminal_mapping_error or "terminal mapping unavailable",
            }
        if not work.formal_reduction_pass:
            record["status"] = {
                "retained": False,
                "stage": work.final_stage(),
                "reasons": [work.final_reason()],
            }
            record["sort"]["retained_priority"] = 1
            record["sort"]["stage_priority"] = 5
        if collect_profiles:
            detailed_profiles.append(record)
        if collect_survivors and work.final_pass:
            survivor_profiles.append(record)

    canonical_class_sizes = [
        len(profile_ids) for profile_ids in canonical_class_members.values()
    ]
    canonicalization_summary = {
        "enabled": canonicalize_solutions,
        "schema_version": solution_canonicalization.SCHEMA_VERSION,
        "profile_instances_with_key": sum(canonical_class_sizes),
        "unique_decorated_solution_class_count": len(canonical_class_sizes),
        "additional_equivalent_profile_instance_count": sum(
            max(0, size - 1) for size in canonical_class_sizes
        ),
        "largest_class_size": max(canonical_class_sizes, default=0),
        "canonicalization_error_count": sum(
            1 for work in angle_ready if work.canonicalization_error is not None
        ),
        "deduplication_applied": bool(canonicalize_solutions and reduce_equivalent_profiles),
        "mapping_included": True,
        "global_mirror_identified": True,
        "parametric_cycle_families_identified": False,
    }

    result = {
        "scope": {
            "placement_cases_with_reflections_allowed": len(cases),
            "placement_cases_direct_copies_only": direct_only_cases,
            "cases_with_at_least_one_downstream_expansion": len(cases_with_terminals),
            "formal_solver_mode": formal_solver_mode,
            "max_solver_depth_per_case": max_depth,
            "max_solver_states_per_case": max_states,
            "max_cycle_unrolls_per_residual_state": max_cycle_unrolls,
            "exact_graph_max_nodes_per_case": exact_graph_max_nodes,
            "exact_graph_max_edges_per_case": exact_graph_max_edges,
            "exact_max_families_per_case": exact_max_families,
            "family_expansion_policy": expansion_policy.to_dict(),
            "curve_term_specialization_enabled": specialize_curve_terms,
            "positive_length_filter_enabled": apply_positive_length_filter,
            "solution_canonicalization_enabled": canonicalize_solutions,
            "global_linear_angle_filter_enabled": enable_global_linear_angle_filter,
            "global_linear_length_filter_enabled": enable_global_linear_length_filter,
            "chord_length_layer_enabled": enable_chord_length_layer,
            "signed_area_layer_enabled": bool(
                enable_chord_length_layer and enable_signed_area_layer
            ),
            "voderberg_type_selection": normalized_type_selection,
            "exhaustiveness": (
                "exact partial formal solving on completely constructed supported residual graphs; "
                "unsupported/truncated systems are retained without downstream profiles; "
                "geometric filters remain necessary-condition filters"
                if formal_solver_mode == "exact-partial"
                else "bounded formal-word enumeration; geometric filters are necessary-condition filters"
            ),
        },
        "method_status": method_status.method_registry(),
        "formal_equation_audit_summary": formal_audit_summary,
        "voderberg_type_classification_summary": voderberg_type_summary,
        "decorated_solution_canonicalization_summary": canonicalization_summary,
        "profile_subsumption_summary": {
            "enabled": bool(canonicalize_solutions and reduce_subsumed_profiles),
            "schema_version": profile_subsumption.SCHEMA_VERSION,
            "absorbed_profile_count": len(subsumption_records),
            "records": [record.to_dict() for record in subsumption_records],
        },
        "physical_pipeline_counts": physical_counts,
        "comparative_parity_impact_counts": comparative_counts,
        "experimental_pipeline_counts": experimental_counts,
        "placement_contact_parity_counts": parity_counts,
        "pipeline_sequence": [
            "placement generation and contact parity",
            "formal equation structure audit",
            "exact positive word-length feasibility filter",
            (
                "exact residual-graph solving with finite/nested-power family compilation"
                if formal_solver_mode == "exact-partial"
                else "bounded formal word solving with depth/state/cycle-cap truncation diagnostics"
            ),
            "independent exponent specialization for the legacy downstream pipeline",
            "formal Voderberg type classification and optional downstream selection",
            "point-angle class resolution",
            "specialized Straight/Mirror curve-term interpretation",
            "optional decorated-solution canonicalization including copy mappings",
            "prototype total-turn filter",
            "two-pole angle filter",
            "prototype translation-holonomy obstruction",
            "optional parity diagnostics",
            "shared inner/outer boundary construction",
            "configurable exact global linear contour blocks (inner/outer angles and normalized perimeters)",
            "elementary joint translation filter",
            "forced point coincidence filter on inner/outer boundaries",
            "shared-frame three-copy coincidence and same-side overlap filter",
            "optional Z3/NLSAT polynomial layers: closure, chord/length metric constraints, signed inner/outer areas, and pointwise global-copy isometries",
        ],
        "interpretation": {
            "z3_unsat": "Exact for the polynomial relaxation and safe for rejection.",
            "z3_sat_or_timeout": "Not a realizability proof; the profile remains a candidate.",
            "survivor_file": settings.AUDIT_SURVIVORS_FILENAME,
            "formal_equation_audit_file": settings.FORMAL_EQUATION_AUDIT_FILENAME,
            "positive_length_filter": (
                "Exact necessary condition for nonempty word substitutions; it is independent "
                "of geometric arc length and rejects before bounded formal branching."
            ),
            "voderberg_type_classifier": (
                "Formal contact-topology compatibility only; not a geometric realizability proof."
            ),
        },
    }
    result["formal_equation_cases"] = case_audit_list
    if collect_profiles:
        result["profiles"] = detailed_profiles
    if collect_survivors:
        result["survivors"] = survivor_profiles
    progress.done(f"{len(detailed_profiles)} detailed records; {len(survivor_profiles)} survivors")
    return result


def _normalize_optional_limit(value: int, option_name: str) -> Optional[int]:
    if value < 0:
        raise ValueError(f"{option_name} must be zero or a positive integer")
    return None if value == settings.FORMAL_SOLVER_UNLIMITED_VALUE else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the sequential contour audit.")
    parser.add_argument(
        "--formal-solver-mode",
        choices=settings.FORMAL_SOLVER_MODE_CHOICES,
        default=settings.DEFAULT_FORMAL_SOLVER_MODE,
        help=(
            "exact-partial builds a complete residual graph within explicit node/edge budgets, "
            "compiles only finite or nested-power families, and leaves harder systems unresolved. "
            "legacy-bounded restores the previous depth/state enumeration."
        ),
    )
    parser.add_argument(
        "--exact-graph-max-nodes",
        type=int,
        default=settings.DEFAULT_EXACT_GRAPH_MAX_NODES,
        help="Maximum residual graph nodes per case in exact-partial mode; 0 removes the bound.",
    )
    parser.add_argument(
        "--exact-graph-max-edges",
        type=int,
        default=settings.DEFAULT_EXACT_GRAPH_MAX_EDGES,
        help="Maximum residual graph edges per case in exact-partial mode; 0 removes the bound.",
    )
    parser.add_argument(
        "--exact-max-families",
        type=int,
        default=settings.DEFAULT_EXACT_MAX_FAMILIES_PER_CASE,
        help="Maximum supported exact formal families compiled for one case.",
    )
    parser.add_argument(
        "--family-expansion-policy",
        choices=settings.FAMILY_EXPANSION_POLICY_CHOICES,
        default=settings.DEFAULT_FAMILY_EXPANSION_POLICY,
        help=(
            "Optional downstream expansion of parametric families: none (default), "
            "minimum, fixed, or every exponent assignment in a bounded range."
        ),
    )
    parser.add_argument(
        "--family-representative-exponent",
        type=int,
        default=settings.DEFAULT_FAMILY_REPRESENTATIVE_EXPONENT,
        help="Exponent value used only by --family-expansion-policy fixed.",
    )
    parser.add_argument(
        "--family-expansion-max-exponent",
        type=int,
        default=settings.DEFAULT_FAMILY_EXPANSION_MAX_EXPONENT,
        help="Inclusive exponent ceiling used only by --family-expansion-policy range.",
    )
    parser.add_argument(
        "--family-expansion-max-specializations",
        type=int,
        default=settings.DEFAULT_FAMILY_EXPANSION_MAX_SPECIALIZATIONS,
        help="Safety cap on finite specializations emitted per formal family.",
    )
    parser.add_argument(
        "--skip-family-representative-expansion",
        action="store_true",
        help=(
            "Deprecated alias forcing --family-expansion-policy none."
        ),
    )
    parser.add_argument(
        "--skip-curve-term-specialization",
        action="store_true",
        help="Disable the independent Straight/Mirror formal interpretation layer.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=settings.DEFAULT_AUDIT_MAX_DEPTH,
        help=(
            "Maximum Nielsen/Levi derivation depth per placement system. "
            f"Use {settings.FORMAL_SOLVER_UNLIMITED_VALUE} for no depth bound; an unbounded search may not terminate."
        ),
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=settings.DEFAULT_AUDIT_MAX_STATES,
        help=(
            "Maximum visited solver states per placement system. "
            f"Use {settings.FORMAL_SOLVER_UNLIMITED_VALUE} for no state bound; an unbounded search may not terminate."
        ),
    )
    parser.add_argument(
        "--max-cycle-unrolls",
        type=int,
        default=settings.DEFAULT_FORMAL_MAX_CYCLE_UNROLLS,
        help=(
            "Temporary maximum number of returns to the same residual equation "
            "system along one branch. 0 disables this independent anti-echo cap."
        ),
    )
    parser.add_argument("--progress-interval", type=int, default=settings.DEFAULT_AUDIT_PROGRESS_INTERVAL)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(settings.AUDIT_SUMMARY_FILENAME))
    parser.add_argument("--profiles-output", type=Path, default=Path(settings.AUDIT_PROFILES_FILENAME))
    parser.add_argument("--survivors-output", type=Path, default=Path(settings.AUDIT_SURVIVORS_FILENAME))
    parser.add_argument("--formal-audit-output", type=Path, default=Path(settings.FORMAL_EQUATION_AUDIT_FILENAME))
    parser.add_argument("--no-profiles-output", action="store_true")
    parser.add_argument(
        "--no-detailed-profiles-output",
        action="store_true",
        help=(
            "Skip the very large all-profile export while still writing the "
            "final-survivor JSON used by the web and geometry tools."
        ),
    )
    parser.add_argument(
        "--voderberg-types",
        default=settings.DEFAULT_VODERBERG_TYPE_SELECTION,
        metavar="SELECTION",
        help=(
            "Restrict every stage after formal terminal-profile construction. "
            "Accepted values: all, type1, type2, type1+type2. "
            "'all' keeps non-Voderberg profiles; 'type1+type2' keeps only either classified type."
        ),
    )
    parser.add_argument("--skip-parity-diagnostics", action="store_true")
    parser.add_argument(
        "--skip-positive-length-filter",
        action="store_true",
        help=(
            "Disable the exact pre-solver rejection based on strictly positive "
            "word lengths. The analysis is still recorded in the formal audit JSON."
        ),
    )
    parser.add_argument(
        "--skip-solution-canonicalization",
        action="store_true",
        help="Disable decorated-solution canonicalization and both formal reductions.",
    )
    parser.add_argument(
        "--keep-equivalent-profiles",
        action="store_true",
        help="Keep all symmetry-equivalent profile instances in the survivor pipeline.",
    )
    parser.add_argument(
        "--skip-profile-subsumption",
        action="store_true",
        help="Disable removal of contour-shape profiles absorbed by curve substitution.",
    )
    parser.add_argument(
        "--skip-global-angle-filter",
        action="store_true",
        help="Disable the exact rational inner/outer angular contour block.",
    )
    parser.add_argument(
        "--skip-global-length-filter",
        action="store_true",
        help="Disable the exact rational normalized inner/outer perimeter block.",
    )
    parser.add_argument(
        "--skip-chord-length-layer",
        action="store_true",
        help=(
            "Disable the polynomial chord/length layer. This also disables the "
            "dependent signed-area layer."
        ),
    )
    parser.add_argument(
        "--skip-signed-area-layer",
        action="store_true",
        help="Disable signed arc areas and the identity A_external = 3*A_inner.",
    )
    z3_group = parser.add_mutually_exclusive_group()
    z3_group.add_argument("--run-z3", dest="run_z3", action="store_true")
    z3_group.add_argument("--skip-z3", dest="run_z3", action="store_false")
    parser.set_defaults(run_z3=settings.DEFAULT_RUN_Z3)
    parser.add_argument("--z3-max-profiles", type=int, default=settings.DEFAULT_Z3_MAX_PROFILES)
    parser.add_argument("--z3-timeout-ms", type=int, default=settings.Z3_DEFAULT_TIMEOUT_MS)
    return parser


def _atomic_json(
    path: Path,
    payload: Dict[str, object],
    *,
    indent: Optional[int] = 2,
) -> None:
    """Write JSON atomically without materializing the complete text in RAM."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoder = json.JSONEncoder(
        ensure_ascii=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            # JSONEncoder.iterencode deliberately yields very small lexical
            # fragments.  Writing each fragment separately becomes extremely
            # slow for the richer exact-family ASTs, so retain bounded memory
            # while coalescing fragments into ordinary I/O-sized blocks.
            pending: List[str] = []
            pending_size = 0
            flush_threshold = settings.WEB_STREAM_CHUNK_SIZE
            for chunk in encoder.iterencode(payload):
                pending.append(chunk)
                pending_size += len(chunk)
                if pending_size >= flush_threshold:
                    handle.write("".join(pending))
                    pending.clear()
                    pending_size = 0
            if pending:
                handle.write("".join(pending))
            handle.write("\n")
            handle.flush()
        temporary.replace(path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def main() -> int:
    args = build_parser().parse_args()
    try:
        max_depth = _normalize_optional_limit(args.max_depth, "--max-depth")
        max_states = _normalize_optional_limit(args.max_states, "--max-states")
        max_cycle_unrolls = _normalize_optional_limit(
            args.max_cycle_unrolls, "--max-cycle-unrolls"
        )
        exact_graph_max_nodes = _normalize_optional_limit(
            args.exact_graph_max_nodes, "--exact-graph-max-nodes"
        )
        exact_graph_max_edges = _normalize_optional_limit(
            args.exact_graph_max_edges, "--exact-graph-max-edges"
        )
        if args.exact_max_families <= 0:
            raise ValueError("--exact-max-families must be positive")
        if args.family_representative_exponent < 0:
            raise ValueError("--family-representative-exponent must be nonnegative")
        if args.family_expansion_max_exponent < 0:
            raise ValueError("--family-expansion-max-exponent must be nonnegative")
        if args.family_expansion_max_specializations <= 0:
            raise ValueError("--family-expansion-max-specializations must be positive")
        resolved_family_expansion_policy = (
            family_expansion.POLICY_NONE
            if args.skip_family_representative_expansion
            else args.family_expansion_policy
        )
        voderberg_type_selection = voderberg_types.normalize_selection(
            args.voderberg_types
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not args.quiet:
        if args.formal_solver_mode == "exact-partial":
            nodes_text = "unbounded" if exact_graph_max_nodes is None else str(exact_graph_max_nodes)
            edges_text = "unbounded" if exact_graph_max_edges is None else str(exact_graph_max_edges)
            print(
                f"Formal solver mode: exact-partial; graph nodes = {nodes_text}; "
                f"graph edges = {edges_text}; max families = {args.exact_max_families}; "
                f"family expansion policy = {resolved_family_expansion_policy}",
                flush=True,
            )
            if exact_graph_max_nodes is None or exact_graph_max_edges is None:
                print(
                    "Warning: an unbounded residual graph can fail to terminate on unsupported systems.",
                    flush=True,
                )
        else:
            depth_text = "unbounded" if max_depth is None else str(max_depth)
            states_text = "unbounded" if max_states is None else str(max_states)
            cycle_text = "disabled" if max_cycle_unrolls is None else str(max_cycle_unrolls)
            print(
                f"Formal solver mode: legacy-bounded; max depth = {depth_text}; "
                f"max states = {states_text}; cycle unroll cap = {cycle_text}",
                flush=True,
            )

    result = audit(
        max_depth,
        max_states,
        formal_solver_mode=args.formal_solver_mode,
        exact_graph_max_nodes=exact_graph_max_nodes,
        exact_graph_max_edges=exact_graph_max_edges,
        exact_max_families=args.exact_max_families,
        family_expansion_policy=resolved_family_expansion_policy,
        family_expansion_max_exponent=args.family_expansion_max_exponent,
        family_expansion_max_specializations=args.family_expansion_max_specializations,
        representative_exponent_value=args.family_representative_exponent,
        specialize_curve_terms=not args.skip_curve_term_specialization,
        max_cycle_unrolls=max_cycle_unrolls,
        apply_positive_length_filter=not args.skip_positive_length_filter,
        canonicalize_solutions=not args.skip_solution_canonicalization,
        reduce_equivalent_profiles=not args.keep_equivalent_profiles,
        reduce_subsumed_profiles=not args.skip_profile_subsumption,
        enable_global_linear_angle_filter=not args.skip_global_angle_filter,
        enable_global_linear_length_filter=not args.skip_global_length_filter,
        enable_chord_length_layer=not args.skip_chord_length_layer,
        enable_signed_area_layer=(
            not args.skip_chord_length_layer
            and not args.skip_signed_area_layer
        ),
        collect_profiles=(
            not args.no_profiles_output and not args.no_detailed_profiles_output
        ),
        collect_survivors=not args.no_profiles_output,
        run_z3=args.run_z3,
        z3_max_profiles=args.z3_max_profiles,
        z3_timeout_ms=args.z3_timeout_ms,
        show_progress=not args.quiet,
        progress_interval=args.progress_interval,
        parity_diagnostics=not args.skip_parity_diagnostics,
        voderberg_type_selection=voderberg_type_selection,
    )
    profiles = result.pop("profiles", [])
    survivors = result.pop("survivors", [])
    formal_equation_cases = result.pop("formal_equation_cases", [])
    print(f"Writing: {args.output}", flush=True)
    _atomic_json(args.output, result)
    print(f"Wrote: {args.output}", flush=True)
    print(f"Writing: {args.formal_audit_output}", flush=True)
    _atomic_json(
        args.formal_audit_output,
        {
            "metadata": {
                "source_audit": str(args.output),
                "case_count": len(formal_equation_cases),
                "formal_solver_mode": args.formal_solver_mode,
                "max_solver_depth_per_case": max_depth,
                "max_solver_states_per_case": max_states,
                "max_cycle_unrolls_per_residual_state": max_cycle_unrolls,
                "exact_graph_max_nodes_per_case": exact_graph_max_nodes,
                "exact_graph_max_edges_per_case": exact_graph_max_edges,
                "exact_max_families_per_case": args.exact_max_families,
                "family_expansion_policy": resolved_family_expansion_policy,
                "family_expansion_fixed_exponent": args.family_representative_exponent,
                "family_expansion_max_exponent": args.family_expansion_max_exponent,
                "family_expansion_max_specializations": args.family_expansion_max_specializations,
                "curve_term_specialization_enabled": not args.skip_curve_term_specialization,
                "positive_length_filter_enabled": not args.skip_positive_length_filter,
                "global_linear_angle_filter_enabled": not args.skip_global_angle_filter,
                "global_linear_length_filter_enabled": not args.skip_global_length_filter,
                "chord_length_layer_enabled": not args.skip_chord_length_layer,
                "signed_area_layer_enabled": bool(
                    not args.skip_chord_length_layer
                    and not args.skip_signed_area_layer
                ),
                "voderberg_type_selection": voderberg_type_selection,
            },
            "summary": result["formal_equation_audit_summary"],
            "cases": formal_equation_cases,
        },
    )
    print(f"Wrote: {args.formal_audit_output}", flush=True)

    if not args.no_profiles_output:
        common_metadata = {
            "schema_version": settings.WEB_SCHEMA_VERSION,
            "source_audit": str(args.output),
            "bounded": args.formal_solver_mode == "legacy-bounded",
            "formal_solver_mode": args.formal_solver_mode,
            "max_solver_depth_per_case": max_depth,
            "max_solver_states_per_case": max_states,
            "max_cycle_unrolls_per_residual_state": max_cycle_unrolls,
            "exact_graph_max_nodes_per_case": exact_graph_max_nodes,
            "exact_graph_max_edges_per_case": exact_graph_max_edges,
            "exact_max_families_per_case": args.exact_max_families,
            "family_expansion_policy": resolved_family_expansion_policy,
            "family_expansion_fixed_exponent": args.family_representative_exponent,
            "family_expansion_max_exponent": args.family_expansion_max_exponent,
            "family_expansion_max_specializations": args.family_expansion_max_specializations,
            "curve_term_specialization_enabled": not args.skip_curve_term_specialization,
            "positive_length_filter_enabled": not args.skip_positive_length_filter,
            "solution_canonicalization_enabled": not args.skip_solution_canonicalization,
            "global_linear_angle_filter_enabled": not args.skip_global_angle_filter,
            "global_linear_length_filter_enabled": not args.skip_global_length_filter,
            "chord_length_layer_enabled": not args.skip_chord_length_layer,
            "signed_area_layer_enabled": bool(
                not args.skip_chord_length_layer
                and not args.skip_signed_area_layer
            ),
            "voderberg_type_selection": voderberg_type_selection,
        }
        # Save the operationally important survivor file first.  A failure while
        # writing the much larger all-profile export therefore cannot erase the
        # data needed by the geometry search or the default web report.
        print(
            f"Writing: {args.survivors_output} ({len(survivors)} survivor records, compact streaming JSON)",
            flush=True,
        )
        _atomic_json(
            args.survivors_output,
            {
                "metadata": {
                    **common_metadata,
                    "profile_count": len(survivors),
                    "selection": "passed every exact filter; Z3 SAT/timeouts/unknown remain candidates, Z3 UNSAT is removed",
                },
                "profiles": survivors,
            },
            indent=None,
        )
        print(f"Wrote: {args.survivors_output}", flush=True)

        if not args.no_detailed_profiles_output:
            print(
                f"Writing: {args.profiles_output} ({len(profiles)} detailed records, compact streaming JSON)",
                flush=True,
            )
            _atomic_json(
                args.profiles_output,
                {
                    "metadata": {**common_metadata, "profile_count": len(profiles)},
                    "summary": {
                        "formal_equation_audit_summary": result["formal_equation_audit_summary"],
                        "voderberg_type_classification_summary": result["voderberg_type_classification_summary"],
                        "physical_pipeline_counts": result["physical_pipeline_counts"],
                        "comparative_parity_impact_counts": result["comparative_parity_impact_counts"],
                        "experimental_pipeline_counts": result["experimental_pipeline_counts"],
                        "placement_contact_parity_counts": result["placement_contact_parity_counts"],
                        "interpretation": result["interpretation"],
                    },
                    "profiles": profiles,
                },
                indent=None,
            )
            print(f"Wrote: {args.profiles_output}", flush=True)

    print(json.dumps(result["physical_pipeline_counts"], indent=2, ensure_ascii=True))
    print(json.dumps(result["experimental_pipeline_counts"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
