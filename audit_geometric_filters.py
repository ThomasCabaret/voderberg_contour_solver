#!/usr/bin/env python3
"""Sequential bounded audit of formal Voderberg contour profiles.

Every independent filter has its own pass over the profile collection.  Fast
filters are evaluated for all terminal profiles so the detailed report retains
complete per-filter diagnostics.  Expensive shared-boundary and Z3 stages are
cascaded and receive only profiles that survived every preceding filter.
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
import forced_point_coincidence as forced_points
import formal_equation_audit as formal_audit
import joint_translation_z3 as z3_backend
import method_status
import results_export
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
    inner_points: Optional[forced_points.ForcedPointCoincidenceAnalysis] = None
    outer_points: Optional[forced_points.ForcedPointCoincidenceAnalysis] = None
    forced_point_error: Optional[str] = None
    z3_problem: Optional[z3_backend.Z3Problem] = None
    z3_result: Optional[z3_backend.Z3Result] = None

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
    def pre_z3_pass(self) -> bool:
        return bool(
            self.core_pass
            and self.external_error is None
            and self.joint_rotation_pass
            and self.joint_translation_pass
            and self.forced_point_error is None
            and self.forced_point_pass
        )

    @property
    def final_pass(self) -> bool:
        if not self.pre_z3_pass:
            return False
        return not bool(self.z3_result and self.z3_result.exact_unsat)

    def final_stage(self) -> str:
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
        if self.external_system is not None and not self.external_system.rotation_analysis.feasible:
            return "joint_rotation"
        if self.external_system is not None and self.external_system.translation_analysis.exact_obstruction:
            return "joint_translation"
        if self.forced_point_error:
            return "forced_point_error"
        if not self.forced_point_pass:
            return "forced_point_coincidence"
        if self.z3_result is not None and self.z3_result.exact_unsat:
            return "z3_unsat"
        return "retained"

    def final_reason(self) -> Optional[str]:
        stage = self.final_stage()
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
        if stage == "joint_rotation":
            return self.external_system.rotation_analysis.reason
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
        return {
            "status": "not_run_after_core_rejection" if not work.core_pass else "external_boundary_error",
            "reason": work.external_error,
            "affects_core_status": False,
            "exact_encoded_model_rejection": False,
            "external_boundary": None,
            "z3_problem": None,
            "z3_result": None,
        }

    if not work.external_system.rotation_analysis.feasible:
        status = "exact_joint_rotation_reject"
        reason = work.external_system.rotation_analysis.reason
    elif work.external_system.translation_analysis.exact_obstruction:
        status = "exact_joint_translation_reject"
        reason = work.external_system.translation_analysis.reason
    elif work.forced_point_error:
        status = "forced_point_filter_error"
        reason = work.forced_point_error
    elif not work.forced_point_pass:
        status = "exact_forced_point_reject"
        reason = work.final_reason()
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
        z3_problem=work.z3_problem,
        z3_result=work.z3_result,
    )
    return analysis.to_dict()


def _pipeline_flags(work: ProfileWork) -> Dict[str, object]:
    return {
        "point_angles_resolved": work.angle_solution is not None,
        "total_turn_pass": bool(work.total_turn and work.total_turn.feasible),
        "pole_angles_pass": bool(work.pole_angles and work.pole_angles.feasible),
        "core_translation_pass": bool(
            work.holonomy and not work.holonomy.translation.exact_obstruction
        ),
        "core_pass": work.core_pass,
        "external_boundary_built": work.external_system is not None,
        "joint_rotation_pass": work.joint_rotation_pass,
        "joint_translation_pass": work.joint_translation_pass,
        "forced_point_pass": work.forced_point_pass,
        "z3_status": None if work.z3_result is None else work.z3_result.status,
        "final_retained": work.final_pass,
        "final_stage": work.final_stage(),
        "final_reason": work.final_reason(),
    }


def audit(
    max_depth: Optional[int],
    max_states: Optional[int],
    *,
    collect_profiles: bool = False,
    collect_survivors: bool = False,
    run_z3: bool = settings.DEFAULT_RUN_Z3,
    z3_max_profiles: int = settings.DEFAULT_Z3_MAX_PROFILES,
    z3_timeout_ms: int = settings.Z3_DEFAULT_TIMEOUT_MS,
    show_progress: bool = settings.DEFAULT_SHOW_AUDIT_PROGRESS,
    progress_interval: int = settings.DEFAULT_AUDIT_PROGRESS_INTERVAL,
    parity_diagnostics: bool = settings.DEFAULT_RUN_PARITY_DIAGNOSTICS,
) -> Dict[str, object]:
    progress = AuditProgress(show_progress, progress_interval)
    stage_total = 14 if parity_diagnostics else 13
    stage = 0

    stage += 1
    progress.stage(stage, stage_total, "Generating placement cases and fixing contact parity")
    cases = list(base.enumerate_placement_cases())
    direct_only_cases = sum(1 for _ in base.enumerate_placement_cases(allow_reflections=False))
    progress.done(f"{len(cases)} placements; {direct_only_cases} use direct copies only")

    stage += 1
    progress.stage(stage, stage_total, "Auditing the structure of generated formal word systems")
    case_audit_records: Dict[int, Dict[str, object]] = {}
    for case_index, case in enumerate(cases, start=1):
        case_audit_records[case.case_id] = {
            "case_id": case.case_id,
            "placement": case.to_dict(),
            "structure": formal_audit.analyze_case_structure(case),
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
    progress.stage(stage, stage_total, "Solving bounded formal word systems and recording search truncation")
    works: List[ProfileWork] = []
    cases_with_terminals = set()
    for case_index, case in enumerate(cases, start=1):
        search_audit = formal_audit.explore_case_with_audit(
            case, max_depth=max_depth, max_states=max_states
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
    progress.done(
        f"{len(works)} terminal profiles from {len(cases_with_terminals)} placements"
    )
    progress.done(
        f"{formal_audit_summary['initially_inconsistent_case_count']} initially inconsistent; "
        f"{formal_audit_summary['submitted_to_branching_solver_case_count']} submitted to the branching solver"
    )
    progress.done(
        "search outcome cross-table: "
        f"{formal_audit_summary['exhausted_with_terminal_profiles_case_count']} exhausted with terminals; "
        f"{formal_audit_summary['truncated_with_terminal_profiles_case_count']} truncated with terminals; "
        f"{formal_audit_summary['exhausted_without_terminal_profiles_case_count']} exhausted without terminals; "
        f"{formal_audit_summary['truncated_without_terminal_profiles_case_count']} truncated without terminals"
    )
    progress.done(
        f"{formal_audit_summary['terminal_profile_sets_potentially_incomplete_due_to_bounds_case_count']} productive searches may be missing additional profiles; "
        f"{formal_audit_summary['solution_existence_unresolved_due_to_bounds_case_count']} empty searches remain unresolved by the configured bounds"
    )

    stage += 1
    progress.stage(stage, stage_total, "Resolving point-angle equivalence classes")
    angle_errors = 0
    for index, work in enumerate(works, start=1):
        try:
            work.angle_solution = pipeline.solve_point_angles(work.terminal.case, work.terminal.state)
        except Exception as exc:
            work.angle_error = f"{type(exc).__name__}: {exc}"
            angle_errors += 1
        progress.update(index, len(works), f"profiles processed: {index}/{len(works)}; errors: {angle_errors}")
    progress.done(f"{len(works) - angle_errors} angle systems resolved; {angle_errors} errors")

    angle_ready = [work for work in works if work.angle_solution is not None]

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
        1 for work in works
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
    core_survivors = [work for work in works if work.core_pass]
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
        for index, work in enumerate(works, start=1):
            outcomes = [_variant_passes(work.terminal.case, work.terminal.state, signs) for signs in MIRROR_VARIANTS]
            angular_any = any(item[0] for item in outcomes)
            se2_any = any(item[1] for item in outcomes)
            if not angular_any:
                comparative_counts["rejected_by_angular_filters_for_all_four_parity_choices"] += 1
            if not se2_any:
                comparative_counts["rejected_by_SE2_filters_for_all_four_parity_choices"] += 1
            if se2_any and not work.core_pass:
                comparative_counts["additionally_rejected_by_placement_time_contact_parity"] += 1
            progress.update(index, len(works), f"profiles processed: {index}/{len(works)}")
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
    progress.stage(stage, stage_total, "Solving exact joint rotation constraints")
    joint_rotation_rejects = 0
    for index, work in enumerate(built, start=1):
        if not work.external_system.rotation_analysis.feasible:
            joint_rotation_rejects += 1
        progress.update(index, len(built), f"profiles processed: {index}/{len(built)}; rejected: {joint_rotation_rejects}")
    after_joint_rotation = [work for work in built if work.joint_rotation_pass]
    progress.done(f"{joint_rotation_rejects} rejected; {len(after_joint_rotation)} survivors")

    stage += 1
    progress.stage(stage, stage_total, "Checking exact elementary joint translation obstructions")
    joint_translation_rejects = 0
    for index, work in enumerate(after_joint_rotation, start=1):
        if not work.joint_translation_pass:
            joint_translation_rejects += 1
        progress.update(index, len(after_joint_rotation), f"profiles processed: {index}/{len(after_joint_rotation)}; rejected: {joint_translation_rejects}")
    after_joint_translation = [work for work in after_joint_rotation if work.joint_translation_pass]
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
    pre_z3_survivors = [work for work in after_joint_translation if work.forced_point_error is None and work.forced_point_pass]
    progress.done(f"{forced_rejects} rejects ({inner_forced_rejects} inner, {outer_forced_rejects} outer); {len(pre_z3_survivors)} survivors")

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
                require_all_chords_nonzero=settings.Z3_REQUIRE_ALL_CHORDS_NONZERO,
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
        "formal_terminal_profile_instances": len(works),
        "rejected_by_angular_filters_under_placement_parity": angular_unique_rejects,
        "additionally_rejected_by_translation_holonomy": additional_translation_rejects,
        "remaining_after_all_current_filters": len(core_survivors),
        "would_be_pruned_at_placement_if_reflections_disabled": sum(
            1 for work in works if work.terminal.case.requires_reflection
        ),
    }
    exact_model_rejects = external_errors + joint_rotation_rejects + joint_translation_rejects + forced_rejects + forced_errors
    experimental_counts = {
        "profiles_examined": len(core_survivors),
        "external_boundary_build_errors": external_errors,
        "exact_joint_rotation_rejections": joint_rotation_rejects,
        "exact_elementary_joint_translation_rejections": joint_translation_rejects,
        "exact_inner_forced_point_coincidence_rejections": inner_forced_rejects,
        "exact_outer_forced_point_coincidence_rejections": outer_forced_rejects,
        "exact_forced_point_coincidence_rejections_total": forced_rejects,
        "forced_point_filter_errors": forced_errors,
        "exact_encoded_model_rejections_total": exact_model_rejects,
        "additional_exact_rejections_among_core_survivors": exact_model_rejects,
        "core_survivors_after_experimental_exact_rejections": len(pre_z3_survivors),
        **z3_counts,
        "final_survivors_after_z3_unsat_rejections": len(final_survivors),
    }

    parity_counts = {"DD": 0, "DR": 0, "RD": 0, "RR": 0}
    for work in works:
        parity_counts[work.terminal.case.parity_label] += 1

    detailed_profiles: List[Dict[str, object]] = []
    survivor_profiles: List[Dict[str, object]] = []
    for work in works:
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
        record["pipeline_status"] = _pipeline_flags(work)
        case_audit = case_audit_records[work.terminal.case.case_id]
        record["formal_equation_audit"] = {
            "case_id": work.terminal.case.case_id,
            "structure": case_audit["structure"],
            "bounded_search": case_audit["bounded_search"],
        }
        if collect_profiles:
            detailed_profiles.append(record)
        if collect_survivors and work.final_pass:
            survivor_profiles.append(record)

    result = {
        "scope": {
            "placement_cases_with_reflections_allowed": len(cases),
            "placement_cases_direct_copies_only": direct_only_cases,
            "cases_with_at_least_one_terminal_profile_in_bound": len(cases_with_terminals),
            "max_solver_depth_per_case": max_depth,
            "max_solver_states_per_case": max_states,
            "exhaustiveness": "bounded formal-word enumeration; geometric filters are necessary-condition filters",
        },
        "method_status": method_status.method_registry(),
        "formal_equation_audit_summary": formal_audit_summary,
        "physical_pipeline_counts": physical_counts,
        "comparative_parity_impact_counts": comparative_counts,
        "experimental_pipeline_counts": experimental_counts,
        "placement_contact_parity_counts": parity_counts,
        "pipeline_sequence": [
            "placement generation and contact parity",
            "formal equation structure audit",
            "bounded formal word solving with truncation diagnostics",
            "point-angle class resolution",
            "prototype total-turn filter",
            "two-pole angle filter",
            "prototype translation-holonomy obstruction",
            "optional parity diagnostics",
            "shared inner/outer boundary construction",
            "joint rotation filter",
            "elementary joint translation filter",
            "forced point coincidence filter",
            "optional Z3/NLSAT polynomial filter",
        ],
        "interpretation": {
            "z3_unsat": "Exact for the polynomial relaxation and safe for rejection.",
            "z3_sat_or_timeout": "Not a realizability proof; the profile remains a candidate.",
            "survivor_file": settings.AUDIT_SURVIVORS_FILENAME,
            "formal_equation_audit_file": settings.FORMAL_EQUATION_AUDIT_FILENAME,
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
    parser = argparse.ArgumentParser(description="Run the sequential bounded contour audit.")
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
    parser.add_argument("--skip-parity-diagnostics", action="store_true")
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
            for chunk in encoder.iterencode(payload):
                handle.write(chunk)
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
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not args.quiet:
        depth_text = "unbounded" if max_depth is None else str(max_depth)
        states_text = "unbounded" if max_states is None else str(max_states)
        print(
            f"Formal solver bounds: max depth per system = {depth_text}; "
            f"max visited states per system = {states_text}",
            flush=True,
        )
        if max_depth is None or max_states is None:
            print(
                "Warning: removing a formal-search bound can make the audit fail to terminate on cyclic systems.",
                flush=True,
            )

    result = audit(
        max_depth,
        max_states,
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
                "max_solver_depth_per_case": max_depth,
                "max_solver_states_per_case": max_states,
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
            "bounded": True,
            "max_solver_depth_per_case": max_depth,
            "max_solver_states_per_case": max_states,
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
