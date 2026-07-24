"""Experimental inner/outer-boundary analysis for one terminal profile."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import external_boundary_constraints as external
import forced_point_coincidence as point_filter
import joint_translation_z3 as z3_backend
import settings
import symbolic_enumerator as base


@dataclass(frozen=True)
class ExperimentalProfileAnalysis:
    status: str
    reason: Optional[str]
    external_system: Optional[external.JointBoundarySystem]
    inner_point_coincidence: Optional[point_filter.ForcedPointCoincidenceAnalysis]
    outer_point_coincidence: Optional[point_filter.ForcedPointCoincidenceAnalysis]
    z3_problem: Optional[z3_backend.Z3Problem]
    z3_result: Optional[z3_backend.Z3Result]

    @property
    def forced_point_rejection(self) -> bool:
        return any(
            analysis is not None and not analysis.passes_filter
            for analysis in (
                self.inner_point_coincidence,
                self.outer_point_coincidence,
            )
        )

    @property
    def exact_encoded_model_rejection(self) -> bool:
        if self.external_system is not None:
            if not self.external_system.rotation_analysis.feasible:
                return True
            if self.external_system.translation_analysis.exact_obstruction:
                return True
        if self.forced_point_rejection:
            return True
        return bool(self.z3_result and self.z3_result.exact_unsat)

    def to_dict(self) -> Dict[str, object]:
        system = self.external_system
        problem = self.z3_problem
        return {
            "status": self.status,
            "reason": self.reason,
            "affects_core_status": False,
            "exact_encoded_model_rejection": self.exact_encoded_model_rejection,
            "model_scope": (
                "Shared inner/outer closure model with exact symbolic forced-point "
                "coincidence checks and an optional polynomial Z3/NLSAT relaxation. "
                "Complete pointwise rigid-isometry realization constraints are not "
                "yet encoded."
            ),
            "external_boundary": None if system is None else {
                "outer_boundary": system.outer_boundary.to_dict(),
                "projection_curve_turn_constraints": system.curve_turn_solution.to_dict(),
                "rotation_equations": [
                    equation.to_dict() for equation in system.rotation_equations
                ],
                "translation_equations": [
                    equation.to_dict() for equation in system.translation_equations
                ],
                "joint_rotation_analysis": system.rotation_analysis.to_dict(),
                "joint_translation_analysis": system.translation_analysis.to_dict(),
                "forced_point_coincidence": {
                    "inner_boundary": (
                        None
                        if self.inner_point_coincidence is None
                        else self.inner_point_coincidence.to_dict()
                    ),
                    "outer_boundary": (
                        None
                        if self.outer_point_coincidence is None
                        else self.outer_point_coincidence.to_dict()
                    ),
                },
            },
            "z3_problem": None if problem is None else {
                **problem.to_dict(),
                "smt2_generated": True,
                "smt2_embedded_in_report": False,
            },
            "z3_result": None if self.z3_result is None else self.z3_result.to_dict(),
        }


def _first_exact_rejection_reason(
    system: external.JointBoundarySystem,
    inner_points: point_filter.ForcedPointCoincidenceAnalysis,
    outer_points: point_filter.ForcedPointCoincidenceAnalysis,
) -> Optional[str]:
    if not system.rotation_analysis.feasible:
        return system.rotation_analysis.reason
    if system.translation_analysis.exact_obstruction:
        return system.translation_analysis.reason
    if not inner_points.passes_filter:
        return "Inner boundary: " + (
            inner_points.discard_reason or "forced point coincidence"
        )
    if not outer_points.passes_filter:
        return "Outer boundary: " + (
            outer_points.discard_reason or "forced point coincidence"
        )
    return None



def execute_prepared_z3(
    analysis: ExperimentalProfileAnalysis,
    *,
    timeout_ms: int = settings.Z3_DEFAULT_TIMEOUT_MS,
) -> ExperimentalProfileAnalysis:
    """Run Z3 for an already prepared profile analysis."""
    if analysis.z3_problem is None:
        return analysis
    result = z3_backend.run_z3_problem(analysis.z3_problem, timeout_ms=timeout_ms)
    return ExperimentalProfileAnalysis(
        status=result.status,
        reason=result.reason,
        external_system=analysis.external_system,
        inner_point_coincidence=analysis.inner_point_coincidence,
        outer_point_coincidence=analysis.outer_point_coincidence,
        z3_problem=analysis.z3_problem,
        z3_result=result,
    )

def analyze_experimental_profile(
    case: base.PlacementCase,
    state: base.SolverState,
    *,
    prepare_z3: bool = settings.DEFAULT_PREPARE_JOINT_TRANSLATION,
    run_z3: bool = settings.DEFAULT_RUN_Z3,
    timeout_ms: int = settings.Z3_DEFAULT_TIMEOUT_MS,
) -> ExperimentalProfileAnalysis:
    try:
        system = external.build_joint_boundary_system(case, state)
    except Exception as exc:
        return ExperimentalProfileAnalysis(
            "external_boundary_error",
            f"{type(exc).__name__}: {exc}",
            None,
            None,
            None,
            None,
            None,
        )

    try:
        inner_points = point_filter.analyze_boundary_path_forced_coincidences(
            system.inner_boundary,
            system.curve_turn_solution,
        )
        outer_points = point_filter.analyze_boundary_path_forced_coincidences(
            system.outer_boundary,
            system.curve_turn_solution,
        )
    except Exception as exc:
        return ExperimentalProfileAnalysis(
            "forced_point_filter_error",
            f"{type(exc).__name__}: {exc}",
            system,
            None,
            None,
            None,
            None,
        )

    rejection_reason = _first_exact_rejection_reason(
        system,
        inner_points,
        outer_points,
    )
    if rejection_reason is not None:
        return ExperimentalProfileAnalysis(
            "exact_encoded_model_reject",
            rejection_reason,
            system,
            inner_points,
            outer_points,
            None,
            None,
        )

    if not prepare_z3 and not run_z3:
        return ExperimentalProfileAnalysis(
            "external_system_built",
            None,
            system,
            inner_points,
            outer_points,
            None,
            None,
        )

    try:
        problem = z3_backend.build_z3_problem(
            system,
            require_all_chords_nonzero=settings.Z3_REQUIRE_ALL_CHORDS_NONZERO,
        )
    except NotImplementedError as exc:
        return ExperimentalProfileAnalysis(
            "z3_encoding_unsupported",
            str(exc),
            system,
            inner_points,
            outer_points,
            None,
            None,
        )
    except Exception as exc:
        return ExperimentalProfileAnalysis(
            "z3_encoding_error",
            f"{type(exc).__name__}: {exc}",
            system,
            inner_points,
            outer_points,
            None,
            None,
        )

    if not run_z3:
        return ExperimentalProfileAnalysis(
            "z3_problem_ready",
            None,
            system,
            inner_points,
            outer_points,
            problem,
            None,
        )

    result = z3_backend.run_z3_problem(problem, timeout_ms=timeout_ms)
    return ExperimentalProfileAnalysis(
        result.status,
        result.reason,
        system,
        inner_points,
        outer_points,
        problem,
        result,
    )
