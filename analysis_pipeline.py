"""Composable analysis stages for one terminal formal contour profile.

Each stage is independently callable and returns its native analysis object.
The pipeline only orchestrates them; it does not merge their logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import angle_constraints as angles
import pole_angle_filter as pole
import se2_holonomy_filter as se2
import symbolic_enumerator as base
import turning_filter as turning


@dataclass(frozen=True)
class ProfileAnalysis:
    angle_solution: angles.AngleSolution
    total_turn: turning.TotalTurnAnalysis
    pole_angles: pole.PoleAngleAnalysis
    se2_holonomy: se2.SE2HolonomyAnalysis

    @property
    def angular_pass(self) -> bool:
        return self.total_turn.feasible and self.pole_angles.feasible

    @property
    def translation_pass(self) -> bool:
        return not self.se2_holonomy.translation.exact_obstruction

    @property
    def passes_all(self) -> bool:
        return self.angular_pass and self.translation_pass


def solve_point_angles(
    case: base.PlacementCase,
    state: base.SolverState,
    mirror_sign_a: Optional[int] = None,
    mirror_sign_b: Optional[int] = None,
) -> angles.AngleSolution:
    return turning.complete_angle_solution(
        case,
        state,
        mirror_sign_a=mirror_sign_a,
        mirror_sign_b=mirror_sign_b,
    )


def check_total_turn(
    case: base.PlacementCase,
    state: base.SolverState,
    mirror_sign_a: Optional[int] = None,
    mirror_sign_b: Optional[int] = None,
    angle_solution: Optional[angles.AngleSolution] = None,
) -> turning.TotalTurnAnalysis:
    return turning.analyze_total_turn(
        case,
        state,
        mirror_sign_a=mirror_sign_a,
        mirror_sign_b=mirror_sign_b,
        angle_solution=angle_solution,
    )


def check_pole_angles(
    case: base.PlacementCase,
    state: base.SolverState,
    angle_solution: angles.AngleSolution,
) -> pole.PoleAngleAnalysis:
    return pole.analyze_pole_angles(case, state, angle_solution=angle_solution)


def check_translation_holonomy(
    case: base.PlacementCase,
    state: base.SolverState,
    angle_solution: angles.AngleSolution,
    total_turn: turning.TotalTurnAnalysis,
    mirror_sign_a: Optional[int] = None,
    mirror_sign_b: Optional[int] = None,
) -> se2.SE2HolonomyAnalysis:
    return se2.analyze_se2_holonomy(
        case,
        state,
        mirror_sign_a=mirror_sign_a,
        mirror_sign_b=mirror_sign_b,
        angle_solution=angle_solution,
        total_turn_analysis=total_turn,
    )


def analyze_terminal_profile(
    case: base.PlacementCase,
    state: base.SolverState,
    mirror_sign_a: Optional[int] = None,
    mirror_sign_b: Optional[int] = None,
) -> ProfileAnalysis:
    """Run the independent stages in dependency order.

    Explicit mirror signs are diagnostic overrides.  Normal callers should omit
    them so that the placement-time contact parity is used.
    """
    angle_solution = solve_point_angles(
        case,
        state,
        mirror_sign_a=mirror_sign_a,
        mirror_sign_b=mirror_sign_b,
    )
    total_turn = check_total_turn(
        case,
        state,
        mirror_sign_a=mirror_sign_a,
        mirror_sign_b=mirror_sign_b,
        angle_solution=angle_solution,
    )
    pole_angles = check_pole_angles(case, state, angle_solution)
    holonomy = check_translation_holonomy(
        case,
        state,
        angle_solution,
        total_turn,
        mirror_sign_a=mirror_sign_a,
        mirror_sign_b=mirror_sign_b,
    )
    return ProfileAnalysis(
        angle_solution=angle_solution,
        total_turn=total_turn,
        pole_angles=pole_angles,
        se2_holonomy=holonomy,
    )
