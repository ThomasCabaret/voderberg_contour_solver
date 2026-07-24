#!/usr/bin/env python3
"""User-facing formatter for complete formal contour profiles.

The word solver resolves curve pieces, while the angle solver resolves the
turning angles carried by the points between those pieces.  This module merges
both layers into one cyclic contour sentence such as::

    (P0 = a0) V0 (-a1) V1 (a2 = 0) V1^-1 (a1) V0^-1 ...

Angle names ``a0``, ``a1``, ... are display aliases.  They are assigned in
first-occurrence contour order and do not replace the internal ``Theta`` names
used by the mathematical filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import angle_constraints as angles
import symbolic_enumerator as base


FORMAT_SCHEMA_VERSION = "formal-profile-v1"
ANGLE_ALIAS_PREFIX = "a"
NAMED_POINTS = frozenset(("P0", "P1"))


@dataclass(frozen=True)
class DisplayPoint:
    point: str
    angle_class: str
    expression: str
    internal_expression: str
    fixed_zero: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "point": self.point,
            "angle_class": self.angle_class,
            "expression": self.expression,
            "internal_expression": self.internal_expression,
            "fixed_zero": self.fixed_zero,
        }


@dataclass(frozen=True)
class DisplayAngleClass:
    alias: str
    fixed_zero: bool
    members: Tuple[Tuple[str, int], ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "alias": self.alias,
            "fixed_zero": self.fixed_zero,
            "members": [
                {"point": point, "sign": sign}
                for point, sign in self.members
            ],
        }


@dataclass(frozen=True)
class FormalContourProfile:
    text: str
    word_contour: str
    point_occurrences: Tuple[DisplayPoint, ...]
    angle_classes: Tuple[DisplayAngleClass, ...]
    curve_parameters: Tuple[str, ...]
    free_angle_parameters: Tuple[str, ...]
    fixed_zero_angle_classes: Tuple[str, ...]

    @property
    def curve_parameter_count(self) -> int:
        return len(self.curve_parameters)

    @property
    def angle_parameter_count(self) -> int:
        return len(self.free_angle_parameters)

    @property
    def total_parameter_count(self) -> int:
        return self.curve_parameter_count + self.angle_parameter_count

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "word_contour": self.word_contour,
            "point_occurrences": [point.to_dict() for point in self.point_occurrences],
            "angle_classes": [item.to_dict() for item in self.angle_classes],
            "curve_parameters": list(self.curve_parameters),
            "free_angle_parameters": list(self.free_angle_parameters),
            "fixed_zero_angle_classes": list(self.fixed_zero_angle_classes),
            "curve_parameter_count": self.curve_parameter_count,
            "angle_parameter_count": self.angle_parameter_count,
            "total_parameter_count": self.total_parameter_count,
        }


class _SimpleDSU:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _curve_parameters(segments: Sequence[base.Literal]) -> Tuple[str, ...]:
    seen = set()
    output: List[str] = []
    for literal in segments:
        if literal.variable not in seen:
            seen.add(literal.variable)
            output.append(literal.variable)
    return tuple(output)


def _zero_component_keys(solution: angles.AngleSolution) -> Dict[str, str]:
    """Recover separate zero classes from the equation graph.

    ``AngleSolution`` intentionally stores every forced member as the value
    ``0``.  For display, this helper preserves which zero points were connected
    by angle equations, so one class can be printed with one stable alias.
    """
    zero_points = set(solution.zero_points)
    dsu = _SimpleDSU(zero_points)
    for equation in solution.equations:
        if equation.left_point in zero_points and equation.right_point in zero_points:
            dsu.union(equation.left_point, equation.right_point)
    return {point: f"zero:{dsu.find(point)}" for point in zero_points}


def _class_key(
    point: str,
    internal_expression: str,
    zero_keys: Mapping[str, str],
) -> str:
    if internal_expression == "0":
        return zero_keys.get(point, f"zero:{point}")
    return f"free:{internal_expression.removeprefix('-')}"


def _sign_of_expression(expression: str) -> int:
    if expression == "0":
        return 0
    return -1 if expression.startswith("-") else 1


def _format_point_token(point: DisplayPoint) -> str:
    if point.fixed_zero:
        value = f"{point.angle_class} = 0"
    else:
        value = point.expression
    if point.point in NAMED_POINTS:
        return f"({point.point} = {value})"
    return f"({value})"


def build_formal_profile(
    case: base.PlacementCase,
    state: base.SolverState,
    angle_solution: angles.AngleSolution,
) -> FormalContourProfile:
    """Merge terminal curve words and solved point-angle classes.

    The returned sentence starts at P0, follows the conventional positive
    contour direction, and stops after the final segment immediately before
    returning to P0.  Therefore the closing P0 is not printed twice.
    """
    if state.equations:
        raise ValueError("Formal profile formatting requires a terminal solver state")

    environment = state.environment_map()
    endpoints = angles.initial_segment_endpoints(case)
    expanded = angles.expand_initial_word(case.cycle_word, environment, endpoints)
    assignments = angle_solution.assignment_map()
    zero_keys = _zero_component_keys(angle_solution)

    contour_points = tuple(occurrence.point for occurrence in expanded.points[:-1])
    missing = [point for point in contour_points if point not in assignments]
    if missing:
        raise ValueError(
            "The angle solution is incomplete for contour display: " + ", ".join(missing)
        )

    alias_by_key: Dict[str, str] = {}
    class_members: Dict[str, List[Tuple[str, int]]] = {}
    display_points: List[DisplayPoint] = []

    for point in contour_points:
        internal_expression = assignments[point]
        key = _class_key(point, internal_expression, zero_keys)
        if key not in alias_by_key:
            alias_by_key[key] = f"{ANGLE_ALIAS_PREFIX}{len(alias_by_key)}"
        alias = alias_by_key[key]
        sign = _sign_of_expression(internal_expression)
        expression = alias if sign >= 0 else f"-{alias}"
        fixed_zero = internal_expression == "0"
        display_points.append(
            DisplayPoint(
                point=point,
                angle_class=alias,
                expression=expression,
                internal_expression=internal_expression,
                fixed_zero=fixed_zero,
            )
        )
        class_members.setdefault(key, []).append((point, sign))

    tokens: List[str] = []
    for point, segment in zip(display_points, expanded.segments):
        tokens.append(_format_point_token(point))
        tokens.append(segment.to_text())

    angle_classes: List[DisplayAngleClass] = []
    for key, alias in alias_by_key.items():
        members = tuple(class_members[key])
        angle_classes.append(
            DisplayAngleClass(
                alias=alias,
                fixed_zero=key.startswith("zero:"),
                members=members,
            )
        )

    free_angles = tuple(item.alias for item in angle_classes if not item.fixed_zero)
    zero_angles = tuple(item.alias for item in angle_classes if item.fixed_zero)
    # Keep a secondary word-only contour for debugging and compatibility.
    a_text, b_text = angles.state_profile_text(case, state)
    word_contour = f"(P0) {a_text} (P1) {b_text}"

    return FormalContourProfile(
        text=" ".join(tokens),
        word_contour=word_contour,
        point_occurrences=tuple(display_points),
        angle_classes=tuple(angle_classes),
        curve_parameters=_curve_parameters(expanded.segments),
        free_angle_parameters=free_angles,
        fixed_zero_angle_classes=zero_angles,
    )
