#!/usr/bin/env python3
"""
Second-pass angle constraint solver for symbolic contour solutions.

The word solver handles curve pieces. This module handles the points between
those pieces. A point carries a signed turning angle tau:

    tau = 0  means that the oriented contour continues straight.

Reversing the path changes tau to -tau. For a non-mirrored copy, an internal
point correspondence therefore induces either tau_p = tau_q or tau_p = -tau_q,
depending on the two path orientations. Endpoints of a mapped interval are not
constrained, because only one incident side of the corner is part of that
interval.

The analysis is performed after a terminal word solution is known. Resolution-
introduced splits are recovered from the terminal expressions of the initial
contour factors, so no contour discretization is introduced in advance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import settings
import symbolic_enumerator as base


SAME = settings.SAME
OPPOSITE = settings.OPPOSITE


@dataclass(frozen=True)
class PointOccurrence:
    point: str
    orientation: int


@dataclass(frozen=True)
class ExpandedPath:
    segments: base.Word
    points: Tuple[PointOccurrence, ...]

    def __post_init__(self) -> None:
        if len(self.points) != len(self.segments) + 1:
            raise ValueError("A path with n segments must have n+1 point occurrences")


@dataclass(frozen=True)
class AngleEquation:
    left_point: str
    right_point: str
    sign: int
    projection: str
    boundary_index: int

    def to_text(self) -> str:
        operator = "=" if self.sign == SAME else "=-"
        return f"turn({self.left_point}) {operator} turn({self.right_point})"


@dataclass(frozen=True)
class AngleAssignment:
    point: str
    expression: str


@dataclass(frozen=True)
class AngleSolution:
    equations: Tuple[AngleEquation, ...]
    assignments: Tuple[AngleAssignment, ...]
    zero_points: Tuple[str, ...]
    parameter_count: int

    def assignment_map(self) -> Dict[str, str]:
        return {item.point: item.expression for item in self.assignments}

    def to_dict(self) -> Dict[str, object]:
        return {
            "convention": {
                "quantity": "signed turning angle",
                "straight": "0",
                "opposite_relation": "tau_left = -tau_right",
                "regularity_assumption": (
                    "turning angles are represented in (-pi, pi); pi U-turns/cusps "
                    "are excluded"
                ),
            },
            "equations": [
                {
                    "left_point": equation.left_point,
                    "right_point": equation.right_point,
                    "sign": "same" if equation.sign == SAME else "opposite",
                    "projection": equation.projection,
                    "boundary_index": equation.boundary_index,
                    "text": equation.to_text(),
                }
                for equation in self.equations
            ],
            "assignments": {
                assignment.point: assignment.expression
                for assignment in self.assignments
            },
            "zero_points": list(self.zero_points),
            "parameter_count": self.parameter_count,
        }


def resolve_mirror_signs(
    case: base.PlacementCase,
    mirror_sign_a: Optional[int] = None,
    mirror_sign_b: Optional[int] = None,
) -> Tuple[int, int]:
    """Use placement-time parities unless an explicit comparison override is given."""
    resolved_a = case.a_mirror_sign if mirror_sign_a is None else mirror_sign_a
    resolved_b = case.b_mirror_sign if mirror_sign_b is None else mirror_sign_b
    if resolved_a not in (SAME, OPPOSITE) or resolved_b not in (SAME, OPPOSITE):
        raise ValueError("Mirror signs must be +1 or -1")
    return resolved_a, resolved_b


class SignedAngleDSU:
    """Union-find for relations angle(x) = sign * angle(y)."""

    def __init__(self, points: Iterable[str]) -> None:
        point_list = sorted(set(points))
        self.index = {point: position for position, point in enumerate(point_list)}
        self.points = point_list
        self.parent = list(range(len(point_list)))
        self.rank = [0] * len(point_list)
        self.parity = [SAME] * len(point_list)
        self.zero_root = [False] * len(point_list)

    def find(self, item: int) -> Tuple[int, int]:
        if self.parent[item] != item:
            parent = self.parent[item]
            root, parent_sign = self.find(parent)
            self.parity[item] *= parent_sign
            self.parent[item] = root
        return self.parent[item], self.parity[item]

    def union(self, left_point: str, right_point: str, sign: int) -> None:
        left = self.index[left_point]
        right = self.index[right_point]
        left_root, left_sign = self.find(left)
        right_root, right_sign = self.find(right)

        # left_angle = left_sign * root_left_angle
        # right_angle = right_sign * root_right_angle
        # Required: left_angle = sign * right_angle.
        if left_root == right_root:
            if left_sign != sign * right_sign:
                self.zero_root[left_root] = True
            return

        root_relation = left_sign * sign * right_sign
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
            self.parity[left_root] = root_relation
            self.zero_root[right_root] = (
                self.zero_root[right_root] or self.zero_root[left_root]
            )
        else:
            self.parent[right_root] = left_root
            self.parity[right_root] = root_relation
            self.zero_root[left_root] = (
                self.zero_root[left_root] or self.zero_root[right_root]
            )
            if self.rank[left_root] == self.rank[right_root]:
                self.rank[left_root] += 1

    def solve(self) -> Tuple[Tuple[AngleAssignment, ...], Tuple[str, ...], int]:
        components: Dict[int, List[Tuple[str, int]]] = {}
        for point, index in self.index.items():
            root, sign = self.find(index)
            components.setdefault(root, []).append((point, sign))

        parameter_by_root: Dict[int, str] = {}
        parameter_index = 0
        assignments: List[AngleAssignment] = []
        zero_points: List[str] = []

        for root in sorted(components, key=lambda value: min(p for p, _ in components[value])):
            members = sorted(components[root])
            if self.zero_root[self.find(root)[0]]:
                for point, _sign in members:
                    assignments.append(AngleAssignment(point, "0"))
                    zero_points.append(point)
                continue

            parameter = f"Theta{parameter_index}"
            parameter_index += 1
            # Normalize the lexicographically first member to +Theta.
            first_sign = members[0][1]
            parameter_by_root[root] = parameter
            for point, sign in members:
                normalized = sign * first_sign
                expression = parameter if normalized == SAME else f"-{parameter}"
                assignments.append(AngleAssignment(point, expression))

        assignments.sort(key=lambda item: item.point)
        zero_points.sort()
        return tuple(assignments), tuple(zero_points), parameter_index


def contour_boundary_points(case: base.PlacementCase) -> Tuple[str, ...]:
    """Return one point identifier per cyclic boundary of the initial factors."""
    segment_count = len(case.cycle_word)
    p1_boundary = len(case.a_word)
    boundary_markers: Dict[int, List[str]] = {}
    for marker, boundary in case.marker_boundaries:
        boundary_markers.setdefault(boundary, []).append(marker)

    points: List[str] = []
    for boundary in range(segment_count):
        if boundary == 0:
            points.append("P0")
        elif boundary == p1_boundary:
            points.append("P1")
        else:
            markers = sorted(boundary_markers.get(boundary, []))
            marker_text = "+".join(markers) if markers else f"boundary_{boundary}"
            points.append(f"S[{marker_text}]")
    return tuple(points)


def initial_segment_endpoints(case: base.PlacementCase) -> Dict[str, Tuple[str, str]]:
    boundaries = contour_boundary_points(case)
    segment_count = len(case.cycle_word)
    endpoints: Dict[str, Tuple[str, str]] = {}
    for index, literal in enumerate(case.cycle_word):
        if literal.inverse:
            raise ValueError("The prototype cycle must use positive factor literals")
        endpoints[literal.variable] = (
            boundaries[index],
            boundaries[(index + 1) % segment_count],
        )
    return endpoints


def positive_segment_path(
    variable: str,
    environment: Mapping[str, base.Word],
    endpoints: Mapping[str, Tuple[str, str]],
) -> ExpandedPath:
    expression = environment[variable]
    start, end = endpoints[variable]
    internal = tuple(
        f"J[{variable}:{index}]"
        for index in range(1, len(expression))
    )
    point_names = (start,) + internal + (end,)
    points = tuple(PointOccurrence(point, base.FORWARD) for point in point_names)
    return ExpandedPath(expression, points)


def literal_path(
    literal: base.Literal,
    environment: Mapping[str, base.Word],
    endpoints: Mapping[str, Tuple[str, str]],
) -> ExpandedPath:
    positive = positive_segment_path(literal.variable, environment, endpoints)
    if not literal.inverse:
        return positive
    return ExpandedPath(
        base.inverse_word(positive.segments),
        tuple(
            PointOccurrence(point.point, base.REVERSE)
            for point in reversed(positive.points)
        ),
    )


def expand_initial_word(
    word: base.Word,
    environment: Mapping[str, base.Word],
    endpoints: Mapping[str, Tuple[str, str]],
) -> ExpandedPath:
    segment_output: List[base.Literal] = []
    point_output: List[PointOccurrence] = []

    for literal_index, literal in enumerate(word):
        path = literal_path(literal, environment, endpoints)
        if literal_index == 0:
            point_output.extend(path.points)
        else:
            if point_output[-1].point != path.points[0].point:
                raise ValueError(
                    "Initial factor word is not a contiguous prototype path: "
                    f"{point_output[-1].point} != {path.points[0].point}"
                )
            # The same geometric boundary can be viewed in the same path direction.
            if point_output[-1].orientation != path.points[0].orientation:
                raise ValueError("Inconsistent traversal orientation at a path joint")
            point_output.extend(path.points[1:])
        segment_output.extend(path.segments)

    return ExpandedPath(tuple(segment_output), tuple(point_output))


def projection_angle_equations(
    case: base.PlacementCase,
    state: base.SolverState,
    mirror_sign_a: Optional[int] = None,
    mirror_sign_b: Optional[int] = None,
) -> Tuple[AngleEquation, ...]:
    """
    Derive angle equations at internal mapped points.

    `mirror_sign_*` is +1 for a direct isometry and -1 for a reflected copy.
    The current word model itself only models path reversal; reflection is kept
    as explicit angle metadata here.
    """
    if state.equations:
        raise ValueError("Angle analysis requires a terminal word-solver state")
    mirror_sign_a, mirror_sign_b = resolve_mirror_signs(
        case, mirror_sign_a, mirror_sign_b
    )

    environment = state.environment_map()
    endpoints = initial_segment_endpoints(case)
    equations: List[AngleEquation] = []

    projection_data = (
        ("A", case.a_word, case.a_target, mirror_sign_a),
        ("B", case.b_word, case.b_target, mirror_sign_b),
    )

    for projection, left_word, right_word, mirror_sign in projection_data:
        left = expand_initial_word(left_word, environment, endpoints)
        right = expand_initial_word(right_word, environment, endpoints)
        if left.segments != right.segments:
            raise ValueError(
                f"Terminal state does not solve projection {projection}: "
                f"{base.word_to_text(left.segments)} != "
                f"{base.word_to_text(right.segments)}"
            )

        # Endpoints are deliberately excluded. A mapped interval contains only
        # one incident contour side at each endpoint, so it does not determine
        # the complete corner angle there.
        for boundary_index in range(1, len(left.segments)):
            left_occurrence = left.points[boundary_index]
            right_occurrence = right.points[boundary_index]
            sign = (
                mirror_sign
                * left_occurrence.orientation
                * right_occurrence.orientation
            )
            equations.append(
                AngleEquation(
                    left_point=left_occurrence.point,
                    right_point=right_occurrence.point,
                    sign=sign,
                    projection=projection,
                    boundary_index=boundary_index,
                )
            )

    return tuple(equations)


def solve_angle_equations(
    equations: Sequence[AngleEquation],
    all_points: Optional[Iterable[str]] = None,
) -> AngleSolution:
    points = set(all_points or ())
    points.update(
        point
        for equation in equations
        for point in (equation.left_point, equation.right_point)
    )
    dsu = SignedAngleDSU(points)
    for equation in equations:
        dsu.union(equation.left_point, equation.right_point, equation.sign)
    assignments, zero_points, parameter_count = dsu.solve()
    return AngleSolution(
        equations=tuple(equations),
        assignments=assignments,
        zero_points=zero_points,
        parameter_count=parameter_count,
    )


def analyze_terminal_state_angles(
    case: base.PlacementCase,
    state: base.SolverState,
    mirror_sign_a: Optional[int] = None,
    mirror_sign_b: Optional[int] = None,
) -> AngleSolution:
    return solve_angle_equations(
        projection_angle_equations(
            case,
            state,
            mirror_sign_a=mirror_sign_a,
            mirror_sign_b=mirror_sign_b,
        )
    )


def state_profile_text(case: base.PlacementCase, state: base.SolverState) -> Tuple[str, str]:
    environment = state.environment_map()
    a: List[base.Literal] = []
    b: List[base.Literal] = []
    for literal in case.a_word:
        a.extend(environment[literal.variable])
    for literal in case.b_word:
        b.extend(environment[literal.variable])
    return base.word_to_text(a), base.word_to_text(b)


def command_case(args: argparse.Namespace) -> int:
    case = base.find_case(args.case_id)
    emitted = 0
    for state, derivation in base.enumerate_terminal_states(
        case,
        max_depth=args.max_depth,
        max_states=args.max_states,
    ):
        angle_solution = analyze_terminal_state_angles(case, state)
        a_text, b_text = state_profile_text(case, state)
        payload = {
            "case_id": case.case_id,
            "derivation": list(derivation),
            "A": a_text,
            "B": b_text,
            "angles": angle_solution.to_dict(),
        }
        print(json.dumps(payload, ensure_ascii=True))
        emitted += 1
        if args.max_solutions is not None and emitted >= args.max_solutions:
            break
    print(f"Emitted angle analyses: {emitted}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Second-pass angle constraints for terminal contour word solutions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    case_parser = subparsers.add_parser("case")
    case_parser.add_argument("case_id", type=int)
    case_parser.add_argument("--max-depth", type=int)
    case_parser.add_argument("--max-states", type=int)
    case_parser.add_argument("--max-solutions", type=int)
    case_parser.set_defaults(handler=command_case)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
