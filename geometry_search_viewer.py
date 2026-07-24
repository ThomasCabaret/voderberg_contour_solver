#!/usr/bin/env python3
"""Heuristic polygonal search and Tk viewer for surviving formal profiles.

The search is deliberately separate from the exact filtering pipeline. Each
formal curve variable is represented by a polyline with a configurable number
of intermediate points. Repeated and inverse occurrences reuse the same
template, point-angle classes share the same turn parameter, and
projection-derived Kappa classes share the same total curve turn. Differential
evolution searches for a closed, positively oriented, non-self-intersecting
prototype contour.

A found candidate is a concrete certificate for the configured polygonal model.
Failure to find one is not a proof of impossibility.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

import geometry_checkpoint
import settings


Point = Tuple[float, float]


@dataclass(frozen=True)
class SegmentOccurrence:
    variable: str
    inverse: bool

    @property
    def text(self) -> str:
        return self.variable + ("^-1" if self.inverse else "")


@dataclass(frozen=True)
class SearchProfile:
    profile_id: int
    case_id: int
    formal_text: str
    occurrences: Tuple[SegmentOccurrence, ...]
    point_expressions: Tuple[str, ...]
    free_angles: Tuple[str, ...]
    curve_variables: Tuple[str, ...]
    kappa_assignments: Mapping[str, str]


@dataclass
class CandidateGeometry:
    profile_id: int
    case_id: int
    formal_text: str
    objective: float
    closure_error: float
    turn_error: float
    signed_area: float
    vertices: List[Point]
    edge_metadata: List[Dict[str, object]]
    angle_values: Dict[str, float]
    kappa_class_values: Dict[str, float]
    intermediate_points_per_variable: int
    curve_lengths: Dict[str, Tuple[float, ...]]
    curve_internal_turns: Dict[str, Tuple[float, ...]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "case_id": self.case_id,
            "formal_profile": self.formal_text,
            "objective": self.objective,
            "closure_error": self.closure_error,
            "turn_error": self.turn_error,
            "signed_area": self.signed_area,
            "vertices": [[x, y] for x, y in self.vertices],
            "edge_metadata": self.edge_metadata,
            "angle_values": self.angle_values,
            "kappa_class_values": self.kappa_class_values,
            "intermediate_points_per_variable": self.intermediate_points_per_variable,
            "curve_lengths": {
                name: list(values) for name, values in self.curve_lengths.items()
            },
            "curve_internal_turns": {
                name: list(values) for name, values in self.curve_internal_turns.items()
            },
            "model": {
                "curve_template": (
                    f"polyline with {self.intermediate_points_per_variable} intermediate "
                    f"point(s) per formal variable"
                ),
                "status": "heuristic geometric candidate",
                "limitations": (
                    "This certifies only the displayed prototype contour in the "
                    "restricted template model. It does not yet certify the full "
                    "three-copy non-overlap realization."
                ),
            },
        }


def _parse_occurrences(word_contour: str) -> Tuple[SegmentOccurrence, ...]:
    output: List[SegmentOccurrence] = []
    for token in word_contour.split():
        if token.startswith("("):
            continue
        inverse = token.endswith("^-1")
        variable = token[:-3] if inverse else token
        output.append(SegmentOccurrence(variable, inverse))
    return tuple(output)


def _extract_profile(record: Mapping[str, object]) -> SearchProfile:
    solution = record["solution"]
    formal = solution["formal_profile"]
    occurrences = _parse_occurrences(solution["word_contour"])
    points = formal["point_occurrences"]
    if len(points) != len(occurrences):
        raise ValueError(
            f"Profile {record.get('profile_id')} has {len(points)} points for "
            f"{len(occurrences)} segment occurrences"
        )
    external = record.get("experimental", {}).get("external_boundary") or {}
    turn_solution = external.get("projection_curve_turn_constraints") or {}
    assignments = turn_solution.get("assignments") or {}
    curve_variables = tuple(formal["curve_parameters"])
    normalized_assignments = {
        variable: str(assignments.get(variable, f"Kappa[{variable}]"))
        for variable in curve_variables
    }
    return SearchProfile(
        profile_id=int(record["profile_id"]),
        case_id=int(record["case_id"]),
        formal_text=str(solution["profile"]),
        occurrences=occurrences,
        point_expressions=tuple(str(item["expression"]) if not item["fixed_zero"] else "0" for item in points),
        free_angles=tuple(str(item) for item in formal["free_angle_parameters"]),
        curve_variables=curve_variables,
        kappa_assignments=normalized_assignments,
    )


def _iter_top_level_array(
    path: Path,
    key: str,
    *,
    chunk_size: int = 1024 * 1024,
) -> Iterable[Mapping[str, object]]:
    """Incrementally decode objects from a named top-level JSON array.

    The audit files are standard JSON objects, not JSON Lines.  This reader
    keeps only a small text buffer and one array item in memory at a time.
    """
    decoder = json.JSONDecoder()
    key_token = json.dumps(key)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        array_started = False
        eof = False

        while not array_started:
            chunk = handle.read(chunk_size)
            if not chunk:
                eof = True
            buffer += chunk
            key_index = buffer.find(key_token)
            if key_index >= 0:
                colon_index = buffer.find(":", key_index + len(key_token))
                array_index = buffer.find("[", colon_index + 1) if colon_index >= 0 else -1
                if array_index >= 0:
                    buffer = buffer[array_index + 1 :]
                    array_started = True
                    break
            if eof:
                raise ValueError(f"Missing top-level JSON array {key!r} in {path}")
            # Retain enough overlap to catch a key split across chunks.
            if len(buffer) > len(key_token) + 128:
                buffer = buffer[-(len(key_token) + 128) :]

        position = 0
        while True:
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n,":
                    position += 1
                if position < len(buffer):
                    break
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(f"Unterminated top-level array {key!r} in {path}")
                buffer = chunk
                position = 0

            if buffer[position] == "]":
                return

            while True:
                try:
                    item, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        raise ValueError(f"Invalid or truncated JSON item in {path}")
                    if position:
                        buffer = buffer[position:] + chunk
                        position = 0
                    else:
                        buffer += chunk
            if not isinstance(item, Mapping):
                raise ValueError(f"Expected an object in {key!r}, got {type(item).__name__}")
            yield item
            buffer = buffer[end:]
            position = 0


def load_survivors(path: Path) -> List[SearchProfile]:
    return [_extract_profile(record) for record in _iter_top_level_array(path, "profiles")]


def _signed_name(expression: str) -> Tuple[int, Optional[str]]:
    if expression == "0":
        return 0, None
    if expression.startswith("-"):
        return -1, expression[1:]
    return 1, expression


def _kappa_classes(profile: SearchProfile) -> Tuple[str, ...]:
    classes: List[str] = []
    seen = set()
    for expression in profile.kappa_assignments.values():
        sign, name = _signed_name(expression)
        if sign != 0 and name is not None and name not in seen:
            seen.add(name)
            classes.append(name)
    return tuple(classes)


def _edge_count(intermediate_points: int) -> int:
    if intermediate_points < 0:
        raise ValueError("intermediate_points must be nonnegative")
    return intermediate_points + 1


def _free_internal_turn_count(intermediate_points: int) -> int:
    # A polyline with E edges has E-1 internal turns. The final one is derived
    # from Kappa so the total curve turn is exact. With one edge, Kappa must be 0.
    return max(0, _edge_count(intermediate_points) - 2)


def _decode_parameters(
    profile: SearchProfile,
    values: Sequence[float],
    intermediate_points: int,
) -> Tuple[
    Dict[str, Tuple[float, ...]],
    Dict[str, Tuple[float, ...]],
    Dict[str, float],
    Dict[str, float],
]:
    cursor = 0
    edge_count = _edge_count(intermediate_points)
    free_turn_count = _free_internal_turn_count(intermediate_points)
    lengths: Dict[str, Tuple[float, ...]] = {}
    shape_turns: Dict[str, Tuple[float, ...]] = {}
    for variable in profile.curve_variables:
        lengths[variable] = tuple(
            float(values[cursor + offset]) for offset in range(edge_count)
        )
        cursor += edge_count
        shape_turns[variable] = tuple(
            float(values[cursor + offset]) for offset in range(free_turn_count)
        )
        cursor += free_turn_count
    angles = {
        name: float(values[cursor + index])
        for index, name in enumerate(profile.free_angles)
    }
    cursor += len(profile.free_angles)
    kappa_classes = _kappa_classes(profile)
    if edge_count == 1:
        kappas = {name: 0.0 for name in kappa_classes}
    else:
        kappas = {
            name: float(values[cursor + index])
            for index, name in enumerate(kappa_classes)
        }
    return lengths, shape_turns, angles, kappas


def _bounds(profile: SearchProfile, intermediate_points: int) -> List[Tuple[float, float]]:
    bounds: List[Tuple[float, float]] = []
    edge_count = _edge_count(intermediate_points)
    free_turn_count = _free_internal_turn_count(intermediate_points)
    angular_bound = math.pi - settings.GEOMETRY_ANGLE_MARGIN
    for _variable in profile.curve_variables:
        bounds.extend(
            [(settings.GEOMETRY_LENGTH_MIN, settings.GEOMETRY_LENGTH_MAX)]
            * edge_count
        )
        bounds.extend([(-angular_bound, angular_bound)] * free_turn_count)
    bounds.extend([(-angular_bound, angular_bound)] * len(profile.free_angles))
    if edge_count > 1:
        bounds.extend([(-angular_bound, angular_bound)] * len(_kappa_classes(profile)))
    return bounds


def _expression_value(expression: str, values: Mapping[str, float]) -> float:
    sign, name = _signed_name(expression)
    if sign == 0 or name is None:
        return 0.0
    return sign * values[name]


def _curve_kappa(profile: SearchProfile, variable: str, class_values: Mapping[str, float]) -> float:
    return _expression_value(profile.kappa_assignments[variable], class_values)


def _rotate(vector: Point, angle: float) -> Point:
    c = math.cos(angle)
    s = math.sin(angle)
    return c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]


def _positive_internal_turns(
    kappa: float,
    free_turns: Sequence[float],
    edge_count: int,
) -> Tuple[float, ...]:
    if edge_count == 1:
        return ()
    return tuple(free_turns) + (kappa - sum(free_turns),)


def _oriented_template(
    lengths: Sequence[float],
    internal_turns: Sequence[float],
    inverse: bool,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if not inverse:
        return tuple(lengths), tuple(internal_turns)
    return tuple(reversed(lengths)), tuple(-value for value in reversed(internal_turns))


def _simulate(
    profile: SearchProfile,
    values: Sequence[float],
    intermediate_points: int,
) -> Tuple[
    List[Point],
    List[Dict[str, object]],
    float,
    Dict[str, float],
    Dict[str, float],
    Dict[str, Tuple[float, ...]],
    Dict[str, Tuple[float, ...]],
]:
    lengths, shape_turns, angle_values, kappa_values = _decode_parameters(
        profile, values, intermediate_points
    )
    edge_count = _edge_count(intermediate_points)
    positive_turns = {
        variable: _positive_internal_turns(
            _curve_kappa(profile, variable, kappa_values),
            shape_turns[variable],
            edge_count,
        )
        for variable in profile.curve_variables
    }
    vertices: List[Point] = [(0.0, 0.0)]
    metadata: List[Dict[str, object]] = []
    position = (0.0, 0.0)
    heading = 0.0
    count = len(profile.occurrences)

    for occurrence_index, occurrence in enumerate(profile.occurrences):
        occurrence_lengths, occurrence_turns = _oriented_template(
            lengths[occurrence.variable],
            positive_turns[occurrence.variable],
            occurrence.inverse,
        )
        first_vertex_index = len(vertices) - 1
        local_heading = 0.0
        for subedge_index, length in enumerate(occurrence_lengths):
            dx, dy = _rotate((length, 0.0), heading + local_heading)
            next_position = (position[0] + dx, position[1] + dy)
            vertices.append(next_position)
            metadata.append(
                {
                    "variable": occurrence.variable,
                    "inverse": occurrence.inverse,
                    "occurrence": occurrence_index,
                    "subedge": subedge_index,
                    "formal_token": occurrence.text,
                    "start_vertex": len(vertices) - 2,
                    "end_vertex": len(vertices) - 1,
                }
            )
            position = next_position
            if subedge_index < len(occurrence_turns):
                local_heading += occurrence_turns[subedge_index]

        curve_turn = sum(occurrence_turns)
        heading += curve_turn
        next_point_expression = profile.point_expressions[(occurrence_index + 1) % count]
        heading += _expression_value(next_point_expression, angle_values)
        metadata[-1]["formal_occurrence_start_vertex"] = first_vertex_index
        metadata[-1]["formal_occurrence_end_vertex"] = len(vertices) - 1

    return (
        vertices,
        metadata,
        heading,
        angle_values,
        kappa_values,
        lengths,
        positive_turns,
    )


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point, eps: float = 1e-9) -> bool:
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        and abs(_orientation(a, b, p)) <= eps
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point, eps: float = 1e-9) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    return any(
        (
            abs(value) <= eps
            and _on_segment(start, end, point, eps)
        )
        for value, start, end, point in (
            (o1, a, b, c),
            (o2, a, b, d),
            (o3, c, d, a),
            (o4, c, d, b),
        )
    )


def _intersection_count(vertices: Sequence[Point]) -> int:
    edge_count = len(vertices) - 1
    count = 0
    for i in range(edge_count):
        a, b = vertices[i], vertices[i + 1]
        for j in range(i + 1, edge_count):
            if j == i + 1:
                continue
            if i == 0 and j == edge_count - 1:
                continue
            c, d = vertices[j], vertices[j + 1]
            if _segments_intersect(a, b, c, d):
                count += 1
    return count


def _signed_area(vertices: Sequence[Point]) -> float:
    points = list(vertices)
    if points[-1] != points[0]:
        points.append(points[0])
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:])
    )


def _metrics(
    profile: SearchProfile,
    values: Sequence[float],
    intermediate_points: int,
) -> Tuple[
    float,
    float,
    float,
    int,
    List[Point],
    List[Dict[str, object]],
    Dict[str, float],
    Dict[str, float],
    Dict[str, Tuple[float, ...]],
    Dict[str, Tuple[float, ...]],
]:
    (
        vertices,
        metadata,
        final_heading,
        angles,
        kappas,
        lengths,
        internal_turns,
    ) = _simulate(profile, values, intermediate_points)
    closure = math.hypot(
        vertices[-1][0] - vertices[0][0],
        vertices[-1][1] - vertices[0][1],
    )
    turn_error = abs(final_heading - 2.0 * math.pi)
    area = _signed_area(vertices)
    intersections = _intersection_count(vertices)
    return (
        closure,
        turn_error,
        area,
        intersections,
        vertices,
        metadata,
        angles,
        kappas,
        lengths,
        internal_turns,
    )


def _objective(
    profile: SearchProfile,
    values: Sequence[float],
    intermediate_points: int,
) -> float:
    closure, turn_error, area, intersections, *_rest = _metrics(
        profile, values, intermediate_points
    )
    area_deficit = max(0.0, settings.GEOMETRY_MIN_ABS_AREA - area)
    return (
        settings.GEOMETRY_CLOSURE_WEIGHT * closure * closure
        + settings.GEOMETRY_TURN_WEIGHT * turn_error * turn_error
        + settings.GEOMETRY_AREA_WEIGHT * area_deficit * area_deficit
        + settings.GEOMETRY_INTERSECTION_PENALTY * intersections
    )


def _valid_candidate(
    profile: SearchProfile,
    values: Sequence[float],
    objective: float,
    intermediate_points: int,
) -> Optional[CandidateGeometry]:
    (
        closure,
        turn_error,
        area,
        intersections,
        vertices,
        metadata,
        angles,
        kappas,
        lengths,
        internal_turns,
    ) = _metrics(profile, values, intermediate_points)
    if closure > settings.GEOMETRY_CLOSURE_TOLERANCE:
        return None
    if turn_error > settings.GEOMETRY_TURN_TOLERANCE:
        return None
    if area <= settings.GEOMETRY_MIN_ABS_AREA:
        return None
    if intersections:
        return None
    vertices[-1] = vertices[0]
    return CandidateGeometry(
        profile_id=profile.profile_id,
        case_id=profile.case_id,
        formal_text=profile.formal_text,
        objective=float(objective),
        closure_error=closure,
        turn_error=turn_error,
        signed_area=area,
        vertices=vertices,
        edge_metadata=metadata,
        angle_values=angles,
        kappa_class_values=kappas,
        intermediate_points_per_variable=intermediate_points,
        curve_lengths=lengths,
        curve_internal_turns=internal_turns,
    )


def search_profile(
    profile: SearchProfile,
    *,
    intermediate_points: int,
    attempts: int,
    max_iterations: int,
    population_size: int,
    seed: int,
) -> Optional[CandidateGeometry]:
    bounds = _bounds(profile, intermediate_points)
    if not bounds:
        return None
    best: Optional[CandidateGeometry] = None
    for attempt in range(attempts):
        result = differential_evolution(
            lambda values: _objective(profile, values, intermediate_points),
            bounds,
            maxiter=max_iterations,
            popsize=population_size,
            seed=seed + 1009 * profile.profile_id + attempt,
            polish=True,
            workers=1,
            updating="immediate",
            disp=False,
        )
        candidate = _valid_candidate(
            profile,
            result.x,
            float(result.fun),
            intermediate_points,
        )
        if candidate is not None and (
            best is None or candidate.objective < best.objective
        ):
            best = candidate
            break
    return best


def select_profiles(
    profiles: Sequence[SearchProfile],
    *,
    max_profiles: int,
) -> List[SearchProfile]:
    selected = list(profiles)
    selected.sort(
        key=lambda profile: (
            len(profile.curve_variables)
            + len(profile.free_angles)
            + len(_kappa_classes(profile)),
            len(profile.occurrences),
            profile.profile_id,
        )
    )
    if max_profiles > 0:
        selected = selected[:max_profiles]
    return selected


def search_profiles(
    profiles: Sequence[SearchProfile],
    *,
    max_profiles: int,
    intermediate_points: int,
    attempts: int,
    max_iterations: int,
    population_size: int,
    seed: int,
) -> List[CandidateGeometry]:
    selected = select_profiles(profiles, max_profiles=max_profiles)
    candidates: List[CandidateGeometry] = []
    for index, profile in enumerate(selected, start=1):
        print(
            f"[geometry {index}/{len(selected)}] profile {profile.profile_id}, "
            f"case {profile.case_id}: searching with {intermediate_points} "
            f"intermediate point(s) per variable...",
            flush=True,
        )
        candidate = search_profile(
            profile,
            intermediate_points=intermediate_points,
            attempts=attempts,
            max_iterations=max_iterations,
            population_size=population_size,
            seed=seed,
        )
        if candidate is None:
            print("      no candidate found in the configured search", flush=True)
        else:
            candidates.append(candidate)
            print(
                f"      found: closure={candidate.closure_error:.3g}, "
                f"turn={candidate.turn_error:.3g}, area={candidate.signed_area:.3g}",
                flush=True,
            )
    return candidates


def _candidate_record(candidate: object) -> Dict[str, object]:
    if isinstance(candidate, CandidateGeometry):
        return candidate.to_dict()
    if isinstance(candidate, Mapping):
        return dict(candidate)
    raise TypeError(f"Unsupported candidate type: {type(candidate).__name__}")


def write_candidates(
    path: Path,
    input_path: Path,
    candidates: Sequence[object],
    intermediate_points: int,
    *,
    search_summary: Optional[Mapping[str, object]] = None,
    checkpoint_path: Optional[Path] = None,
) -> None:
    payload = {
        "metadata": {
            "source_survivors": str(input_path),
            "candidate_count": len(candidates),
            "intermediate_points_per_variable": intermediate_points,
            "edges_per_variable": _edge_count(intermediate_points),
            "method": (
                "heuristic configurable polyline templates with differential evolution"
            ),
            "proof_status": (
                "found candidates are concrete for the configured polygonal model; "
                "failures are inconclusive"
            ),
            "transactional_checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "search_summary": dict(search_summary or {}),
        },
        "candidates": [_candidate_record(candidate) for candidate in candidates],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    temporary.replace(path)


PALETTE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


def _load_candidate_file(path: Path) -> List[Mapping[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("candidates", []))


def launch_viewer(candidates: Sequence[Mapping[str, object]]) -> None:
    if not candidates:
        print("No geometric candidates to display.")
        return
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise RuntimeError("Tkinter is required for the interactive viewer") from exc

    root = tk.Tk()
    root.title("Voderberg geometric candidates")
    root.geometry(f"{settings.GEOMETRY_VIEW_WIDTH}x{settings.GEOMETRY_VIEW_HEIGHT}")
    root.minsize(800, 560)

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    title_var = tk.StringVar()
    ttk.Label(top, textvariable=title_var, font=("Segoe UI", 11, "bold"), wraplength=900).pack(fill="x")

    body = ttk.Frame(root, padding=(8, 0, 8, 8))
    body.pack(fill="both", expand=True)
    canvas = tk.Canvas(body, background="white", highlightthickness=1, highlightbackground="#888")
    canvas.pack(side="left", fill="both", expand=True)
    side = ttk.Frame(body, padding=(10, 0, 0, 0), width=250)
    side.pack(side="right", fill="y")
    info = tk.Text(side, width=34, height=32, wrap="word", state="disabled")
    info.pack(fill="both", expand=True)

    controls = ttk.Frame(root, padding=8)
    controls.pack(fill="x")
    index_var = tk.StringVar()
    current = {"index": 0}

    def render() -> None:
        candidate = candidates[current["index"]]
        vertices = [tuple(item) for item in candidate["vertices"]]
        metadata = candidate["edge_metadata"]
        variables = sorted({item["variable"] for item in metadata})
        color_by_variable = {
            variable: PALETTE[index % len(PALETTE)]
            for index, variable in enumerate(variables)
        }
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        xs = [point[0] for point in vertices]
        ys = [point[1] for point in vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)
        margin = 45
        scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)

        def screen(point: Point) -> Point:
            return (
                margin + (point[0] - min_x) * scale,
                height - margin - (point[1] - min_y) * scale,
            )

        occurrence_points: Dict[int, List[Point]] = {}
        for item in metadata:
            start = screen(vertices[int(item["start_vertex"])])
            end = screen(vertices[int(item["end_vertex"])])
            canvas.create_line(
                start[0], start[1], end[0], end[1],
                fill=color_by_variable[item["variable"]],
                width=3,
                dash=(8, 4) if item["inverse"] else None,
            )
            occurrence_points.setdefault(int(item["occurrence"]), []).extend((start, end))

        seen_occurrences = set()
        for item in metadata:
            occurrence = int(item["occurrence"])
            if occurrence in seen_occurrences:
                continue
            seen_occurrences.add(occurrence)
            points = occurrence_points[occurrence]
            x = sum(point[0] for point in points) / len(points)
            y = sum(point[1] for point in points) / len(points)
            canvas.create_text(x, y, text=item["formal_token"], fill="#111", font=("Consolas", 9, "bold"))

        for point in vertices[:-1]:
            x, y = screen(point)
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#111", outline="")

        legend_y = 18
        for variable in variables:
            canvas.create_line(15, legend_y, 42, legend_y, fill=color_by_variable[variable], width=4)
            canvas.create_text(48, legend_y, text=variable, anchor="w", fill="#111", font=("Consolas", 9))
            legend_y += 18

        title_var.set(
            f"Profile {candidate['profile_id']} / case {candidate['case_id']}\n"
            f"{candidate['formal_profile']}"
        )
        index_var.set(f"{current['index'] + 1} / {len(candidates)}")
        text = (
            f"Objective: {candidate['objective']:.6g}\n"
            f"Closure error: {candidate['closure_error']:.6g}\n"
            f"Turn error: {candidate['turn_error']:.6g}\n"
            f"Signed area: {candidate['signed_area']:.6g}\n"
            f"Intermediate points per variable: "
            f"{candidate.get('intermediate_points_per_variable', 1)}\n\n"
            "Solid = forward occurrence\n"
            "Dashed = inverse occurrence\n\n"
            "Angle values (radians):\n"
            + "\n".join(f"  {key}: {value:.5f}" for key, value in candidate["angle_values"].items())
            + "\n\nKappa classes (radians):\n"
            + "\n".join(f"  {key}: {value:.5f}" for key, value in candidate["kappa_class_values"].items())
        )
        info.configure(state="normal")
        info.delete("1.0", "end")
        info.insert("1.0", text)
        info.configure(state="disabled")

    def move(delta: int) -> None:
        current["index"] = (current["index"] + delta) % len(candidates)
        render()

    ttk.Button(controls, text="Previous", command=lambda: move(-1)).pack(side="left")
    ttk.Label(controls, textvariable=index_var, padding=10).pack(side="left")
    ttk.Button(controls, text="Next", command=lambda: move(1)).pack(side="left")
    ttk.Button(controls, text="Close", command=root.destroy).pack(side="right")
    root.bind("<Left>", lambda _event: move(-1))
    root.bind("<Right>", lambda _event: move(1))
    canvas.bind("<Configure>", lambda _event: render())
    root.after(100, render)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search and display heuristic polygonal contour candidates.")
    parser.add_argument("--input", type=Path, default=Path(settings.AUDIT_SURVIVORS_FILENAME))
    parser.add_argument("--output", type=Path, default=Path(settings.GEOMETRY_CANDIDATES_FILENAME))
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=settings.GEOMETRY_DEFAULT_MAX_PROFILES,
        help="Maximum number of survivor profiles to search; 0 means all profiles (default).",
    )
    parser.add_argument(
        "--intermediate-points",
        type=int,
        default=settings.GEOMETRY_DEFAULT_INTERMEDIATE_POINTS,
        help=(
            "Number of interior polyline vertices used for every formal curve "
            "variable. 0 means one straight edge, 1 means two edges, etc."
        ),
    )
    parser.add_argument("--attempts", type=int, default=settings.GEOMETRY_DEFAULT_ATTEMPTS_PER_PROFILE)
    parser.add_argument("--max-iterations", type=int, default=settings.GEOMETRY_DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--population-size", type=int, default=settings.GEOMETRY_DEFAULT_POPULATION_SIZE)
    parser.add_argument("--seed", type=int, default=settings.GEOMETRY_DEFAULT_SEED)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(settings.GEOMETRY_CHECKPOINT_FILENAME),
        help="SQLite checkpoint written transactionally after every completed profile.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume a matching checkpoint run (default).",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable checkpointing and use the former one-shot behavior.",
    )
    parser.set_defaults(resume=settings.GEOMETRY_DEFAULT_RESUME)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear results for the matching checkpoint configuration before starting.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry profiles previously recorded as errors; found/no-candidate rows remain resumed.",
    )
    parser.add_argument("--view-only", action="store_true", help="Open an existing candidate JSON without searching")
    parser.add_argument("--no-gui", action="store_true", help="Search and write candidates without opening the window")
    return parser


def _checkpoint_configuration(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "intermediate_points": int(args.intermediate_points),
        "attempts": max(1, int(args.attempts)),
        "max_iterations": max(1, int(args.max_iterations)),
        "population_size": max(2, int(args.population_size)),
        "seed": int(args.seed),
        "max_profiles": int(args.max_profiles),
    }


def _summary_dict(summary: geometry_checkpoint.ResumeSummary, *, complete: bool) -> Dict[str, object]:
    return {
        "run_key": summary.run_key,
        "selected_profile_count": summary.selected_count,
        "completed_profile_count": summary.completed_count,
        "remaining_profile_count": summary.remaining_count,
        "found_count": summary.found_count,
        "no_candidate_count": summary.no_candidate_count,
        "error_count": summary.error_count,
        "complete": complete,
    }


def _run_transactional_search(
    args: argparse.Namespace,
    selected: Sequence[SearchProfile],
) -> Tuple[List[Dict[str, object]], Dict[str, object], int]:
    input_sha256 = geometry_checkpoint.sha256_file(args.input)
    configuration = _checkpoint_configuration(args)
    run_key, identity = geometry_checkpoint.build_run_identity(
        input_path=args.input,
        input_sha256=input_sha256,
        selected_profiles=selected,
        configuration=configuration,
    )

    interrupted = False
    with geometry_checkpoint.GeometryCheckpoint(args.checkpoint) as checkpoint:
        summary = checkpoint.prepare_run(
            run_key=run_key,
            identity=identity,
            selected_count=len(selected),
            fresh=args.fresh,
        )
        if args.retry_errors:
            cleared_errors = checkpoint.clear_error_results(summary.run_id)
            if cleared_errors:
                print(f"Retry mode cleared {cleared_errors} prior error row(s).", flush=True)
        completed_ids = checkpoint.completed_profile_ids(
            summary.run_id, retry_errors=False
        )
        print(
            f"Transactional geometry run {run_key[:12]}: "
            f"{len(completed_ids)}/{len(selected)} profiles already durable in {args.checkpoint}",
            flush=True,
        )
        if args.fresh:
            print("Fresh mode cleared the matching checkpoint run.", flush=True)

        try:
            for ordinal, profile in enumerate(selected, start=1):
                if profile.profile_id in completed_ids:
                    continue
                print(
                    f"[geometry {ordinal}/{len(selected)}] profile {profile.profile_id}, "
                    f"case {profile.case_id}: searching with {args.intermediate_points} "
                    f"intermediate point(s) per variable...",
                    flush=True,
                )
                try:
                    candidate = search_profile(
                        profile,
                        intermediate_points=args.intermediate_points,
                        attempts=max(1, args.attempts),
                        max_iterations=max(1, args.max_iterations),
                        population_size=max(2, args.population_size),
                        seed=args.seed,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    checkpoint.record_result(
                        run_id=summary.run_id,
                        profile_id=profile.profile_id,
                        case_id=profile.case_id,
                        ordinal=ordinal,
                        status="error",
                        error_text=error_text,
                    )
                    print(f"      error recorded transactionally: {error_text}", flush=True)
                    continue

                if candidate is None:
                    checkpoint.record_result(
                        run_id=summary.run_id,
                        profile_id=profile.profile_id,
                        case_id=profile.case_id,
                        ordinal=ordinal,
                        status="no_candidate",
                    )
                    print("      no candidate found; result checkpointed", flush=True)
                else:
                    checkpoint.record_result(
                        run_id=summary.run_id,
                        profile_id=profile.profile_id,
                        case_id=profile.case_id,
                        ordinal=ordinal,
                        status="found",
                        candidate=candidate.to_dict(),
                    )
                    print(
                        f"      found and checkpointed: closure={candidate.closure_error:.3g}, "
                        f"turn={candidate.turn_error:.3g}, area={candidate.signed_area:.3g}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            interrupted = True
            print(
                "Geometry search interrupted. Every previously completed profile is durable; "
                "the interrupted profile will be retried by the same command.",
                flush=True,
            )

        summary = checkpoint.summary(summary.run_id, run_key, len(selected))
        candidates = checkpoint.load_candidates(summary.run_id)
        summary_data = _summary_dict(
            summary, complete=(summary.remaining_count == 0 and not interrupted)
        )
        errors = checkpoint.error_records(summary.run_id)
        if errors:
            summary_data["errors"] = errors

    exit_code = 130 if interrupted else 0
    return candidates, summary_data, exit_code


def main() -> int:
    args = build_parser().parse_args()
    if args.view_only:
        candidates = _load_candidate_file(args.output)
        if not args.no_gui:
            launch_viewer(candidates)
        return 0

    if not args.input.exists():
        print(f"Missing survivor file: {args.input}", file=sys.stderr)
        print("Run the audit first.", file=sys.stderr)
        return 2
    if args.intermediate_points < 0:
        print("--intermediate-points must be nonnegative", file=sys.stderr)
        return 2
    if args.fresh and not args.resume:
        print("--fresh requires transactional resume/checkpoint mode", file=sys.stderr)
        return 2

    profiles = load_survivors(args.input)
    selected = select_profiles(profiles, max_profiles=args.max_profiles)
    print(
        f"Loaded {len(profiles)} surviving profiles from {args.input}; "
        f"selected {len(selected)} for this run",
        flush=True,
    )

    if args.resume:
        candidates, search_summary, exit_code = _run_transactional_search(
            args, selected
        )
        write_candidates(
            args.output,
            args.input,
            candidates,
            args.intermediate_points,
            search_summary=search_summary,
            checkpoint_path=args.checkpoint,
        )
    else:
        found = search_profiles(
            selected,
            max_profiles=0,
            intermediate_points=args.intermediate_points,
            attempts=max(1, args.attempts),
            max_iterations=max(1, args.max_iterations),
            population_size=max(2, args.population_size),
            seed=args.seed,
        )
        candidates = [candidate.to_dict() for candidate in found]
        search_summary = {
            "selected_profile_count": len(selected),
            "completed_profile_count": len(selected),
            "found_count": len(found),
            "complete": True,
            "checkpoint_disabled": True,
        }
        write_candidates(
            args.output,
            args.input,
            candidates,
            args.intermediate_points,
            search_summary=search_summary,
        )
        exit_code = 0

    print(
        f"Wrote {len(candidates)} candidates to {args.output}; "
        f"completed {search_summary.get('completed_profile_count', 0)}/"
        f"{search_summary.get('selected_profile_count', len(selected))}",
        flush=True,
    )
    if not args.no_gui and candidates:
        launch_viewer(candidates)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
