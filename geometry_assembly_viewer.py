#!/usr/bin/env python3
"""Display geometric candidates using their formal terminal mappings.

The viewer is deliberately passive. For each candidate it:

1. retrieves the unique formal survivor with the same profile_id;
2. checks that case_id and formal profile text match;
3. reads source/target boundary indices and mirror_sign from terminal_mapping;
4. constructs the unique rigid isometry sending the target endpoint pair to the
   source endpoint pair;
5. draws the reference polygon and every copy whose endpoint pair determines a
   valid isometry.

It performs no mapping search, no least-squares fit, no interpolation, no
candidate ranking and no contact validation. Invalid or missing metadata only
prevent the affected copy from being displayed.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import settings

Point = Tuple[float, float]


@dataclass(frozen=True)
class CopyMapping:
    copy_index: int
    mirror_sign: int
    source_start_boundary: int
    source_end_boundary: int
    target_start_boundary: int
    target_end_boundary: int


@dataclass(frozen=True)
class SurvivorLink:
    profile_id: int
    case_id: int
    formal_profile: str
    mappings: Tuple[CopyMapping, ...]


@dataclass(frozen=True)
class CandidateRecord:
    payload: Mapping[str, object]
    profile_id: int
    case_id: int
    formal_profile: str
    vertices: Tuple[Point, ...]
    boundary_points: Mapping[int, Point]


@dataclass(frozen=True)
class Transform2D:
    mirror: bool
    a_real: float
    a_imag: float
    t_real: float
    t_imag: float

    def apply(self, point: Point) -> Point:
        x, y = point
        if self.mirror:
            # a * conjugate(z) + t
            return (
                self.a_real * x + self.a_imag * y + self.t_real,
                self.a_imag * x - self.a_real * y + self.t_imag,
            )
        # a * z + t
        return (
            self.a_real * x - self.a_imag * y + self.t_real,
            self.a_imag * x + self.a_real * y + self.t_imag,
        )


@dataclass(frozen=True)
class CopyPlacement:
    mapping: CopyMapping
    transform: Optional[Transform2D]
    error: Optional[str]

    @property
    def ok(self) -> bool:
        return self.transform is not None and self.error is None


@dataclass(frozen=True)
class AssembledCandidate:
    candidate: CandidateRecord
    placements: Tuple[CopyPlacement, ...]
    link_error: Optional[str] = None


def _normalize_profile_text(value: object) -> str:
    return " ".join(str(value).split())


def _iter_top_level_array(
    path: Path,
    key: str,
    *,
    chunk_size: int = 1024 * 1024,
) -> Iterable[Mapping[str, object]]:
    decoder = json.JSONDecoder()
    key_token = json.dumps(key)
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"Missing top-level JSON array {key!r} in {path}")
            buffer += chunk
            key_index = buffer.find(key_token)
            if key_index >= 0:
                colon_index = buffer.find(":", key_index + len(key_token))
                array_index = buffer.find("[", colon_index + 1) if colon_index >= 0 else -1
                if array_index >= 0:
                    buffer = buffer[array_index + 1 :]
                    break
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


def _candidate_boundary_points(candidate: Mapping[str, object]) -> Dict[int, Point]:
    raw_vertices = candidate.get("vertices")
    raw_metadata = candidate.get("edge_metadata")
    if not isinstance(raw_vertices, Sequence) or isinstance(raw_vertices, (str, bytes)):
        raise ValueError("Candidate has no valid vertices array")
    if not isinstance(raw_metadata, Sequence) or isinstance(raw_metadata, (str, bytes)):
        raise ValueError("Candidate has no valid edge_metadata array")

    vertices: List[Point] = [(float(item[0]), float(item[1])) for item in raw_vertices]
    first_edge_by_occurrence: Dict[int, Tuple[int, Mapping[str, object]]] = {}
    for ordinal, item in enumerate(raw_metadata):
        if not isinstance(item, Mapping):
            raise ValueError("Each edge_metadata entry must be an object")
        occurrence = int(item["occurrence"])
        subedge = int(item.get("subedge_index", item.get("edge_index", ordinal)))
        previous = first_edge_by_occurrence.get(occurrence)
        if previous is None or subedge < previous[0]:
            first_edge_by_occurrence[occurrence] = (subedge, item)

    if not first_edge_by_occurrence:
        raise ValueError("Candidate edge_metadata contains no formal occurrences")

    occurrence_count = max(first_edge_by_occurrence) + 1
    missing = [index for index in range(occurrence_count) if index not in first_edge_by_occurrence]
    if missing:
        raise ValueError(f"Candidate is missing formal occurrence(s): {missing}")

    return {
        occurrence: vertices[int(item["start_vertex"])]
        for occurrence, (_subedge, item) in first_edge_by_occurrence.items()
    }


def _load_candidates(path: Path) -> List[CandidateRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_candidates = payload.get("candidates", [])
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        raise ValueError(f"{path} does not contain a valid candidates array")

    output: List[CandidateRecord] = []
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            raise ValueError("Each candidate must be an object")
        vertices = tuple((float(item[0]), float(item[1])) for item in raw["vertices"])
        output.append(
            CandidateRecord(
                payload=raw,
                profile_id=int(raw["profile_id"]),
                case_id=int(raw.get("case_id", -1)),
                formal_profile=_normalize_profile_text(raw.get("formal_profile", "")),
                vertices=vertices,
                boundary_points=_candidate_boundary_points(raw),
            )
        )
    return output


def _parse_copy_mapping(raw: Mapping[str, object], fallback_index: int) -> CopyMapping:
    mirror_sign = int(raw.get("mirror_sign", 0))
    if mirror_sign not in (-1, 1):
        raise ValueError(f"Invalid mirror_sign {mirror_sign!r}")
    return CopyMapping(
        copy_index=int(raw.get("copy_index", fallback_index)),
        mirror_sign=mirror_sign,
        source_start_boundary=int(raw["source_start_boundary"]),
        source_end_boundary=int(raw["source_end_boundary"]),
        target_start_boundary=int(raw["target_start_boundary"]),
        target_end_boundary=int(raw["target_end_boundary"]),
    )


def _load_survivor_links(path: Path) -> Dict[int, List[SurvivorLink]]:
    links: Dict[int, List[SurvivorLink]] = {}
    for record in _iter_top_level_array(path, "profiles"):
        terminal_mapping = record.get("terminal_mapping")
        solution = record.get("solution")
        if not isinstance(terminal_mapping, Mapping) or not isinstance(solution, Mapping):
            continue
        raw_mappings = terminal_mapping.get("mappings")
        if not isinstance(raw_mappings, Sequence) or isinstance(raw_mappings, (str, bytes)):
            continue
        parsed = tuple(
            _parse_copy_mapping(raw, index)
            for index, raw in enumerate(raw_mappings)
            if isinstance(raw, Mapping)
        )
        profile_id = int(record["profile_id"])
        link = SurvivorLink(
            profile_id=profile_id,
            case_id=int(record.get("case_id", -1)),
            formal_profile=_normalize_profile_text(solution.get("profile", "")),
            mappings=tuple(sorted(parsed, key=lambda item: item.copy_index)),
        )
        links.setdefault(profile_id, []).append(link)
    return links


def _point(boundaries: Mapping[int, Point], index: int) -> Point:
    try:
        return boundaries[index]
    except KeyError as exc:
        raise ValueError(f"Boundary {index} is absent from the geometric candidate") from exc


def _complex(point: Point) -> complex:
    return complex(point[0], point[1])


def _build_formal_transform(
    candidate: CandidateRecord,
    mapping: CopyMapping,
    *,
    relative_length_tolerance: float,
) -> Transform2D:
    """Construct the isometry prescribed by the formal endpoint mapping.

    The copy is a transformed prototype. Therefore the target boundary pair on
    the prototype copy is sent to the source boundary pair on the fixed
    reference prototype.
    """
    source_start = _complex(_point(candidate.boundary_points, mapping.source_start_boundary))
    source_end = _complex(_point(candidate.boundary_points, mapping.source_end_boundary))
    target_start = _complex(_point(candidate.boundary_points, mapping.target_start_boundary))
    target_end = _complex(_point(candidate.boundary_points, mapping.target_end_boundary))

    source_vector = source_end - source_start
    target_vector = target_end - target_start
    source_length = abs(source_vector)
    target_length = abs(target_vector)
    scale = max(source_length, target_length, 1.0)
    if source_length <= 1e-14 or target_length <= 1e-14:
        raise ValueError("The formal endpoint pair is degenerate, so no unique isometry is defined")
    if abs(source_length - target_length) > relative_length_tolerance * scale:
        raise ValueError(
            "The prescribed endpoint pairs have different chord lengths "
            f"({source_length:.12g} versus {target_length:.12g})"
        )

    if mapping.mirror_sign == -1:
        raw_multiplier = source_vector / target_vector.conjugate()
        multiplier = raw_multiplier / abs(raw_multiplier)
        translation = source_start - multiplier * target_start.conjugate()
        return Transform2D(
            mirror=True,
            a_real=float(multiplier.real),
            a_imag=float(multiplier.imag),
            t_real=float(translation.real),
            t_imag=float(translation.imag),
        )

    raw_multiplier = source_vector / target_vector
    multiplier = raw_multiplier / abs(raw_multiplier)
    translation = source_start - multiplier * target_start
    return Transform2D(
        mirror=False,
        a_real=float(multiplier.real),
        a_imag=float(multiplier.imag),
        t_real=float(translation.real),
        t_imag=float(translation.imag),
    )


def _select_exact_link(candidate: CandidateRecord, links: Sequence[SurvivorLink]) -> SurvivorLink:
    if len(links) != 1:
        if not links:
            raise ValueError("No survivor record has this profile_id")
        raise ValueError(f"The survivor file contains {len(links)} records with this profile_id")
    link = links[0]
    if link.case_id != candidate.case_id:
        raise ValueError(
            f"case_id mismatch: candidate has {candidate.case_id}, survivor has {link.case_id}"
        )
    if link.formal_profile != candidate.formal_profile:
        raise ValueError("formal profile mismatch between candidate and survivor files")
    return link


def assemble_candidates(
    candidate_path: Path,
    survivor_path: Path,
    *,
    relative_length_tolerance: float = 1e-7,
) -> List[AssembledCandidate]:
    candidates = _load_candidates(candidate_path)
    links_by_profile = _load_survivor_links(survivor_path)
    if not candidates:
        raise ValueError(f"No candidate found in {candidate_path}")

    assembled: List[AssembledCandidate] = []
    for candidate in candidates:
        try:
            link = _select_exact_link(candidate, links_by_profile.get(candidate.profile_id, ()))
        except Exception as exc:
            assembled.append(
                AssembledCandidate(candidate=candidate, placements=(), link_error=str(exc))
            )
            continue

        placements: List[CopyPlacement] = []
        for mapping in link.mappings:
            try:
                transform = _build_formal_transform(
                    candidate,
                    mapping,
                    relative_length_tolerance=relative_length_tolerance,
                )
                placements.append(CopyPlacement(mapping, transform, None))
            except Exception as exc:
                placements.append(CopyPlacement(mapping, None, str(exc)))
        assembled.append(AssembledCandidate(candidate=candidate, placements=tuple(placements)))
    return assembled


def _prototype(candidate: CandidateRecord) -> Tuple[Point, ...]:
    if candidate.vertices and candidate.vertices[0] == candidate.vertices[-1]:
        return candidate.vertices[:-1]
    return candidate.vertices


def _combined_bounds(polygons: Sequence[Sequence[Point]]) -> Tuple[float, float, float, float]:
    xs = [point[0] for polygon in polygons for point in polygon]
    ys = [point[1] for polygon in polygons for point in polygon]
    return min(xs), max(xs), min(ys), max(ys)


def launch_viewer(assembled_candidates: Sequence[AssembledCandidate]) -> None:
    if not assembled_candidates:
        print("No candidates to display.")
        return
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise RuntimeError("Tkinter is required for the interactive viewer") from exc

    root = tk.Tk()
    root.title("Voderberg assembly viewer")
    root.geometry("1400x900")
    root.minsize(1000, 700)

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    title_var = tk.StringVar()
    ttk.Label(top, textvariable=title_var, font=("Segoe UI", 12, "bold")).pack(fill="x")

    body = ttk.Frame(root, padding=(8, 0, 8, 8))
    body.pack(fill="both", expand=True)
    canvas = tk.Canvas(
        body,
        background="#343941",
        highlightthickness=1,
        highlightbackground="#69707b",
    )
    canvas.pack(side="left", fill="both", expand=True)
    side = ttk.Frame(body, padding=(12, 0, 0, 0), width=310)
    side.pack(side="right", fill="y")
    info = tk.Text(side, width=38, height=28, wrap="word", state="disabled")
    info.pack(fill="both", expand=True)

    controls = ttk.Frame(root, padding=8)
    controls.pack(fill="x")
    index = {"value": 0}
    show_reference = tk.BooleanVar(value=True)
    show_copy1 = tk.BooleanVar(value=True)
    show_copy2 = tk.BooleanVar(value=True)
    index_var = tk.StringVar()
    status_var = tk.StringVar()
    view = {
        "scale": 1.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "initialized_for": None,
        "last_drag": None,
    }
    colors = {"reference": "#2f81f7", "copy1": "#ff7a00", "copy2": "#23c55e"}

    def candidate_polygons(item: AssembledCandidate) -> List[Tuple[str, Sequence[Point]]]:
        prototype = _prototype(item.candidate)
        output: List[Tuple[str, Sequence[Point]]] = []
        if show_reference.get():
            output.append(("reference", prototype))
        for placement in item.placements:
            if not placement.ok or placement.transform is None:
                continue
            layer = f"copy{placement.mapping.copy_index + 1}"
            if layer == "copy1" and not show_copy1.get():
                continue
            if layer == "copy2" and not show_copy2.get():
                continue
            output.append((layer, [placement.transform.apply(point) for point in prototype]))
        return output

    def reset_view(item: AssembledCandidate) -> None:
        polygons = [polygon for _name, polygon in candidate_polygons(item)] or [_prototype(item.candidate)]
        min_x, max_x, min_y, max_y = _combined_bounds(polygons)
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)
        margin = 55.0
        view["scale"] = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        view["offset_x"] = width * 0.5 - view["scale"] * center_x
        view["offset_y"] = height * 0.5 + view["scale"] * center_y
        view["initialized_for"] = index["value"]

    def screen(point: Point) -> Point:
        return (
            view["offset_x"] + view["scale"] * point[0],
            view["offset_y"] - view["scale"] * point[1],
        )

    def format_info(item: AssembledCandidate) -> str:
        payload = item.candidate.payload
        lines = [
            f"Profile: {item.candidate.profile_id}",
            f"Case: {item.candidate.case_id}",
            f"Objective: {payload.get('objective', 'n/a')}",
            f"Signed area: {payload.get('signed_area', 'n/a')}",
            "",
            "Formal profile:",
            item.candidate.formal_profile,
            "",
        ]
        if item.link_error:
            lines.extend(["Metadata link error:", item.link_error, ""])
        for placement in item.placements:
            mapping = placement.mapping
            parity = "reflected" if mapping.mirror_sign == -1 else "direct"
            state = "OK" if placement.ok else "FAILED"
            lines.extend(
                [
                    f"Copy {mapping.copy_index + 1}: {parity} [{state}]",
                    f"  target boundaries: {mapping.target_start_boundary} -> {mapping.target_end_boundary}",
                    f"  source boundaries: {mapping.source_start_boundary} -> {mapping.source_end_boundary}",
                ]
            )
            if placement.error:
                lines.append(f"  error: {placement.error}")
            lines.append("")
        return "\n".join(lines)

    def render(*_args: object) -> None:
        item = assembled_candidates[index["value"]]
        if view["initialized_for"] != index["value"]:
            reset_view(item)
        canvas.delete("all")
        polygons = candidate_polygons(item)
        for layer, polygon in polygons:
            coordinates: List[float] = []
            for point in polygon:
                sx, sy = screen(point)
                coordinates.extend((sx, sy))
            if len(coordinates) >= 6:
                canvas.create_polygon(coordinates, fill=colors.get(layer, "#a0a0a0"), outline="")

        failed = sum(1 for placement in item.placements if not placement.ok)
        ok = sum(1 for placement in item.placements if placement.ok)
        if item.link_error:
            status_var.set("Reference only: formal metadata link failed.")
        elif failed:
            status_var.set(f"Partial display: {ok} copy/copies shown, {failed} unavailable.")
        else:
            status_var.set("")
        title_var.set(
            f"Candidate {index['value'] + 1}/{len(assembled_candidates)}  •  "
            f"profile {item.candidate.profile_id}  •  case {item.candidate.case_id}"
        )
        index_var.set(f"{index['value'] + 1} / {len(assembled_candidates)}")
        info.configure(state="normal")
        info.delete("1.0", "end")
        info.insert("1.0", format_info(item))
        info.configure(state="disabled")

    def move(delta: int) -> None:
        index["value"] = (index["value"] + delta) % len(assembled_candidates)
        view["initialized_for"] = None
        render()

    def refresh_visibility() -> None:
        view["initialized_for"] = None
        render()

    def press(event: object) -> None:
        view["last_drag"] = (float(event.x), float(event.y))

    def drag(event: object) -> None:
        previous = view["last_drag"]
        if previous is None:
            return
        view["offset_x"] += float(event.x) - previous[0]
        view["offset_y"] += float(event.y) - previous[1]
        view["last_drag"] = (float(event.x), float(event.y))
        render()

    def release(_event: object) -> None:
        view["last_drag"] = None

    def zoom(x: float, y: float, factor: float) -> None:
        old_scale = view["scale"]
        new_scale = max(1e-6, min(old_scale * factor, 1e6))
        world_x = (x - view["offset_x"]) / old_scale
        world_y = (view["offset_y"] - y) / old_scale
        view["scale"] = new_scale
        view["offset_x"] = x - new_scale * world_x
        view["offset_y"] = y + new_scale * world_y
        render()

    def mousewheel(event: object) -> None:
        delta = getattr(event, "delta", 0)
        if delta:
            zoom(float(event.x), float(event.y), 1.1 if delta > 0 else 1.0 / 1.1)

    ttk.Button(controls, text="Previous", command=lambda: move(-1)).pack(side="left")
    ttk.Label(controls, textvariable=index_var, padding=(10, 0)).pack(side="left")
    ttk.Button(controls, text="Next", command=lambda: move(1)).pack(side="left")
    ttk.Button(
        controls,
        text="Reset view",
        command=lambda: (reset_view(assembled_candidates[index["value"]]), render()),
    ).pack(side="left", padx=(10, 18))
    ttk.Checkbutton(controls, text="Reference", variable=show_reference, command=refresh_visibility).pack(side="left")
    ttk.Checkbutton(controls, text="Copy 1", variable=show_copy1, command=refresh_visibility).pack(side="left")
    ttk.Checkbutton(controls, text="Copy 2", variable=show_copy2, command=refresh_visibility).pack(side="left")
    ttk.Button(controls, text="Close", command=root.destroy).pack(side="right")
    ttk.Label(controls, textvariable=status_var, foreground="#b42318").pack(side="right", padx=(0, 12))

    root.bind("<Left>", lambda _event: move(-1))
    root.bind("<Right>", lambda _event: move(1))
    canvas.bind("<Configure>", lambda _event: render())
    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", release)
    canvas.bind("<MouseWheel>", mousewheel)
    canvas.bind("<Button-4>", lambda event: zoom(float(event.x), float(event.y), 1.1))
    canvas.bind("<Button-5>", lambda event: zoom(float(event.x), float(event.y), 1.0 / 1.1))

    root.after(100, render)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display candidates using their exact formal endpoint mappings."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(settings.GEOMETRY_CANDIDATES_FILENAME),
        help="Geometric candidate JSON file.",
    )
    parser.add_argument(
        "--survivors",
        type=Path,
        default=Path(settings.AUDIT_SURVIVORS_FILENAME),
        help="Formal survivor JSON containing terminal_mapping.",
    )
    parser.add_argument(
        "--relative-length-tolerance",
        type=float,
        default=1e-7,
        help="Tolerance used only to decide whether the two endpoint chords define an isometry.",
    )
    parser.add_argument("--no-gui", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    assembled = assemble_candidates(
        args.input,
        args.survivors,
        relative_length_tolerance=args.relative_length_tolerance,
    )
    print(f"Loaded {len(assembled)} candidate(s).")
    if args.no_gui:
        for item in assembled:
            ok = sum(1 for placement in item.placements if placement.ok)
            failed = sum(1 for placement in item.placements if not placement.ok)
            link = "link-error" if item.link_error else "linked"
            print(f"profile {item.candidate.profile_id}: {link}, {ok} copy/copies, {failed} failed")
        return 0
    launch_viewer(assembled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
