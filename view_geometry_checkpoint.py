#!/usr/bin/env python3
"""Export found geometry candidates from a live SQLite checkpoint and view them.

This script does not modify the solver or the checkpoint. It can run while the
geometry search is still writing completed profiles in WAL mode.

Typical use, from the project directory:

    py -3 view_geometry_checkpoint.py

It exports the most recently updated run to
``geometric_candidates_snapshot.json`` and opens the existing Tk viewer.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

DEFAULT_CHECKPOINT = "geometry_search_checkpoint.sqlite3"
DEFAULT_OUTPUT = "geometric_candidates_snapshot.json"
FINAL_STATUSES = ("found", "no_candidate", "error")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_project_directory(script_path: Path) -> Path:
    candidates = (
        Path.cwd(),
        script_path.parent,
        Path.cwd() / "srn2solver",
        script_path.parent / "srn2solver",
    )
    for candidate in candidates:
        if (candidate / "geometry_search_viewer.py").is_file():
            return candidate.resolve()
    return Path.cwd().resolve()


def _resolve_path(path: Optional[Path], *, base: Path, default_name: str) -> Path:
    if path is None:
        return base / default_name
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    connection = sqlite3.connect(str(path), timeout=15.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=15000")
    return connection


def _load_runs(connection: sqlite3.Connection, limit: int = 20) -> List[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            r.run_id,
            r.run_key,
            r.identity_json,
            r.selected_count,
            r.created_at,
            r.updated_at,
            SUM(CASE WHEN p.status = 'found' THEN 1 ELSE 0 END) AS found_count,
            SUM(CASE WHEN p.status = 'no_candidate' THEN 1 ELSE 0 END) AS no_candidate_count,
            SUM(CASE WHEN p.status = 'error' THEN 1 ELSE 0 END) AS error_count
        FROM runs AS r
        LEFT JOIN profile_results AS p ON p.run_id = r.run_id
        GROUP BY r.run_id
        ORDER BY r.updated_at DESC, r.run_id DESC
        LIMIT ?
        """,
        (max(1, limit),),
    ).fetchall()


def _print_runs(rows: Sequence[sqlite3.Row]) -> None:
    if not rows:
        print("No checkpoint runs found.")
        return
    print("run_id  completed/selected  found  updated_at                   run_key")
    for row in rows:
        found = int(row["found_count"] or 0)
        no_candidate = int(row["no_candidate_count"] or 0)
        errors = int(row["error_count"] or 0)
        completed = found + no_candidate + errors
        selected = int(row["selected_count"])
        print(
            f"{int(row['run_id']):6d}  {completed:9d}/{selected:<8d}  "
            f"{found:5d}  {str(row['updated_at']):27s}  {str(row['run_key'])[:16]}"
        )


def _select_run(
    connection: sqlite3.Connection,
    *,
    run_id: Optional[int],
    run_key_prefix: Optional[str],
) -> sqlite3.Row:
    if run_id is not None:
        row = connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No checkpoint run with run_id={run_id}")
        return row

    if run_key_prefix:
        rows = connection.execute(
            """
            SELECT * FROM runs
            WHERE run_key LIKE ?
            ORDER BY updated_at DESC, run_id DESC
            """,
            (run_key_prefix + "%",),
        ).fetchall()
        if not rows:
            raise ValueError(f"No checkpoint run begins with {run_key_prefix!r}")
        if len(rows) > 1:
            keys = ", ".join(str(row["run_key"])[:16] for row in rows[:5])
            raise ValueError(
                f"Run-key prefix {run_key_prefix!r} is ambiguous; matches include {keys}"
            )
        return rows[0]

    row = connection.execute(
        "SELECT * FROM runs ORDER BY updated_at DESC, run_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise ValueError("The checkpoint contains no runs")
    return row


def _snapshot_run(
    connection: sqlite3.Connection,
    run: sqlite3.Row,
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    run_id = int(run["run_id"])
    identity = json.loads(str(run["identity_json"]))
    if not isinstance(identity, dict):
        raise ValueError(f"Run {run_id} has invalid identity_json")

    status_rows = connection.execute(
        """
        SELECT status, COUNT(*) AS count
        FROM profile_results
        WHERE run_id = ?
        GROUP BY status
        """,
        (run_id,),
    ).fetchall()
    counts = {str(row["status"]): int(row["count"]) for row in status_rows}

    candidate_rows = connection.execute(
        """
        SELECT candidate_json
        FROM profile_results
        WHERE run_id = ? AND status = 'found' AND candidate_json IS NOT NULL
        ORDER BY ordinal, profile_id
        """,
        (run_id,),
    ).fetchall()
    candidates: List[Dict[str, object]] = []
    for row in candidate_rows:
        candidate = json.loads(str(row["candidate_json"]))
        if not isinstance(candidate, dict):
            raise ValueError(f"Run {run_id} contains a non-object candidate")
        candidates.append(candidate)

    error_rows = connection.execute(
        """
        SELECT profile_id, case_id, ordinal, error_text, completed_at
        FROM profile_results
        WHERE run_id = ? AND status = 'error'
        ORDER BY ordinal, profile_id
        """,
        (run_id,),
    ).fetchall()

    selected_count = int(run["selected_count"])
    completed_count = sum(counts.get(status, 0) for status in FINAL_STATUSES)
    summary: Dict[str, object] = {
        "run_id": run_id,
        "run_key": str(run["run_key"]),
        "selected_profile_count": selected_count,
        "completed_profile_count": completed_count,
        "remaining_profile_count": max(0, selected_count - completed_count),
        "found_count": counts.get("found", 0),
        "no_candidate_count": counts.get("no_candidate", 0),
        "error_count": counts.get("error", 0),
        "complete": completed_count >= selected_count,
        "run_created_at": str(run["created_at"]),
        "run_updated_at": str(run["updated_at"]),
        "snapshot_generated_at": _utc_now(),
    }
    if error_rows:
        summary["errors"] = [dict(row) for row in error_rows]
    return candidates, summary, identity


def _write_snapshot(
    path: Path,
    *,
    checkpoint_path: Path,
    candidates: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    identity: Mapping[str, object],
) -> None:
    configuration = identity.get("configuration", {})
    if not isinstance(configuration, Mapping):
        configuration = {}
    intermediate_points = int(configuration.get("intermediate_points", 0))
    payload = {
        "metadata": {
            "source_survivors": identity.get("input_path"),
            "candidate_count": len(candidates),
            "intermediate_points_per_variable": intermediate_points,
            "edges_per_variable": intermediate_points + 1,
            "method": "live snapshot exported from transactional SQLite checkpoint",
            "proof_status": (
                "found candidates are heuristic geometric candidates stored by the "
                "geometry search at the time of this snapshot"
            ),
            "transactional_checkpoint": str(checkpoint_path),
            "search_configuration": dict(configuration),
            "search_summary": dict(summary),
        },
        "candidates": list(candidates),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    temporary.replace(path)


def _launch_project_viewer(
    project_directory: Path,
    candidates: Sequence[Mapping[str, object]],
) -> None:
    if str(project_directory) not in sys.path:
        sys.path.insert(0, str(project_directory))
    try:
        from geometry_search_viewer import launch_viewer
    except ImportError as exc:
        raise RuntimeError(
            "Could not import geometry_search_viewer.py. Put this script in the "
            "project directory, or in the directory containing srn2solver/."
        ) from exc
    launch_viewer(candidates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export candidates already stored in a live geometry SQLite checkpoint "
            "and open the existing viewer."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=f"Checkpoint path (default: project/{DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Snapshot JSON path (default: checkpoint directory/{DEFAULT_OUTPUT}).",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--run-id", type=int, help="Export a specific SQLite run_id.")
    selector.add_argument(
        "--run-key",
        help="Export the unique run whose run_key begins with this prefix.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List recent runs and exit without exporting.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Write the snapshot JSON without opening the Tk viewer.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    script_path = Path(__file__).resolve()
    project_directory = _find_project_directory(script_path)
    checkpoint_path = _resolve_path(
        args.checkpoint, base=project_directory, default_name=DEFAULT_CHECKPOINT
    )

    try:
        with _connect_read_only(checkpoint_path) as connection:
            connection.execute("BEGIN")
            if args.list_runs:
                _print_runs(_load_runs(connection))
                return 0
            run = _select_run(
                connection, run_id=args.run_id, run_key_prefix=args.run_key
            )
            candidates, summary, identity = _snapshot_run(connection, run)
    except (FileNotFoundError, sqlite3.Error, ValueError, json.JSONDecodeError) as exc:
        print(f"Snapshot export failed: {exc}", file=sys.stderr)
        return 2

    output_path = _resolve_path(
        args.output,
        base=checkpoint_path.parent,
        default_name=DEFAULT_OUTPUT,
    )
    try:
        _write_snapshot(
            output_path,
            checkpoint_path=checkpoint_path,
            candidates=candidates,
            summary=summary,
            identity=identity,
        )
    except OSError as exc:
        print(f"Could not write snapshot: {exc}", file=sys.stderr)
        return 2

    print(
        f"Exported {len(candidates)} candidate(s) from run "
        f"{summary['run_id']} ({summary['completed_profile_count']}/"
        f"{summary['selected_profile_count']} profiles completed) to {output_path}",
        flush=True,
    )

    if args.no_gui:
        return 0
    if not candidates:
        print("No found candidate is currently stored; the viewer was not opened.")
        return 0
    try:
        _launch_project_viewer(project_directory, candidates)
    except RuntimeError as exc:
        print(f"Viewer launch failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
