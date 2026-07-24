#!/usr/bin/env python3
"""Transactional checkpoint storage for long heuristic geometry searches.

The checkpoint is intentionally independent from the numerical optimizer.  It
stores one durable row after every completed profile, including negative
results and errors.  A run is identified by the survivor-file content, the
ordered selected profile list, and every numerical search setting that can
change the result.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "geometry-search-v2"
FINAL_STATUSES = frozenset(("found", "no_candidate", "error"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def selected_profiles_digest(profiles: Sequence[object]) -> str:
    payload = [
        [int(getattr(profile, "profile_id")), int(getattr(profile, "case_id"))]
        for profile in profiles
    ]
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def build_run_identity(
    *,
    input_path: Path,
    input_sha256: str,
    selected_profiles: Sequence[object],
    configuration: Mapping[str, object],
) -> Tuple[str, Dict[str, object]]:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "input_path": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "selected_profile_count": len(selected_profiles),
        "selected_profiles_sha256": selected_profiles_digest(selected_profiles),
        "configuration": dict(configuration),
    }
    run_key = hashlib.sha256(_canonical_json(identity).encode("ascii")).hexdigest()
    return run_key, identity


@dataclass(frozen=True)
class ResumeSummary:
    run_id: int
    run_key: str
    selected_count: int
    completed_count: int
    found_count: int
    no_candidate_count: int
    error_count: int

    @property
    def remaining_count(self) -> int:
        return max(0, self.selected_count - self.completed_count)


class GeometryCheckpoint:
    """Small SQLite journal with one transaction per completed profile."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), timeout=60.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "GeometryCheckpoint":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_key TEXT NOT NULL UNIQUE,
                    identity_json TEXT NOT NULL,
                    selected_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS profile_results (
                    run_id INTEGER NOT NULL,
                    profile_id INTEGER NOT NULL,
                    case_id INTEGER NOT NULL,
                    ordinal INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    candidate_json TEXT,
                    error_text TEXT,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, profile_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS profile_results_run_ordinal
                    ON profile_results(run_id, ordinal);
                """
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO checkpoint_metadata(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    def prepare_run(
        self,
        *,
        run_key: str,
        identity: Mapping[str, object],
        selected_count: int,
        fresh: bool = False,
    ) -> ResumeSummary:
        now = _utc_now()
        identity_json = _canonical_json(identity)
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO runs(
                    run_key, identity_json, selected_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (run_key, identity_json, selected_count, now, now),
            )
            row = self.connection.execute(
                "SELECT run_id, identity_json, selected_count FROM runs WHERE run_key = ?",
                (run_key,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to create or retrieve geometry checkpoint run")
            if row["identity_json"] != identity_json or int(row["selected_count"]) != selected_count:
                raise RuntimeError("Geometry checkpoint run-key collision or incompatible schema")
            run_id = int(row["run_id"])
            if fresh:
                self.connection.execute(
                    "DELETE FROM profile_results WHERE run_id = ?", (run_id,)
                )
            self.connection.execute(
                "UPDATE runs SET updated_at = ? WHERE run_id = ?", (now, run_id)
            )
        return self.summary(run_id, run_key, selected_count)

    def summary(self, run_id: int, run_key: str, selected_count: int) -> ResumeSummary:
        rows = self.connection.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM profile_results
            WHERE run_id = ?
            GROUP BY status
            """,
            (run_id,),
        ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        completed = sum(counts.get(status, 0) for status in FINAL_STATUSES)
        return ResumeSummary(
            run_id=run_id,
            run_key=run_key,
            selected_count=selected_count,
            completed_count=completed,
            found_count=counts.get("found", 0),
            no_candidate_count=counts.get("no_candidate", 0),
            error_count=counts.get("error", 0),
        )

    def clear_error_results(self, run_id: int) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM profile_results WHERE run_id = ? AND status = 'error'",
                (run_id,),
            )
        return int(cursor.rowcount)

    def completed_profile_ids(self, run_id: int, *, retry_errors: bool) -> set[int]:
        statuses = ("found", "no_candidate") if retry_errors else tuple(FINAL_STATUSES)
        placeholders = ",".join("?" for _ in statuses)
        rows = self.connection.execute(
            f"SELECT profile_id FROM profile_results WHERE run_id = ? AND status IN ({placeholders})",
            (run_id, *statuses),
        ).fetchall()
        return {int(row["profile_id"]) for row in rows}

    def record_result(
        self,
        *,
        run_id: int,
        profile_id: int,
        case_id: int,
        ordinal: int,
        status: str,
        candidate: Optional[Mapping[str, object]] = None,
        error_text: Optional[str] = None,
    ) -> None:
        if status not in FINAL_STATUSES:
            raise ValueError(f"Unsupported checkpoint status: {status}")
        if status == "found" and candidate is None:
            raise ValueError("A found result must include a candidate")
        candidate_json = None if candidate is None else _canonical_json(candidate)
        now = _utc_now()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO profile_results(
                    run_id, profile_id, case_id, ordinal, status,
                    candidate_json, error_text, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, profile_id) DO UPDATE SET
                    case_id = excluded.case_id,
                    ordinal = excluded.ordinal,
                    status = excluded.status,
                    candidate_json = excluded.candidate_json,
                    error_text = excluded.error_text,
                    completed_at = excluded.completed_at
                """,
                (
                    run_id,
                    profile_id,
                    case_id,
                    ordinal,
                    status,
                    candidate_json,
                    error_text,
                    now,
                ),
            )
            self.connection.execute(
                "UPDATE runs SET updated_at = ? WHERE run_id = ?", (now, run_id)
            )

    def load_candidates(self, run_id: int) -> List[Dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT candidate_json
            FROM profile_results
            WHERE run_id = ? AND status = 'found'
            ORDER BY ordinal, profile_id
            """,
            (run_id,),
        ).fetchall()
        return [json.loads(str(row["candidate_json"])) for row in rows]

    def error_records(self, run_id: int) -> List[Dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT profile_id, case_id, ordinal, error_text, completed_at
            FROM profile_results
            WHERE run_id = ? AND status = 'error'
            ORDER BY ordinal, profile_id
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]
