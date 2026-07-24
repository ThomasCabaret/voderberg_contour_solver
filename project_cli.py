#!/usr/bin/env python3
"""Single stable command-line entry point for the project."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import settings

ROOT = Path(__file__).resolve().parent


def _run(command: Sequence[str], *, capture: bool = False) -> int:
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        capture_output=capture,
        check=False,
    )
    if capture:
        stdout, stderr = completed.stdout or "", completed.stderr or ""
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        report = stdout + (
            ("\n" if stdout and not stdout.endswith("\n") else "") + stderr
            if stderr
            else ""
        )
        (ROOT / settings.TEST_RESULTS_FILENAME).write_text(report, encoding="utf-8")
        print(f"Test report: {settings.TEST_RESULTS_FILENAME}")
    return completed.returncode


def command_tests(extra: Sequence[str]) -> int:
    return _run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-v",
            "-p",
            settings.TEST_MODULE_PATTERN,
            *extra,
        ],
        capture=True,
    )


def command_audit(extra: Sequence[str]) -> int:
    return _run([sys.executable, settings.AUDIT_SCRIPT_FILENAME, *extra])


def command_web(extra: Sequence[str]) -> int:
    return _run([sys.executable, settings.WEB_SCRIPT_FILENAME, *extra])



def command_geometry(extra: Sequence[str]) -> int:
    return _run([sys.executable, settings.GEOMETRY_SCRIPT_FILENAME, *extra])


def command_z3_voderberg(extra: Sequence[str]) -> int:
    return _run(
        [
            sys.executable,
            settings.Z3_SCRIPT_FILENAME,
            "--voderberg",
            "--output",
            settings.Z3_DEFAULT_SMT2_FILENAME,
            "--metadata",
            settings.Z3_DEFAULT_METADATA_FILENAME,
            *extra,
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Voderberg contour solver launcher")
    parser.add_argument(
        "command",
        choices=("tests", "audit", "web", "geometry", "z3-voderberg"),
    )
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    commands = {
        "tests": command_tests,
        "audit": command_audit,
        "web": command_web,
        "geometry": command_geometry,
        "z3-voderberg": command_z3_voderberg,
    }
    return commands[args.command](args.extra)


if __name__ == "__main__":
    raise SystemExit(main())
