#!/usr/bin/env python3
"""Contact-side validation and reporting.

Opposite-side parity is now assigned while a ``PlacementCase`` is constructed.
This module remains as a small independent validator/reporting layer and as a
backward-compatible command-line entry point.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import projection_topology as topology
import symbolic_enumerator as base

DIRECT = topology.DIRECT
REFLECTED = topology.REFLECTED
LEFT = topology.LEFT
RIGHT = topology.RIGHT
required_mirror_sign = topology.required_mirror_sign


@dataclass(frozen=True)
class ContactSideConstraint:
    projection: str
    target_direction: int
    mirror_sign: int
    reference_interior_side: int
    copy_interior_side: int
    satisfied: bool

    @property
    def required_mirror_sign(self) -> int:
        """Backward-compatible alias."""
        return self.mirror_sign

    @property
    def required_isometry(self) -> str:
        return topology.isometry_name(self.mirror_sign)

    def to_dict(self) -> Dict[str, object]:
        return {
            "projection": self.projection,
            "target_direction": base.direction_name(self.target_direction),
            "mirror_sign": self.mirror_sign,
            "required_mirror_sign": self.mirror_sign,
            "required_isometry": self.required_isometry,
            "reference_interior_side": "left",
            "copy_interior_side": "right" if self.copy_interior_side == RIGHT else "left",
            "satisfied": self.satisfied,
        }


@dataclass(frozen=True)
class ContactSideAnalysis:
    constraints: Tuple[ContactSideConstraint, ContactSideConstraint]
    allow_reflections: bool
    feasible: bool
    discard_reason: Optional[str]

    @property
    def mirror_sign_a(self) -> int:
        return self.constraints[0].mirror_sign

    @property
    def mirror_sign_b(self) -> int:
        return self.constraints[1].mirror_sign

    @property
    def requires_reflection(self) -> bool:
        return any(item.mirror_sign == REFLECTED for item in self.constraints)

    def parity_label(self) -> str:
        return topology.parity_label(self.mirror_sign_a, self.mirror_sign_b)

    def to_dict(self) -> Dict[str, object]:
        return {
            "stage": "placement invariant validation",
            "convention": {
                "prototype_orientation": "counterclockwise",
                "positive_boundary_interior_side": "left",
                "contact_requirement": "the two tile interiors lie on opposite sides",
                "parity_equation": "mirror_sign * target_direction = -1",
            },
            "allow_reflections": self.allow_reflections,
            "embedded_parity": self.parity_label(),
            "requires_reflection": self.requires_reflection,
            "feasible": self.feasible,
            "discard_reason": self.discard_reason,
            "constraints": [item.to_dict() for item in self.constraints],
        }


def _constraint(projection: str, direction: int, mirror_sign: int) -> ContactSideConstraint:
    copy_side = topology.copy_interior_side(direction, mirror_sign)
    return ContactSideConstraint(
        projection=projection,
        target_direction=direction,
        mirror_sign=mirror_sign,
        reference_interior_side=LEFT,
        copy_interior_side=copy_side,
        satisfied=(
            mirror_sign == topology.required_mirror_sign(direction)
            and copy_side == RIGHT
        ),
    )


def analyze_contact_sides(
    case: base.PlacementCase,
    allow_reflections: bool = True,
) -> ContactSideAnalysis:
    """Validate the parity already embedded in ``case``."""
    constraints = (
        _constraint("A", case.a_direction, case.a_mirror_sign),
        _constraint("B", case.b_direction, case.b_mirror_sign),
    )
    requires_reflection = any(item.mirror_sign == REFLECTED for item in constraints)
    feasible = all(item.satisfied for item in constraints)
    reason: Optional[str] = None

    if not feasible:
        reason = "The placement contains a parity inconsistent with opposite-side contact."
    elif requires_reflection and not allow_reflections:
        feasible = False
        reason = "This placement requires at least one reflected copy."

    return ContactSideAnalysis(
        constraints=constraints,
        allow_reflections=allow_reflections,
        feasible=feasible,
        discard_reason=reason,
    )


def command_case(args: argparse.Namespace) -> int:
    case = base.find_case(args.case_id)
    analysis = analyze_contact_sides(case, allow_reflections=not args.direct_copies_only)
    print(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the contact parity embedded in a placement case."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    case_parser = subparsers.add_parser("case")
    case_parser.add_argument("case_id", type=int)
    case_parser.add_argument("--direct-copies-only", action="store_true")
    case_parser.set_defaults(handler=command_case)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
