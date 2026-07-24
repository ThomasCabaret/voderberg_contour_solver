"""Local topology attached to a contour projection.

The prototype boundary is counterclockwise, hence its interior is on the left
of a forward traversal.  If a target factor is read with direction ``d`` and
the copy is placed by an isometry of parity ``m``, the copy interior lies on
side ``m*d`` relative to the positive reference-boundary tangent.

A valid shared boundary requires opposite sides, therefore::

    m * d = -1
    m = -d

This is known as soon as a projection direction is chosen.  It is therefore an
invariant of a placement, not a posteriori geometric guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import settings

FORWARD = settings.FORWARD
REVERSE = settings.REVERSE
DIRECT = settings.DIRECT
REFLECTED = settings.REFLECTED
LEFT = settings.REFERENCE_INTERIOR_SIDE
RIGHT = settings.OPPOSITE_INTERIOR_SIDE


def validate_direction(direction: int) -> None:
    if direction not in (FORWARD, REVERSE):
        raise ValueError("direction must be FORWARD or REVERSE")


def validate_mirror_sign(mirror_sign: int) -> None:
    if mirror_sign not in (DIRECT, REFLECTED):
        raise ValueError("mirror_sign must be DIRECT or REFLECTED")


def required_mirror_sign(direction: int) -> int:
    """Return the unique copy parity placing interiors on opposite sides."""
    validate_direction(direction)
    return -direction


def copy_interior_side(direction: int, mirror_sign: int) -> int:
    validate_direction(direction)
    validate_mirror_sign(mirror_sign)
    return direction * mirror_sign


def is_opposite_side_contact(direction: int, mirror_sign: int) -> bool:
    return copy_interior_side(direction, mirror_sign) == RIGHT


def isometry_name(mirror_sign: int) -> str:
    validate_mirror_sign(mirror_sign)
    return "direct" if mirror_sign == DIRECT else "reflected"


def parity_label(mirror_sign_a: int, mirror_sign_b: int) -> str:
    return "".join(
        "D" if sign == DIRECT else "R"
        for sign in (mirror_sign_a, mirror_sign_b)
    )


@dataclass(frozen=True)
class ProjectionTopology:
    """Direction and parity data fixed at placement construction time."""

    projection: str
    direction: int
    mirror_sign: int

    def __post_init__(self) -> None:
        validate_direction(self.direction)
        validate_mirror_sign(self.mirror_sign)
        if self.mirror_sign != required_mirror_sign(self.direction):
            raise ValueError(
                "projection parity is inconsistent with opposite-side contact"
            )

    @classmethod
    def from_direction(cls, projection: str, direction: int) -> "ProjectionTopology":
        return cls(
            projection=projection,
            direction=direction,
            mirror_sign=required_mirror_sign(direction),
        )

    @property
    def requires_reflection(self) -> bool:
        return self.mirror_sign == REFLECTED

    @property
    def isometry(self) -> str:
        return isometry_name(self.mirror_sign)
