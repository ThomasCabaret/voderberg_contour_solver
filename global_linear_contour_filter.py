#!/usr/bin/env python3
"""Exact global linear filter for the two formal contour boundaries.

The terminal formal solver already guarantees the contact identifications.  This
module deliberately ignores internal interfaces and works only with:

* ``C``: the decorated boundary of the reference tile;
* ``E``: the decorated external boundary of the union of the three copies.

It solves the two maximal independent linear blocks available before chord
variables are introduced.

Angular block
-------------

All quantities are divided by pi.  The block contains simultaneously:

* the total-turn equations of C and E;
* the two local pole-sector inequalities;
* the principal bounds on every point-angle class;
* the principal bounds on every *actual boundary turn form* occurring on C and
  E, including the composite turns at the two external poles.

The last item is stronger than merely bounding the primitive Theta variables.
For example, the external-pole turn is a sum of three prototype turns minus
``2*pi`` and must itself lie strictly in ``(-pi, pi)`` for the external contour
to remain a regular decorated Jordan contour.

Length block
------------

Each formal curve variable receives a strictly positive geometric arc length.
Inversion and reflection preserve that length.  The reference perimeter fixes
the scale to one, and the external perimeter is required to have the same
length.  The latter identity follows from the three-copy contact topology: the
external boundary consists of the complementary arcs of the two covering
copies, hence has total length ``L(A)+L(B)=L(C)``.

Both strict systems are decided exactly by maximizing a rational common margin
``delta``.  The angular and length blocks are independent at this level, so
solving them separately is equivalent to solving one block-diagonal LP and is
both faster and more diagnostic.

Deliberate stopping point
-------------------------

Scalar area variables are not introduced into this LP.  Before signed arc
areas and chords are coupled through the determinant term in Chen's
concatenation law, constraints such as ``A_external = 3*A_inner`` are always
satisfiable by choosing an arbitrarily small positive area.

The next layers are compiled independently by
``global_metric_contour_model.py`` and ``joint_translation_z3.py``:

* positive arc lengths and chord-length inequalities;
* exact degree-two signed-area concatenation on both contours;
* ``A_external = 3*A_inner`` and rational isoperimetric bounds.

Keeping those polynomial layers outside this module preserves a small exact
rational LP and lets every level be enabled or disabled independently.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import external_boundary_constraints as external
import pole_angle_filter as poles
import rational_linear_program as rational_lp


SCHEMA_VERSION = "global-linear-contour-filter-v2"


@dataclass(frozen=True)
class RationalValue:
    numerator: int
    denominator: int

    @staticmethod
    def from_fraction(value: Fraction) -> "RationalValue":
        return RationalValue(value.numerator, value.denominator)

    def to_dict(self) -> Dict[str, object]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "decimal": self.numerator / self.denominator,
        }


@dataclass(frozen=True)
class LinearBlockAnalysis:
    feasible: bool
    status: str
    discard_reason: Optional[str]
    strict_margin: RationalValue
    variable_names: Tuple[str, ...]
    equality_count: int
    strict_inequality_family_count: int
    witness: Tuple[Tuple[str, RationalValue], ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "feasible": self.feasible,
            "status": self.status,
            "discard_reason": self.discard_reason,
            "strict_margin": self.strict_margin.to_dict(),
            "variable_names": list(self.variable_names),
            "equality_count": self.equality_count,
            "strict_inequality_family_count": self.strict_inequality_family_count,
            "witness": {
                name: value.to_dict() for name, value in self.witness
            },
        }


@dataclass(frozen=True)
class BoundaryTurnConstraint:
    boundary: str
    physical_point: str
    kind: str
    turn: external.AngleForm

    def to_dict(self) -> Dict[str, object]:
        return {
            "boundary": self.boundary,
            "physical_point": self.physical_point,
            "kind": self.kind,
            "turn": self.turn.to_text(),
            "principal_domain": "-pi < turn < pi",
        }


@dataclass(frozen=True)
class GlobalLinearContourAnalysis:
    angle_block_enabled: bool
    length_block_enabled: bool
    feasible: bool
    status: str
    discard_reason: Optional[str]
    strict_margin: RationalValue
    angle_block: LinearBlockAnalysis
    length_block: LinearBlockAnalysis
    theta_variables: Tuple[str, ...]
    kappa_variables: Tuple[str, ...]
    length_variables: Tuple[str, ...]
    boundary_turn_constraints: Tuple[BoundaryTurnConstraint, ...]
    inner_perimeter_coefficients: Tuple[Tuple[str, int], ...]
    outer_perimeter_coefficients: Tuple[Tuple[str, int], ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "configuration": {
                "angle_block_enabled": self.angle_block_enabled,
                "length_block_enabled": self.length_block_enabled,
            },
            "feasible": self.feasible,
            "status": self.status,
            "discard_reason": self.discard_reason,
            "strict_margin": self.strict_margin.to_dict(),
            "angle_block": self.angle_block.to_dict(),
            "length_block": self.length_block.to_dict(),
            "theta_variables": list(self.theta_variables),
            "kappa_variables": list(self.kappa_variables),
            "length_variables": list(self.length_variables),
            "boundary_turn_constraints": [
                constraint.to_dict() for constraint in self.boundary_turn_constraints
            ],
            "perimeter_normalization": {
                "reference_boundary": {
                    "coefficients": dict(self.inner_perimeter_coefficients),
                    "value": 1,
                },
                "external_boundary": {
                    "coefficients": dict(self.outer_perimeter_coefficients),
                    "value": 1,
                },
            },
            "scope": (
                "Exact rational feasibility of the maximal useful linear model "
                "on the decorated reference and external contours.  Internal "
                "interfaces are not re-solved: their compatibility belongs to "
                "the formal word/contact solver."
            ),
            "next_levels": [
                "polynomial chord/length layer with normalized perimeters",
                "signed arc-area layer with A_external = 3*A_inner",
                "future convex LP/SOCP/SDP relaxations for earlier rejection",
            ],
        }


def _is_kappa(name: str) -> bool:
    return name.startswith("Kappa[") or name.startswith("KappaClass")


def _unique_boundary_turn_constraints(
    system: external.JointBoundarySystem,
) -> Tuple[BoundaryTurnConstraint, ...]:
    """Return every nonduplicated point turn on C and E.

    BoundaryPath repeats its first point at the end.  The closing duplicate is
    omitted.  Constraints are otherwise kept pointwise even when two points
    share the same affine form, because the diagnostic should identify the
    physical location at which a principal-turn condition was imposed.
    """
    output = []
    for boundary in (system.inner_boundary, system.outer_boundary):
        for point in boundary.points[:-1]:
            output.append(
                BoundaryTurnConstraint(
                    boundary=boundary.name,
                    physical_point=point.physical_point,
                    kind=point.kind,
                    turn=point.turn,
                )
            )
    return tuple(output)


def _angle_variable_names(
    system: external.JointBoundarySystem,
    pole_analysis: poles.PoleAngleAnalysis,
    point_constraints: Sequence[BoundaryTurnConstraint],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    names = set()
    for equation in system.rotation_equations:
        names.update(equation.normalized_coefficients())
    for constraint in pole_analysis.constraints:
        names.update(constraint.coefficient_map())
    for constraint in point_constraints:
        names.update(name for name, _coefficient in constraint.turn.coefficients)
    ordered = tuple(sorted(names))
    theta = tuple(name for name in ordered if not _is_kappa(name))
    kappa = tuple(name for name in ordered if _is_kappa(name))
    return ordered, theta, kappa


def _solve_angle_block(
    system: external.JointBoundarySystem,
    pole_analysis: poles.PoleAngleAnalysis,
    point_constraints: Sequence[BoundaryTurnConstraint],
) -> Tuple[LinearBlockAnalysis, Tuple[str, ...], Tuple[str, ...]]:
    variable_names, theta_names, kappa_names = _angle_variable_names(
        system, pole_analysis, point_constraints
    )
    index = {name: position for position, name in enumerate(variable_names)}
    delta_index = len(variable_names)
    width = delta_index + 1

    def empty_row() -> list[Fraction]:
        return [Fraction(0) for _ in range(width)]

    def row_from_mapping(
        mapping: Mapping[str, int | Fraction],
    ) -> list[Fraction]:
        row = empty_row()
        for name, value in mapping.items():
            row[index[name]] = Fraction(value)
        return row

    inequalities: list[tuple[list[Fraction], Fraction]] = []

    # Primitive point classes are themselves physical point turns somewhere in
    # C, but retain their direct bounds as a defensive invariant and to cover
    # future profile encodings in which an angle class is only referenced by a
    # composite external point.
    for theta_name in theta_names:
        upper = empty_row()
        upper[index[theta_name]] = Fraction(1)
        upper[delta_index] = Fraction(1)
        inequalities.append((upper, Fraction(1)))

        lower = empty_row()
        lower[index[theta_name]] = Fraction(-1)
        lower[delta_index] = Fraction(1)
        inequalities.append((lower, Fraction(1)))

    # Every actual point turn form t = c.x + k*pi must satisfy
    # -1 + delta <= t/pi <= 1 - delta.
    for constraint in point_constraints:
        mapping = dict(constraint.turn.coefficients)
        upper = row_from_mapping(mapping)
        upper[delta_index] = Fraction(1)
        inequalities.append((upper, Fraction(1) - constraint.turn.pi_constant))

        lower = row_from_mapping(
            {name: -coefficient for name, coefficient in mapping.items()}
        )
        lower[delta_index] = Fraction(1)
        inequalities.append((lower, Fraction(1) + constraint.turn.pi_constant))

    equality_count = 0
    for equation in system.rotation_equations:
        row = row_from_mapping(equation.normalized_coefficients())
        rhs = Fraction(equation.normalized_rhs())
        inequalities.append((row, rhs))
        inequalities.append(([-value for value in row], -rhs))
        equality_count += 1

    for pole_constraint in pole_analysis.constraints:
        row = row_from_mapping(pole_constraint.coefficient_map())
        inequalities.append(([-value for value in row], Fraction(-1)))

    delta_lower = empty_row()
    delta_lower[delta_index] = Fraction(-1)
    inequalities.append((delta_lower, Fraction(0)))
    delta_upper = empty_row()
    delta_upper[delta_index] = Fraction(1)
    inequalities.append((delta_upper, Fraction(1)))

    objective = empty_row()
    objective[delta_index] = Fraction(1)
    result = rational_lp.maximize_free_variables(inequalities, objective)

    if result.status == "infeasible":
        return (
            LinearBlockAnalysis(
                feasible=False,
                status="infeasible_closed_angular_system",
                discard_reason=(
                    "The inner/outer total-turn equations, the pole-sector "
                    "constraints, and the principal bounds of the actual "
                    "decorated boundary points are mutually inconsistent."
                ),
                strict_margin=RationalValue(0, 1),
                variable_names=variable_names,
                equality_count=equality_count,
                strict_inequality_family_count=(
                    len(theta_names) + len(point_constraints)
                ),
                witness=(),
            ),
            theta_names,
            kappa_names,
        )
    if result.status != "optimal" or result.optimum is None:
        raise RuntimeError(f"Unexpected angular LP status: {result.status}")

    margin = result.optimum
    witness = tuple(
        (
            name,
            RationalValue.from_fraction(result.solution[index[name]]),
        )
        for name in variable_names
    )
    feasible = margin > 0
    return (
        LinearBlockAnalysis(
            feasible=feasible,
            status=(
                "feasible_with_strict_margin"
                if feasible
                else "only_degenerate_boundary_turns_feasible"
            ),
            discard_reason=(
                None
                if feasible
                else (
                    "The angular equations can be satisfied only when at least "
                    "one primitive or composite boundary turn reaches +/-pi."
                )
            ),
            strict_margin=RationalValue.from_fraction(margin),
            variable_names=variable_names,
            equality_count=equality_count,
            strict_inequality_family_count=(
                len(theta_names) + len(point_constraints)
            ),
            witness=witness,
        ),
        theta_names,
        kappa_names,
    )


def _perimeter_coefficients(
    boundary: external.BoundaryPath,
) -> Tuple[Tuple[str, int], ...]:
    counts = Counter(segment.literal.variable for segment in boundary.segments)
    return tuple(sorted(counts.items()))


def _solve_length_block(
    system: external.JointBoundarySystem,
) -> Tuple[
    LinearBlockAnalysis,
    Tuple[str, ...],
    Tuple[Tuple[str, int], ...],
    Tuple[Tuple[str, int], ...],
]:
    inner_coefficients = _perimeter_coefficients(system.inner_boundary)
    outer_coefficients = _perimeter_coefficients(system.outer_boundary)
    variable_names = tuple(
        sorted(
            {name for name, _count in inner_coefficients}
            | {name for name, _count in outer_coefficients}
        )
    )
    if not variable_names:
        analysis = LinearBlockAnalysis(
            feasible=False,
            status="empty_boundary",
            discard_reason="A closed tile contour cannot have zero curve segments.",
            strict_margin=RationalValue(0, 1),
            variable_names=(),
            equality_count=2,
            strict_inequality_family_count=0,
            witness=(),
        )
        return analysis, (), inner_coefficients, outer_coefficients

    index = {name: position for position, name in enumerate(variable_names)}
    delta_index = len(variable_names)
    width = delta_index + 1

    def empty_row() -> list[Fraction]:
        return [Fraction(0) for _ in range(width)]

    inequalities: list[tuple[list[Fraction], Fraction]] = []

    # L_i >= delta after the reference perimeter is normalized to one.
    for name in variable_names:
        row = empty_row()
        row[index[name]] = Fraction(-1)
        row[delta_index] = Fraction(1)
        inequalities.append((row, Fraction(0)))

    def add_perimeter_equality(
        coefficients: Iterable[Tuple[str, int]],
    ) -> None:
        row = empty_row()
        for name, count in coefficients:
            row[index[name]] = Fraction(count)
        inequalities.append((row, Fraction(1)))
        inequalities.append(([-value for value in row], Fraction(-1)))

    add_perimeter_equality(inner_coefficients)
    add_perimeter_equality(outer_coefficients)

    delta_lower = empty_row()
    delta_lower[delta_index] = Fraction(-1)
    inequalities.append((delta_lower, Fraction(0)))
    delta_upper = empty_row()
    delta_upper[delta_index] = Fraction(1)
    inequalities.append((delta_upper, Fraction(1)))

    objective = empty_row()
    objective[delta_index] = Fraction(1)
    result = rational_lp.maximize_free_variables(inequalities, objective)

    if result.status == "infeasible":
        analysis = LinearBlockAnalysis(
            feasible=False,
            status="incompatible_inner_outer_perimeters",
            discard_reason=(
                "No strictly positive assignment of formal curve lengths can "
                "normalize both the reference and external perimeters to the "
                "same value."
            ),
            strict_margin=RationalValue(0, 1),
            variable_names=variable_names,
            equality_count=2,
            strict_inequality_family_count=len(variable_names),
            witness=(),
        )
        return analysis, variable_names, inner_coefficients, outer_coefficients
    if result.status != "optimal" or result.optimum is None:
        raise RuntimeError(f"Unexpected length LP status: {result.status}")

    margin = result.optimum
    witness = tuple(
        (
            name,
            RationalValue.from_fraction(result.solution[index[name]]),
        )
        for name in variable_names
    )
    feasible = margin > 0
    analysis = LinearBlockAnalysis(
        feasible=feasible,
        status=(
            "feasible_with_strict_margin"
            if feasible
            else "only_zero_length_boundary_feasible"
        ),
        discard_reason=(
            None
            if feasible
            else (
                "The two perimeter equations can be satisfied only when at "
                "least one formal curve has zero geometric length."
            )
        ),
        strict_margin=RationalValue.from_fraction(margin),
        variable_names=variable_names,
        equality_count=2,
        strict_inequality_family_count=len(variable_names),
        witness=witness,
    )
    return analysis, variable_names, inner_coefficients, outer_coefficients


def _disabled_block(name: str) -> LinearBlockAnalysis:
    return LinearBlockAnalysis(
        feasible=True,
        status=f"{name}_disabled",
        discard_reason=None,
        strict_margin=RationalValue(0, 1),
        variable_names=(),
        equality_count=0,
        strict_inequality_family_count=0,
        witness=(),
    )


def analyze_global_linear_contours(
    system: external.JointBoundarySystem,
    pole_analysis: poles.PoleAngleAnalysis,
    *,
    enable_angle_block: bool = True,
    enable_length_block: bool = True,
) -> GlobalLinearContourAnalysis:
    """Solve the enabled exact linear blocks for C and E.

    The two blocks are independent.  They are separately configurable so an
    audit can measure their marginal effect or temporarily bypass one layer
    without changing the construction of the decorated boundaries.
    """
    point_constraints = _unique_boundary_turn_constraints(system)
    if enable_angle_block:
        angle_block, theta_names, kappa_names = _solve_angle_block(
            system, pole_analysis, point_constraints
        )
    else:
        angle_block = _disabled_block("angular_block")
        theta_names = ()
        kappa_names = ()

    if enable_length_block:
        (
            length_block,
            length_names,
            inner_coefficients,
            outer_coefficients,
        ) = _solve_length_block(system)
    else:
        length_block = _disabled_block("length_block")
        length_names = ()
        inner_coefficients = _perimeter_coefficients(system.inner_boundary)
        outer_coefficients = _perimeter_coefficients(system.outer_boundary)

    feasible = angle_block.feasible and length_block.feasible
    if enable_angle_block and not angle_block.feasible:
        status = "angular_block_reject"
        reason = angle_block.discard_reason
    elif enable_length_block and not length_block.feasible:
        status = "length_block_reject"
        reason = length_block.discard_reason
    elif not enable_angle_block and not enable_length_block:
        status = "all_linear_blocks_disabled"
        reason = None
    else:
        status = "feasible_with_strict_margin"
        reason = None

    enabled_margins = []
    if enable_angle_block:
        enabled_margins.append(Fraction(
            angle_block.strict_margin.numerator,
            angle_block.strict_margin.denominator,
        ))
    if enable_length_block:
        enabled_margins.append(Fraction(
            length_block.strict_margin.numerator,
            length_block.strict_margin.denominator,
        ))
    margin = min(enabled_margins) if enabled_margins else Fraction(0)

    return GlobalLinearContourAnalysis(
        angle_block_enabled=enable_angle_block,
        length_block_enabled=enable_length_block,
        feasible=feasible,
        status=status,
        discard_reason=reason,
        strict_margin=RationalValue.from_fraction(margin),
        angle_block=angle_block,
        length_block=length_block,
        theta_variables=theta_names,
        kappa_variables=kappa_names,
        length_variables=length_names,
        boundary_turn_constraints=point_constraints,
        inner_perimeter_coefficients=inner_coefficients,
        outer_perimeter_coefficients=outer_coefficients,
    )
