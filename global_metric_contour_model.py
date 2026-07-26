#!/usr/bin/env python3
"""Solver-independent metric model for the inner and outer formal contours.

This module is the bridge between the exact rational linear contour filter and
polynomial backends such as Z3/NLSAT.  It deliberately contains no solver code
and no optional dependency.

For each formal curve variable X it introduces the semantic quantities needed
by the next two onion layers:

* L[X] > 0: geometric arc length;
* D[X] in R^2: endpoint chord in the canonical local frame;
* S[X] in R: signed area between the oriented arc and its chord.

The ordered occurrence model records, for every segment of the reference and
external contours:

* the exact phase applied to D[X];
* whether the chord is conjugated by a reflected copy;
* the sign applied to S[X] under reversal/reflection.

A polynomial backend can therefore compile:

    |D[X]| <= L[X]
    perimeter(inner) = perimeter(outer) = 1

and, at the signed-area layer, Chen's degree-two concatenation law:

    area(XY) = area(X) + area(Y) + 1/2 det(D(X), D(Y)).

No internal contact interface is reconstructed here.  Its compatibility is
already encoded by the terminal formal mappings; only the two closed contours
and their global relation are modelled.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple

import external_boundary_constraints as external


SCHEMA_VERSION = "global-metric-contour-model-v1"


@dataclass(frozen=True)
class MetricSegmentOccurrence:
    variable: str
    phase: external.AngleForm
    conjugated_chord: bool
    signed_arc_area_sign: int
    occurrence_index: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "variable": self.variable,
            "phase": self.phase.to_text(),
            "conjugated_chord": self.conjugated_chord,
            "signed_arc_area_sign": self.signed_arc_area_sign,
            "occurrence_index": self.occurrence_index,
        }


@dataclass(frozen=True)
class MetricBoundaryModel:
    name: str
    segments: Tuple[MetricSegmentOccurrence, ...]
    perimeter_coefficients: Tuple[Tuple[str, int], ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "segments": [segment.to_dict() for segment in self.segments],
            "perimeter_coefficients": dict(self.perimeter_coefficients),
        }


@dataclass(frozen=True)
class GlobalMetricContourModel:
    curve_variables: Tuple[str, ...]
    inner_boundary: MetricBoundaryModel
    outer_boundary: MetricBoundaryModel

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "curve_variables": list(self.curve_variables),
            "inner_boundary": self.inner_boundary.to_dict(),
            "outer_boundary": self.outer_boundary.to_dict(),
            "scope": (
                "Metric invariants of the decorated reference and external "
                "contours only; formal contact interfaces are not re-solved."
            ),
        }


def _perimeter_coefficients(
    boundary: external.BoundaryPath,
) -> Tuple[Tuple[str, int], ...]:
    counts = Counter(segment.literal.variable for segment in boundary.segments)
    return tuple(sorted(counts.items()))


def _build_boundary_model(
    boundary: external.BoundaryPath,
    curve_turn_solution: external.CurveTurnSolution,
) -> MetricBoundaryModel:
    heading = external.AngleForm.zero()
    occurrences = []

    for index, segment in enumerate(boundary.segments):
        kappa = f"Kappa[{segment.literal.variable}]"
        phase = heading
        if segment.literal.inverse:
            phase = phase.add_term(kappa, -segment.mirror_sign)
        phase = external.apply_curve_turn_solution(phase, curve_turn_solution)

        # Reversal changes signed arc area, and reflection changes orientation.
        # Their product is therefore the exact coefficient multiplying S[X].
        area_sign = segment.traversal_sign * segment.mirror_sign
        if area_sign not in (-1, 1):
            raise ValueError("A metric segment area sign must be +/-1")

        occurrences.append(
            MetricSegmentOccurrence(
                variable=segment.literal.variable,
                phase=phase,
                conjugated_chord=segment.conjugated_chord,
                signed_arc_area_sign=area_sign,
                occurrence_index=segment.occurrence_index,
            )
        )

        heading = heading.add_term(kappa, segment.physical_turn_sign)
        heading = heading.add(boundary.points[index + 1].turn)
        heading = external.apply_curve_turn_solution(heading, curve_turn_solution)

    return MetricBoundaryModel(
        name=boundary.name,
        segments=tuple(occurrences),
        perimeter_coefficients=_perimeter_coefficients(boundary),
    )


def build_global_metric_contour_model(
    system: external.JointBoundarySystem,
) -> GlobalMetricContourModel:
    inner = _build_boundary_model(
        system.inner_boundary, system.curve_turn_solution
    )
    outer = _build_boundary_model(
        system.outer_boundary, system.curve_turn_solution
    )
    variables = tuple(
        sorted(
            {segment.variable for segment in inner.segments}
            | {segment.variable for segment in outer.segments}
        )
    )
    if not variables:
        raise ValueError("A metric contour model requires at least one curve variable")
    return GlobalMetricContourModel(
        curve_variables=variables,
        inner_boundary=inner,
        outer_boundary=outer,
    )
