#!/usr/bin/env python3
"""Specialized formal interpretation of terminal curve mappings.

This layer runs *after* generic word-equation solving and before any numeric
geometry.  It does not alter the Levi/Nielsen solver and it deliberately has no
import of the polygonal template compiler.

It turns terminal direct/reflected contact relations into explicit curve terms:

* a free representative remains a free curve parameter;
* a same-direction reflected self-relation becomes ``Straight(lambda)``;
* an endpoint-swapping reflected self-relation becomes
  ``Y · Mirror(Inverse(Y))`` with an internal join-angle parameter;
* relations between distinct variables are written with explicit ``Inverse``
  and ``Mirror`` constructors;
* direct endpoint swapping is only marked as a central symmetry already
  resolved by the generic word solver.

The separate module ``curve_template_constraints`` compiles these formal terms
for a chosen number of polygonal intermediate points.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import curve_relation_algebra as algebra


SCHEMA_VERSION = "curve-term-solution-v2"


@dataclass(frozen=True)
class CurveTermSolution:
    enabled: bool
    terms: Mapping[str, Mapping[str, object]]
    length_classes: Mapping[str, Mapping[str, object]]
    internal_angle_parameters: Tuple[str, ...]
    relation_analysis: algebra.CurveRelationAnalysis

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": self.enabled,
            "interpretation_stage": "after_generic_word_solving_before_geometry",
            "generic_word_solver_modified": False,
            "numeric_geometry_embedded": False,
            "terms": {name: dict(term) for name, term in self.terms.items()},
            "straight_length_classes": {
                name: dict(value) for name, value in self.length_classes.items()
            },
            "internal_curve_angle_parameters": list(self.internal_angle_parameters),
            "mirror_and_inverse_commute": True,
            "formal_relation_analysis": self.relation_analysis.to_dict(),
        }


def _term_text(term: Mapping[str, object]) -> str:
    kind = str(term["kind"])
    if kind == "curve_parameter":
        return str(term["name"])
    if kind == "straight":
        return f"Straight({term['length_class']})"
    if kind == "inverse":
        return f"Inverse({_term_text(term['operand'])})"
    if kind == "mirror":
        return f"Mirror({_term_text(term['operand'])})"
    if kind == "mirror_symmetric_join":
        half = str(term["half_curve_parameter"])
        angle = str(term["internal_angle_parameter"])
        return f"{half} ({angle}) Mirror(Inverse({half}))"
    if kind == "word_solver_resolved_central_symmetry":
        return str(term["variable"])
    raise ValueError(f"Unknown curve term kind {kind!r}")


def _with_text(term: Dict[str, object]) -> Dict[str, object]:
    term["text"] = _term_text(term)
    return term


def _apply_transform(
    term: Mapping[str, object],
    transform: algebra.CurveTransform,
) -> Dict[str, object]:
    output: Dict[str, object] = dict(term)
    if transform.reverse:
        output = _with_text({"kind": "inverse", "operand": output})
    if transform.mirror:
        output = _with_text({"kind": "mirror", "operand": output})
    return output


def solve_curve_terms(
    *,
    curve_variables: Sequence[str],
    occurrences: Sequence[object],
    terminal_mapping: Optional[Mapping[str, object]],
    enabled: bool = True,
) -> CurveTermSolution:
    """Return the geometry-independent formal curve-term solution.

    The result contains no requested edge count and no numeric parameter plan.
    Consequently the same audit output can be reused with any later value of
    ``--intermediate-points``.
    """

    variables = tuple(str(variable) for variable in curve_variables)
    relation_analysis = algebra.analyze_curve_relations(
        curve_variables=variables,
        occurrences=occurrences,
        terminal_mapping=terminal_mapping,
        enabled=enabled,
    )

    terms: Dict[str, Mapping[str, object]] = {}
    length_classes: Dict[str, Mapping[str, object]] = {}
    internal_angles = []

    for component in relation_analysis.components:
        component_index = component.index
        if component.mode == "straight":
            length_class = f"lambda{len(length_classes)}"
            representative_term = _with_text({
                "kind": "straight",
                "length_class": length_class,
                "positive": True,
            })
            length_classes[length_class] = {
                "positive": True,
                "component_index": component_index,
                "members": list(component.variables),
                "value_kind": "straight_segment_length",
            }
        elif component.mode == "endpoint_swapping_reflection":
            half = f"C{component_index}_half"
            angle = f"curve_angle{len(internal_angles)}"
            internal_angles.append(angle)
            representative_term = _with_text({
                "kind": "mirror_symmetric_join",
                "half_curve_parameter": half,
                "internal_angle_parameter": angle,
                "left": _with_text({"kind": "curve_parameter", "name": half}),
                "right": _with_text({
                    "kind": "mirror",
                    "operand": _with_text({
                        "kind": "inverse",
                        "operand": _with_text({"kind": "curve_parameter", "name": half}),
                    }),
                }),
            })
        elif component.mode == "half_turn":
            # The generic involutive word solver is responsible for splitting
            # X into Y (0) Y^-1.  This record is a validation marker, not a
            # second symbolic resolution rule.
            representative_term = _with_text({
                "kind": "word_solver_resolved_central_symmetry",
                "variable": component.representative,
                "expected_word_form": "Y (0) Inverse(Y)",
                "adds_new_symbolic_degrees_of_freedom": False,
            })
        else:
            representative_term = _with_text({
                "kind": "curve_parameter",
                "name": f"C{component_index}",
            })

        for variable in component.variables:
            term = _apply_transform(
                representative_term,
                component.transforms[variable],
            )
            term.update({
                "component_index": component_index,
                "source_component_mode": component.mode,
                "transform_from_representative": component.transforms[variable].label,
                "formal_value_not_numeric_constraint": True,
            })
            terms[variable] = term

    return CurveTermSolution(
        enabled=enabled,
        terms=terms,
        length_classes=length_classes,
        internal_angle_parameters=tuple(internal_angles),
        relation_analysis=relation_analysis,
    )
