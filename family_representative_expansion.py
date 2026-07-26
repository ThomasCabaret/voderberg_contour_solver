#!/usr/bin/env python3
"""Explicit expansion policies for exact parametric word families.

Formal solving never calls this module.  It is an optional adapter from an
exact family AST to ordinary terminal ``SolverState`` objects for legacy
geometric filters.

The default policy is ``none``: only genuinely finite families are materialized.
Parametric families remain symbolic until a caller explicitly requests either
one assignment or a bounded range of assignments.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import parametric_expressions as expr
import symbolic_enumerator as base


SCHEMA_VERSION = "family-expansion-policy-v2"

POLICY_NONE = "none"
POLICY_MINIMUM = "minimum"
POLICY_FIXED = "fixed"
POLICY_RANGE = "range"
POLICY_CHOICES = (POLICY_NONE, POLICY_MINIMUM, POLICY_FIXED, POLICY_RANGE)


@dataclass(frozen=True)
class ExpansionPolicy:
    kind: str = POLICY_NONE
    fixed_exponent: int = 1
    maximum_exponent: int = 1
    max_specializations: Optional[int] = 10000

    def __post_init__(self) -> None:
        if self.kind not in POLICY_CHOICES:
            raise ValueError(
                f"Unknown family expansion policy {self.kind!r}; "
                f"expected one of {POLICY_CHOICES}"
            )
        if self.fixed_exponent < 0:
            raise ValueError("fixed_exponent must be nonnegative")
        if self.maximum_exponent < 0:
            raise ValueError("maximum_exponent must be nonnegative")
        if self.max_specializations is not None and self.max_specializations <= 0:
            raise ValueError("max_specializations must be positive or None")

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "fixed_exponent": self.fixed_exponent,
            "maximum_exponent": self.maximum_exponent,
            "max_specializations": self.max_specializations,
            "parametric_expansion_is_opt_in": True,
        }


@dataclass(frozen=True)
class ExpandedFamilyState:
    assignment: Tuple[Tuple[str, int], ...]
    state: base.SolverState

    def assignment_map(self) -> Dict[str, int]:
        return dict(self.assignment)


def exponent_assignment(
    minimums: Mapping[str, int],
    requested_value: int,
) -> Dict[str, int]:
    """Backward-compatible helper for one fixed exponent value."""
    if requested_value < 0:
        raise ValueError("requested_value must be nonnegative")
    return {
        name: max(int(requested_value), int(minimum))
        for name, minimum in minimums.items()
    }


def _assignment_sequence(
    minimums: Mapping[str, int],
    policy: ExpansionPolicy,
) -> Iterable[Dict[str, int]]:
    names = tuple(sorted(minimums))
    if not names:
        yield {}
        return
    if policy.kind == POLICY_NONE:
        return
    if policy.kind == POLICY_MINIMUM:
        yield {name: int(minimums[name]) for name in names}
        return
    if policy.kind == POLICY_FIXED:
        yield exponent_assignment(minimums, policy.fixed_exponent)
        return
    if policy.kind == POLICY_RANGE:
        ranges = []
        for name in names:
            minimum = int(minimums[name])
            if minimum > policy.maximum_exponent:
                return
            ranges.append(range(minimum, policy.maximum_exponent + 1))
        for values in product(*ranges):
            yield dict(zip(names, values))
        return
    raise AssertionError(policy.kind)


def expand_family(
    environment: Mapping[str, expr.WordExpression],
    minimums: Mapping[str, int],
    *,
    policy: ExpansionPolicy,
    depth: int = 0,
) -> Tuple[ExpandedFamilyState, ...]:
    """Materialize a finite family or explicitly selected parametric instances.

    A nonparametric environment always yields exactly one state.  A parametric
    environment yields no state under the default ``none`` policy.
    """
    output = []
    for assignment in _assignment_sequence(minimums, policy):
        if (
            policy.max_specializations is not None
            and len(output) >= policy.max_specializations
        ):
            raise ValueError(
                "Family expansion exceeded max_specializations="
                f"{policy.max_specializations}"
            )
        state = base.SolverState(
            equations=(),
            environment=tuple(
                (variable, expr.expand(value, assignment))
                for variable, value in sorted(environment.items())
            ),
            depth=depth,
        )
        output.append(
            ExpandedFamilyState(
                assignment=tuple(sorted(assignment.items())),
                state=state,
            )
        )
    return tuple(output)


def expand_environment(
    environment: Mapping[str, expr.WordExpression],
    minimums: Mapping[str, int],
    *,
    requested_value: int = 1,
    depth: int = 0,
) -> Tuple[base.SolverState, Dict[str, int]]:
    """Backward-compatible one-assignment API.

    New code should use :func:`expand_family` with an explicit policy.
    """
    policy = ExpansionPolicy(
        kind=POLICY_FIXED,
        fixed_exponent=requested_value,
        maximum_exponent=requested_value,
        max_specializations=1,
    )
    expanded = expand_family(
        environment,
        minimums,
        policy=policy,
        depth=depth,
    )
    if len(expanded) != 1:
        raise AssertionError("Fixed expansion must produce exactly one state")
    item = expanded[0]
    return item.state, item.assignment_map()
