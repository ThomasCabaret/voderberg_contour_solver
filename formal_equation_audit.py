#!/usr/bin/env python3
"""Structural audit for generated word-equation systems.

The audit is descriptive. It does not change solver results. It records the
initial occurrence structure of each placement system and the behavior of the
current bounded Levi/Nielsen exploration so that an appropriate complete
solver strategy can be chosen later.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import formal_cycle_cap
import symbolic_enumerator as base


@dataclass(frozen=True)
class SolverSearchAudit:
    terminal_states: Tuple[Tuple[base.SolverState, Tuple[str, ...]], ...]
    visited_states: int
    unique_depth_states: int
    duplicate_depth_states_skipped: int
    maximum_depth_reached: int
    depth_frontier_cut_count: int
    state_limit_hit: bool
    queue_size_when_state_limit_hit: int
    residual_equation_revisit_count: int
    structural_state_revisit_count: int
    terminal_count: int
    branch_counts: Tuple[Tuple[str, int], ...]
    cycle_unroll_cap_enabled: bool = False
    max_cycle_unrolls: Optional[int] = None
    cycle_unroll_pruned_state_count: int = 0
    cycle_unroll_cap_hit: bool = False
    initial_inconsistent: bool = False

    @property
    def search_truncated(self) -> bool:
        return (
            self.state_limit_hit
            or self.depth_frontier_cut_count > 0
            or self.cycle_unroll_cap_hit
        )

    @property
    def search_exhausted_within_current_graph(self) -> bool:
        return not self.initial_inconsistent and not self.search_truncated

    @property
    def has_terminal_profiles(self) -> bool:
        return self.terminal_count > 0

    @property
    def terminal_profile_set_may_be_incomplete(self) -> bool:
        return self.has_terminal_profiles and self.search_truncated

    @property
    def existence_unresolved_by_bounds(self) -> bool:
        return not self.has_terminal_profiles and self.search_truncated

    @property
    def terminal_profile_set_complete_within_current_graph(self) -> bool:
        return self.has_terminal_profiles and self.search_exhausted_within_current_graph

    @property
    def no_terminal_after_exhausting_current_graph(self) -> bool:
        return not self.has_terminal_profiles and self.search_exhausted_within_current_graph

    @property
    def search_outcome_class(self) -> str:
        if self.initial_inconsistent:
            return "initially_inconsistent"
        if self.search_truncated:
            return (
                "truncated_with_terminal_profiles"
                if self.has_terminal_profiles
                else "truncated_without_terminal_profiles"
            )
        return (
            "exhausted_with_terminal_profiles"
            if self.has_terminal_profiles
            else "exhausted_without_terminal_profiles"
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "initial_inconsistent": self.initial_inconsistent,
            "visited_states": self.visited_states,
            "unique_depth_states": self.unique_depth_states,
            "duplicate_depth_states_skipped": self.duplicate_depth_states_skipped,
            "maximum_depth_reached": self.maximum_depth_reached,
            "depth_frontier_cut_count": self.depth_frontier_cut_count,
            "state_limit_hit": self.state_limit_hit,
            "queue_size_when_state_limit_hit": self.queue_size_when_state_limit_hit,
            "residual_equation_revisit_count": self.residual_equation_revisit_count,
            "structural_state_revisit_count": self.structural_state_revisit_count,
            "reachable_cycle_evidence": bool(
                self.residual_equation_revisit_count
                or self.structural_state_revisit_count
            ),
            "cycle_unroll_cap_enabled": self.cycle_unroll_cap_enabled,
            "max_cycle_unrolls": self.max_cycle_unrolls,
            "cycle_unroll_pruned_state_count": self.cycle_unroll_pruned_state_count,
            "cycle_unroll_cap_hit": self.cycle_unroll_cap_hit,
            "terminal_count": self.terminal_count,
            "has_terminal_profiles": self.has_terminal_profiles,
            "branch_counts": dict(self.branch_counts),
            "search_truncated": self.search_truncated,
            "search_truncation_reasons": {
                "depth_bound": self.depth_frontier_cut_count > 0,
                "state_bound": self.state_limit_hit,
                "cycle_unroll_cap": self.cycle_unroll_cap_hit,
            },
            "search_exhausted_within_current_graph": self.search_exhausted_within_current_graph,
            "search_outcome_class": self.search_outcome_class,
            "terminal_profile_set_may_be_incomplete": self.terminal_profile_set_may_be_incomplete,
            "existence_unresolved_by_bounds": self.existence_unresolved_by_bounds,
            "terminal_profile_set_complete_within_current_graph": self.terminal_profile_set_complete_within_current_graph,
            "no_terminal_after_exhausting_current_graph": self.no_terminal_after_exhausting_current_graph,
        }


def _equation_variables(equation: base.Equation) -> set[str]:
    return {literal.variable for literal in equation.left + equation.right}


def _interaction_components(equations: Sequence[base.Equation]) -> List[List[str]]:
    variables = sorted({
        literal.variable
        for equation in equations
        for literal in equation.left + equation.right
    })
    adjacency: Dict[str, set[str]] = {variable: set() for variable in variables}
    for equation in equations:
        members = sorted(_equation_variables(equation))
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)

    components: List[List[str]] = []
    unseen = set(variables)
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component: List[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _occurrence_class(maximum: int) -> str:
    if maximum <= 1:
        return "linear"
    if maximum <= 2:
        return "quadratic"
    if maximum <= 3:
        return "cubic"
    return "higher_multiplicity"


def analyze_case_structure(case: base.PlacementCase) -> Dict[str, object]:
    equations = case.equations
    total_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    inverse_counts: Counter[str] = Counter()
    per_side_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"left": 0, "right": 0}
    )

    equation_data: List[Dict[str, object]] = []
    for equation_index, equation in enumerate(equations):
        for side_name, word in (("left", equation.left), ("right", equation.right)):
            for literal in word:
                total_counts[literal.variable] += 1
                per_side_counts[literal.variable][side_name] += 1
                if literal.inverse:
                    inverse_counts[literal.variable] += 1
                else:
                    positive_counts[literal.variable] += 1
        equation_data.append(
            {
                "equation_index": equation_index,
                "text": equation.to_text(),
                "left_length": len(equation.left),
                "right_length": len(equation.right),
                "total_length": len(equation.left) + len(equation.right),
                "variable_count": len(_equation_variables(equation)),
                "contains_inverse_literal": any(
                    literal.inverse for literal in equation.left + equation.right
                ),
            }
        )

    variables = sorted(total_counts)
    maximum = max(total_counts.values(), default=0)
    components = _interaction_components(equations)
    at_most_once_per_equation_side = all(
        sum(1 for literal in word if literal.variable == variable) <= 1
        for equation in equations
        for word in (equation.left, equation.right)
        for variable in variables
    )
    both_orientations = [
        variable
        for variable in variables
        if positive_counts[variable] and inverse_counts[variable]
    ]
    occurrence_class = _occurrence_class(maximum)
    quadratic = maximum <= 2

    if quadratic:
        recommendation = "quadratic_nielsen_graph_candidate"
        rationale = (
            "Every variable occurs at most twice in the initial system. "
            "A dedicated quadratic word-equation solver is the first complete-method candidate."
        )
    else:
        recommendation = "general_recompression_or_specialized_cycle_compression"
        rationale = (
            "At least one variable occurs more than twice. The current bounded Nielsen search "
            "is diagnostic; a general recompression method or a specialized proof for this "
            "generated subclass may be required for completeness."
        )

    return {
        "case_id": case.case_id,
        "equations": equation_data,
        "equation_count": len(equations),
        "variable_count": len(variables),
        "variables": variables,
        "total_literal_occurrences": sum(total_counts.values()),
        "occurrence_count_by_variable": dict(sorted(total_counts.items())),
        "positive_occurrence_count_by_variable": dict(sorted(positive_counts.items())),
        "inverse_occurrence_count_by_variable": dict(sorted(inverse_counts.items())),
        "side_occurrence_count_by_variable": {
            variable: per_side_counts[variable] for variable in variables
        },
        "maximum_variable_occurrence_count": maximum,
        "occurrence_class": occurrence_class,
        "is_linear_system": maximum <= 1,
        "is_quadratic_system": quadratic,
        "is_cubic_or_less": maximum <= 3,
        "at_most_once_per_equation_side": at_most_once_per_equation_side,
        "contains_involution": bool(inverse_counts),
        "variables_occurring_in_both_orientations": both_orientations,
        "interaction_component_count": len(components),
        "interaction_components": components,
        "single_variable_system": len(variables) == 1,
        "two_variable_system": len(variables) == 2,
        "recommended_complete_solver_family": recommendation,
        "recommendation_rationale": rationale,
    }


def _residual_key(equations: Sequence[base.Equation]) -> Tuple[str, ...]:
    return tuple(equation.to_text() for equation in equations)


def explore_case_with_audit(
    case: base.PlacementCase,
    *,
    max_depth: Optional[int],
    max_states: Optional[int],
    max_cycle_unrolls: Optional[int] = None,
) -> SolverSearchAudit:
    """Run the bounded search and collect independent truncation diagnostics."""
    initial = base.initial_solver_state(case)
    cycle_cap_enabled = max_cycle_unrolls is not None
    if initial is None:
        return SolverSearchAudit(
            terminal_states=(),
            visited_states=0,
            unique_depth_states=0,
            duplicate_depth_states_skipped=0,
            maximum_depth_reached=0,
            depth_frontier_cut_count=0,
            state_limit_hit=False,
            queue_size_when_state_limit_hit=0,
            residual_equation_revisit_count=0,
            structural_state_revisit_count=0,
            cycle_unroll_cap_enabled=cycle_cap_enabled,
            max_cycle_unrolls=max_cycle_unrolls,
            cycle_unroll_pruned_state_count=0,
            cycle_unroll_cap_hit=False,
            terminal_count=0,
            branch_counts=(),
            initial_inconsistent=True,
        )

    initial_history = formal_cycle_cap.CycleVisitHistory.start(
        _residual_key(initial.equations)
    )
    queue = deque([(initial, tuple(), initial_history)])
    seen_at_depth = set()
    seen_structural: set[Tuple[Tuple[base.Equation, ...], Tuple[Tuple[str, base.Word], ...]]] = set()
    seen_residual: set[Tuple[base.Equation, ...]] = set()
    terminal_states: List[Tuple[base.SolverState, Tuple[str, ...]]] = []
    branch_counts: Counter[str] = Counter()
    visited_states = 0
    unique_depth_states = 0
    duplicates = 0
    max_depth_reached = 0
    depth_frontier_cut_count = 0
    state_limit_hit = False
    queue_size_when_state_limit_hit = 0
    residual_revisits = 0
    structural_revisits = 0
    cycle_unroll_pruned_state_count = 0

    while queue:
        if max_states is not None and visited_states >= max_states:
            state_limit_hit = True
            queue_size_when_state_limit_hit = len(queue)
            break

        state, derivation, cycle_history = queue.popleft()
        visited_states += 1
        max_depth_reached = max(max_depth_reached, state.depth)

        cycle_signature = cycle_history.signature() if cycle_cap_enabled else ()
        depth_key = (state.depth, state.equations, state.environment, cycle_signature)
        if depth_key in seen_at_depth:
            duplicates += 1
            continue
        seen_at_depth.add(depth_key)
        unique_depth_states += 1

        structural_key = (state.equations, state.environment)
        if structural_key in seen_structural:
            structural_revisits += 1
        else:
            seen_structural.add(structural_key)

        if state.equations in seen_residual:
            residual_revisits += 1
        else:
            seen_residual.add(state.equations)

        if not state.equations:
            terminal_states.append((state, derivation))
            continue

        if max_depth is not None and state.depth >= max_depth:
            depth_frontier_cut_count += 1
            continue

        environment = state.environment_map()
        for branch_name, substitution in base.branch_substitutions(
            state.equations, environment
        ):
            branch_counts[branch_name] += 1
            child = base.advance_state(state, substitution)
            if child is None:
                continue
            child_history = cycle_history
            if child.equations:
                child_history, capped = cycle_history.advance(
                    _residual_key(child.equations), max_cycle_unrolls
                )
                if capped:
                    cycle_unroll_pruned_state_count += 1
                    continue
                if child_history is None:
                    raise RuntimeError("Cycle history unexpectedly missing")
            queue.append((child, derivation + (branch_name,), child_history))

    return SolverSearchAudit(
        terminal_states=tuple(terminal_states),
        visited_states=visited_states,
        unique_depth_states=unique_depth_states,
        duplicate_depth_states_skipped=duplicates,
        maximum_depth_reached=max_depth_reached,
        depth_frontier_cut_count=depth_frontier_cut_count,
        state_limit_hit=state_limit_hit,
        queue_size_when_state_limit_hit=queue_size_when_state_limit_hit,
        residual_equation_revisit_count=residual_revisits,
        structural_state_revisit_count=structural_revisits,
        cycle_unroll_cap_enabled=cycle_cap_enabled,
        max_cycle_unrolls=max_cycle_unrolls,
        cycle_unroll_pruned_state_count=cycle_unroll_pruned_state_count,
        cycle_unroll_cap_hit=cycle_unroll_pruned_state_count > 0,
        terminal_count=len(terminal_states),
        branch_counts=tuple(sorted(branch_counts.items())),
    )


def summarize_case_audits(case_records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    occurrence_classes: Counter[str] = Counter()
    variable_counts: Counter[str] = Counter()
    maximum_occurrences: Counter[str] = Counter()
    recommendations: Counter[str] = Counter()
    terminal_profiles_by_class: Counter[str] = Counter()
    cases_with_terminals_by_class: Counter[str] = Counter()
    truncated_by_class: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    outcome_counts_by_occurrence_class: Dict[str, Counter[str]] = defaultdict(Counter)

    for record in case_records:
        structure = record["structure"]
        search = record["bounded_search"]
        occurrence_class = str(structure["occurrence_class"])
        outcome = str(search["search_outcome_class"])
        occurrence_classes[occurrence_class] += 1
        outcome_counts[outcome] += 1
        outcome_counts_by_occurrence_class[occurrence_class][outcome] += 1
        variable_counts[str(structure["variable_count"])] += 1
        maximum_occurrences[str(structure["maximum_variable_occurrence_count"])] += 1
        recommendations[str(structure["recommended_complete_solver_family"])] += 1
        terminal_count = int(search["terminal_count"])
        terminal_profiles_by_class[occurrence_class] += terminal_count
        if terminal_count:
            cases_with_terminals_by_class[occurrence_class] += 1
        if bool(search["search_truncated"]):
            truncated_by_class[occurrence_class] += 1

    quadratic_cases = sum(
        1 for record in case_records if record["structure"]["is_quadratic_system"]
    )
    initial_inconsistent_cases = outcome_counts["initially_inconsistent"]
    cases_with_terminals = sum(
        1 for record in case_records if record["bounded_search"]["terminal_count"] > 0
    )
    involutive_cases = sum(
        1 for record in case_records if record["structure"]["contains_involution"]
    )
    once_per_side_cases = sum(
        1 for record in case_records
        if record["structure"]["at_most_once_per_equation_side"]
    )
    truncated_cases = sum(
        1 for record in case_records if record["bounded_search"]["search_truncated"]
    )
    state_limited_cases = sum(
        1 for record in case_records if record["bounded_search"]["state_limit_hit"]
    )
    depth_limited_cases = sum(
        1
        for record in case_records
        if record["bounded_search"]["depth_frontier_cut_count"] > 0
    )
    cycle_capped_cases = sum(
        1
        for record in case_records
        if record["bounded_search"].get("cycle_unroll_cap_hit", False)
    )
    cycle_pruned_states = sum(
        int(record["bounded_search"].get("cycle_unroll_pruned_state_count", 0))
        for record in case_records
    )
    cycle_capped_with_terminals = sum(
        1
        for record in case_records
        if record["bounded_search"].get("cycle_unroll_cap_hit", False)
        and record["bounded_search"]["terminal_count"] > 0
    )
    cycle_capped_without_terminals = cycle_capped_cases - cycle_capped_with_terminals
    exhausted_cases = sum(
        1
        for record in case_records
        if record["bounded_search"]["search_exhausted_within_current_graph"]
    )

    truncated_with_terminals = outcome_counts["truncated_with_terminal_profiles"]
    truncated_without_terminals = outcome_counts["truncated_without_terminal_profiles"]
    exhausted_with_terminals = outcome_counts["exhausted_with_terminal_profiles"]
    exhausted_without_terminals = outcome_counts["exhausted_without_terminal_profiles"]
    submitted_to_branching_solver = len(case_records) - initial_inconsistent_cases

    return {
        "placement_system_count": len(case_records),
        "quadratic_system_count": quadratic_cases,
        "nonquadratic_system_count": len(case_records) - quadratic_cases,
        "contains_involution_case_count": involutive_cases,
        "at_most_once_per_equation_side_case_count": once_per_side_cases,
        "initially_inconsistent_case_count": initial_inconsistent_cases,
        "submitted_to_branching_solver_case_count": submitted_to_branching_solver,
        "cases_with_terminal_profiles_within_bounds": cases_with_terminals,
        "cases_without_terminal_profiles_within_bounds": len(case_records) - cases_with_terminals,
        "search_outcome_counts": dict(sorted(outcome_counts.items())),
        "search_outcome_counts_by_occurrence_class": {
            occurrence_class: dict(sorted(counts.items()))
            for occurrence_class, counts in sorted(outcome_counts_by_occurrence_class.items())
        },
        "truncated_with_terminal_profiles_case_count": truncated_with_terminals,
        "truncated_without_terminal_profiles_case_count": truncated_without_terminals,
        "exhausted_with_terminal_profiles_case_count": exhausted_with_terminals,
        "exhausted_without_terminal_profiles_case_count": exhausted_without_terminals,
        "terminal_profile_sets_complete_within_current_graph_case_count": exhausted_with_terminals,
        "terminal_profile_sets_potentially_incomplete_due_to_bounds_case_count": truncated_with_terminals,
        "solution_existence_unresolved_due_to_bounds_case_count": truncated_without_terminals,
        "no_terminal_after_exhausting_current_graph_case_count": exhausted_without_terminals,
        "occurrence_class_counts": dict(sorted(occurrence_classes.items())),
        "variable_count_histogram": dict(sorted(variable_counts.items(), key=lambda item: int(item[0]))),
        "maximum_occurrence_histogram": dict(sorted(maximum_occurrences.items(), key=lambda item: int(item[0]))),
        "recommended_solver_family_counts": dict(sorted(recommendations.items())),
        "cases_with_terminal_profiles_by_occurrence_class": dict(sorted(cases_with_terminals_by_class.items())),
        "terminal_profile_instances_by_occurrence_class": dict(sorted(terminal_profiles_by_class.items())),
        "bounded_search_truncated_case_count": truncated_cases,
        "bounded_search_state_limited_case_count": state_limited_cases,
        "bounded_search_depth_limited_case_count": depth_limited_cases,
        "bounded_search_cycle_capped_case_count": cycle_capped_cases,
        "bounded_search_cycle_capped_with_terminal_profiles_case_count": cycle_capped_with_terminals,
        "bounded_search_cycle_capped_without_terminal_profiles_case_count": cycle_capped_without_terminals,
        "bounded_search_cycle_pruned_state_count": cycle_pruned_states,
        "bounded_search_exhausted_case_count": exhausted_cases,
        "truncated_case_count_by_occurrence_class": dict(sorted(truncated_by_class.items())),
        "interpretation": {
            "quadratic": "Every initial variable occurs at most twice across all equation sides.",
            "search_exhausted": "The current transition-system exploration emptied its queue without hitting either configured bound.",
            "search_truncated_with_terminals": "At least one terminal profile was found, but more terminal profiles may exist beyond the configured bounds.",
            "search_truncated_without_terminals": "No terminal profile was found, and existence remains unresolved because a configured depth/state bound or the temporary cycle-unroll cap was hit.",
            "cycle_unroll_cap": "A temporary anti-echo policy prunes a branch after the configured number of returns to the same residual equation system. It is not parametric cycle recognition and every hit is reported as truncation.",
            "exhausted_without_terminals": "No terminal profile is reachable in the explored transition graph. This is not a general completeness theorem for arbitrary word equations.",
            "method_selection": "Try a complete quadratic Nielsen-graph solver for quadratic systems before implementing general recompression.",
        },
    }
