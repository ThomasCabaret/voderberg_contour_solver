#!/usr/bin/env python3
"""Exact partial word-equation solver with an explicit supported frontier.

The solver first builds the cycle-preserving residual Nielsen graph from
``parametric_graph``.  A result is called exact only when that graph is fully
constructed.  The downstream-supported language is deliberately smaller than
EDT0L: finite families plus cycles whose complete morphism only adds fixed
left/right contexts around one evolving variable.  Several such components
along a path naturally produce nested powers.

More complicated useful SCCs are retained as explicit unsupported frontiers
and are never silently unrolled.  Other supported branches of the same graph
remain available.  Truncated graphs are reported as
``UNRESOLVED_GRAPH_LIMIT``.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from itertools import product
import json
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import parametric_expressions as expr
import parametric_graph as graph_layer
import symbolic_enumerator as base


SCHEMA_VERSION = "exact-partial-word-solver-v2"

EXACT_UNSAT = "EXACT_UNSAT"
EXACT_FINITE = "EXACT_FINITE"
EXACT_POWER = "EXACT_POWER"
EXACT_NESTED_POWER = "EXACT_NESTED_POWER"
EXACT_GRAPH_UNSUPPORTED = "EXACT_GRAPH_UNSUPPORTED_FAMILY_LANGUAGE"
EXACT_MIXED_SUPPORTED_AND_UNSUPPORTED = (
    "EXACT_SUPPORTED_FAMILIES_WITH_UNSUPPORTED_FRONTIER"
)
UNRESOLVED_GRAPH_LIMIT = "UNRESOLVED_GRAPH_LIMIT"
UNRESOLVED_FAMILY_LIMIT = "UNRESOLVED_FAMILY_LIMIT"


@dataclass(frozen=True)
class ExactFormalFamily:
    family_id: int
    kind: str
    environment: Tuple[Tuple[str, expr.WordExpression], ...]
    a_expression: expr.WordExpression
    b_expression: expr.WordExpression
    exponent_minimums: Tuple[Tuple[str, int], ...]
    trace: Tuple[str, ...]

    @property
    def parametric(self) -> bool:
        return bool(self.exponent_minimums)

    @property
    def minimum_assignment(self) -> Dict[str, int]:
        return dict(self.exponent_minimums)

    def to_profile_dict(self) -> Dict[str, object]:
        """Compact profile annotation; the complete AST stays in the formal audit."""
        return {
            "family_id": self.family_id,
            "kind": self.kind,
            "complete_family": True,
            "downstream_supported": True,
            "parametric": self.parametric,
            "primitive_finite": not self.parametric,
            "A": {"text": expr.to_text(self.a_expression)},
            "B": {"text": expr.to_text(self.b_expression)},
            "exponent_parameters": {
                name: {"minimum": minimum}
                for name, minimum in self.exponent_minimums
            },
            "expanded_for_downstream": False,
            "full_ast_location": "formal_equation_audit.json / cases / exact_formal_solver / families",
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "family_id": self.family_id,
            "kind": self.kind,
            "complete_family": True,
            "downstream_supported": True,
            "parametric": self.parametric,
            "primitive_finite": not self.parametric,
            "A": {
                "text": expr.to_text(self.a_expression),
                "ast": expr.to_dict(self.a_expression),
            },
            "B": {
                "text": expr.to_text(self.b_expression),
                "ast": expr.to_dict(self.b_expression),
            },
            "environment": {
                variable: {
                    "text": expr.to_text(value),
                    "ast": expr.to_dict(value),
                }
                for variable, value in self.environment
            },
            "exponent_parameters": {
                name: {"minimum": minimum}
                for name, minimum in self.exponent_minimums
            },
            "expanded_for_downstream": False,
            "trace": list(self.trace),
        }


@dataclass(frozen=True)
class UnsupportedFormalFamily:
    """A complete graph frontier whose dynamic language is not compiled yet.

    The entry environment is retained symbolically, but the SCC is never
    traversed or unrolled.  This lets other finite or supported parametric
    branches of the same placement case survive independently.
    """

    frontier_id: int
    component_id: int
    entry_node: int
    component_nodes: Tuple[int, ...]
    reason: str
    entry_environment: Tuple[Tuple[str, expr.WordExpression], ...]
    trace: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "frontier_id": self.frontier_id,
            "kind": "UNSUPPORTED_DYNAMIC_SCC",
            "complete_graph_frontier": True,
            "expanded": False,
            "component_id": self.component_id,
            "entry_node": self.entry_node,
            "component_nodes": list(self.component_nodes),
            "reason": self.reason,
            "entry_environment": {
                variable: {
                    "text": expr.to_text(value),
                    "ast": expr.to_dict(value),
                }
                for variable, value in self.entry_environment
            },
            "trace": list(self.trace),
        }


@dataclass(frozen=True)
class ExactCaseResult:
    case_id: int
    status: str
    graph_complete: bool
    graph_summary: Mapping[str, object]
    families: Tuple[ExactFormalFamily, ...]
    unsupported_families: Tuple[UnsupportedFormalFamily, ...]
    unsupported_reasons: Tuple[str, ...]
    suppressed_finite_specialization_count: int = 0

    @property
    def exact(self) -> bool:
        return self.status.startswith("EXACT_")

    @property
    def downstream_supported(self) -> bool:
        return bool(self.families)

    @property
    def complete_family_language_compiled(self) -> bool:
        return self.graph_complete and not self.unsupported_families

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "case_id": self.case_id,
            "status": self.status,
            "exact": self.exact,
            "graph_complete": self.graph_complete,
            "downstream_supported": self.downstream_supported,
            "complete_family_language_compiled": self.complete_family_language_compiled,
            "graph": dict(self.graph_summary),
            "family_count": len(self.families),
            "finite_family_count": sum(not family.parametric for family in self.families),
            "parametric_family_count": sum(family.parametric for family in self.families),
            "families": [family.to_dict() for family in self.families],
            "unsupported_family_count": len(self.unsupported_families),
            "unsupported_families": [
                family.to_dict() for family in self.unsupported_families
            ],
            "unsupported_reasons": list(self.unsupported_reasons),
            "suppressed_finite_specialization_count": (
                self.suppressed_finite_specialization_count
            ),
            "legacy_bounded_search_used": False,
        }


@dataclass
class _NameFactory:
    word_index: int = 0
    exponent_index: int = 0

    def clone(self) -> "_NameFactory":
        return _NameFactory(self.word_index, self.exponent_index)

    def word(self) -> str:
        value = f"Q{self.word_index}"
        self.word_index += 1
        return value

    def exponent(self) -> str:
        value = f"n{self.exponent_index}"
        self.exponent_index += 1
        return value


@dataclass(frozen=True)
class _CycleInfo:
    component: Tuple[int, ...]
    internal_edge_by_source: Mapping[int, graph_layer.GraphEdge]


class _FamilyLimitReached(RuntimeError):
    pass


def _substitute_base_word(
    word: base.Word,
    substitution: Mapping[str, base.Word],
) -> base.Word:
    return base.substitute_word(word, substitution)


def _compose_morphisms(
    first: Mapping[str, base.Word],
    second: Mapping[str, base.Word],
) -> Dict[str, base.Word]:
    return {
        variable: _substitute_base_word(word, second)
        for variable, word in first.items()
    }


def _identity_morphism(variables: Iterable[str]) -> Dict[str, base.Word]:
    return {variable: (base.Literal(variable),) for variable in variables}


def _adjacency(graph: graph_layer.DerivationGraph) -> Dict[int, List[graph_layer.GraphEdge]]:
    output: Dict[int, List[graph_layer.GraphEdge]] = defaultdict(list)
    for edge in graph.edges:
        output[edge.source].append(edge)
    for edges in output.values():
        edges.sort(key=lambda edge: edge.edge_id)
    return output


def _nodes_reaching_terminal(graph: graph_layer.DerivationGraph) -> set[int]:
    reverse: Dict[int, List[int]] = defaultdict(list)
    for edge in graph.edges:
        reverse[edge.target].append(edge.source)
    queue = deque(node.node_id for node in graph.nodes if node.terminal)
    seen = set(queue)
    while queue:
        node = queue.popleft()
        for parent in reverse.get(node, ()):  # pragma: no branch - tiny graph helper
            if parent not in seen:
                seen.add(parent)
                queue.append(parent)
    return seen


def _reachable_from_root(graph: graph_layer.DerivationGraph) -> set[int]:
    adjacency = _adjacency(graph)
    queue = deque([graph.root])
    seen = {graph.root}
    while queue:
        node = queue.popleft()
        for edge in adjacency.get(node, ()):  # pragma: no branch
            if edge.target not in seen:
                seen.add(edge.target)
                queue.append(edge.target)
    return seen


def _component_maps(
    graph: graph_layer.DerivationGraph,
) -> Tuple[Dict[int, int], Dict[int, Tuple[int, ...]]]:
    components = graph_layer.strongly_connected_components(graph.nodes, graph.edges)
    component_of: Dict[int, int] = {}
    by_id: Dict[int, Tuple[int, ...]] = {}
    for index, members in enumerate(components):
        ordered = tuple(sorted(members))
        by_id[index] = ordered
        for member in ordered:
            component_of[member] = index
    return component_of, by_id


def _is_cyclic_component(
    component: Sequence[int],
    graph: graph_layer.DerivationGraph,
) -> bool:
    members = set(component)
    if len(members) > 1:
        return True
    node = next(iter(members))
    return any(edge.source == node and edge.target == node for edge in graph.edges)


def _simple_cycle_info(
    component: Sequence[int],
    graph: graph_layer.DerivationGraph,
) -> Optional[_CycleInfo]:
    members = set(component)
    internal = [
        edge for edge in graph.edges
        if edge.source in members and edge.target in members
    ]
    indegree = Counter(edge.target for edge in internal)
    outdegree = Counter(edge.source for edge in internal)
    if len(internal) != len(members):
        return None
    if any(indegree[node] != 1 or outdegree[node] != 1 for node in members):
        return None
    return _CycleInfo(
        component=tuple(sorted(members)),
        internal_edge_by_source={edge.source: edge for edge in internal},
    )


def _ordered_cycle_edges(info: _CycleInfo, entry: int) -> Tuple[graph_layer.GraphEdge, ...]:
    output = []
    current = entry
    while True:
        edge = info.internal_edge_by_source[current]
        output.append(edge)
        current = edge.target
        if current == entry:
            return tuple(output)
        if len(output) > len(info.component):
            raise RuntimeError("Invalid deterministic cycle")


def _cycle_repeat_mapping(
    graph: graph_layer.DerivationGraph,
    info: _CycleInfo,
    entry: int,
    exponent_name: str,
) -> Tuple[Optional[Dict[str, expr.WordExpression]], Optional[str]]:
    edges = _ordered_cycle_edges(info, entry)
    if any(edge.local_parameters for edge in edges):
        return None, "cycle introduces fresh word parameters on every traversal"
    variables = graph_layer.variables_in_equations(graph.nodes[entry].equations)
    morphism = _identity_morphism(variables)
    for edge in edges:
        morphism = _compose_morphisms(morphism, edge.morphism_map())

    fixed = {
        variable for variable in variables
        if morphism.get(variable) == (base.Literal(variable),)
    }
    output: Dict[str, expr.WordExpression] = {}
    for variable in variables:
        word = morphism[variable]
        if variable in fixed:
            output[variable] = expr.atom(variable)
            continue
        evolving_positions = [
            index for index, literal in enumerate(word)
            if literal.variable not in fixed
        ]
        if len(evolving_positions) != 1:
            return None, (
                f"cycle maps {variable} to {base.word_to_text(word)}, which contains "
                "zero or several evolving variable occurrences"
            )
        position = evolving_positions[0]
        literal = word[position]
        if literal.variable != variable or literal.inverse:
            return None, (
                f"cycle maps {variable} through a permutation, inversion or another "
                f"evolving variable ({literal.to_text()})"
            )
        left = expr.from_word(word[:position])
        right = expr.from_word(word[position + 1 :])
        pieces: List[expr.WordExpression] = []
        if not isinstance(left, expr.Concat) or left.parts:
            pieces.append(expr.Repeat(left, exponent_name, minimum=0))
        pieces.append(expr.atom(variable))
        if not isinstance(right, expr.Concat) or right.parts:
            pieces.append(expr.Repeat(right, exponent_name, minimum=0))
        output[variable] = expr.concat(*pieces)
    return output, None


def _commutation_mapping(
    graph: graph_layer.DerivationGraph,
    node_id: int,
    names: _NameFactory,
) -> Optional[Tuple[Dict[str, expr.WordExpression], Dict[str, int], str]]:
    node = graph.nodes[node_id]
    if len(node.equations) != 1:
        return None
    equation = node.equations[0]
    if len(equation.left) != 2 or len(equation.right) != 2:
        return None
    x, y = equation.left
    if any(literal.inverse for literal in equation.left + equation.right):
        return None
    if equation.right != (y, x) or x.variable == y.variable:
        return None
    adjacency = _adjacency(graph)
    if not any(graph.nodes[edge.target].terminal for edge in adjacency.get(node_id, ())):
        return None
    root = names.word()
    exponent_x = names.exponent()
    exponent_y = names.exponent()
    mapping = {
        x.variable: expr.Repeat(expr.atom(root), exponent_x, minimum=1),
        y.variable: expr.Repeat(expr.atom(root), exponent_y, minimum=1),
    }
    return mapping, {exponent_x: 1, exponent_y: 1}, "commutation theorem X Y = Y X"


def _edge_expression_mapping(
    edge: graph_layer.GraphEdge,
    names: _NameFactory,
) -> Dict[str, expr.WordExpression]:
    local_names = {
        local: names.word() for local in edge.local_parameters
    }
    output: Dict[str, expr.WordExpression] = {}
    for variable, word in edge.morphism:
        parts = []
        for literal in word:
            name = local_names.get(literal.variable, literal.variable)
            parts.append(expr.atom(name, literal.inverse))
        output[variable] = expr.concat(*parts)
    return output


def _apply_mapping(
    environment: Mapping[str, expr.WordExpression],
    mapping: Mapping[str, expr.WordExpression],
) -> Dict[str, expr.WordExpression]:
    return {
        variable: expr.substitute(value, mapping)
        for variable, value in environment.items()
    }


def _partial_cycle_edges(
    info: _CycleInfo,
    entry: int,
    target_source: int,
) -> Tuple[graph_layer.GraphEdge, ...]:
    output = []
    current = entry
    while current != target_source:
        edge = info.internal_edge_by_source[current]
        output.append(edge)
        current = edge.target
        if len(output) > len(info.component):
            raise RuntimeError("Cycle exit source is not reachable from entry")
    return tuple(output)


def _canonicalize_family_environment(
    environment: Mapping[str, expr.WordExpression],
) -> Dict[str, expr.WordExpression]:
    ordered_variables = sorted(environment)
    atom_order: List[str] = []
    seen = set()
    for variable in ordered_variables:
        for name in expr.atom_names(environment[variable]):
            if name not in seen:
                seen.add(name)
                atom_order.append(name)
    renaming = {name: f"W{index}" for index, name in enumerate(atom_order)}
    return {
        variable: expr.rename_atoms(environment[variable], renaming)
        for variable in ordered_variables
    }


def _family_key(environment: Mapping[str, expr.WordExpression]) -> str:
    payload = {
        variable: expr.to_dict(value)
        for variable, value in sorted(environment.items())
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _family_kind(
    a_expression: expr.WordExpression,
    b_expression: expr.WordExpression,
) -> str:
    parameters = set(expr.exponent_parameters(a_expression)) | set(
        expr.exponent_parameters(b_expression)
    )
    if not parameters:
        return EXACT_FINITE
    depth = max(
        expr.repeat_nesting_depth(a_expression),
        expr.repeat_nesting_depth(b_expression),
    )
    if len(parameters) == 1 and depth <= 1:
        return EXACT_POWER
    return EXACT_NESTED_POWER


def _finalize_family(
    *,
    case: base.PlacementCase,
    environment: Mapping[str, expr.WordExpression],
    exponent_minimums: Mapping[str, int],
    trace: Sequence[str],
    family_id: int,
) -> ExactFormalFamily:
    normalized_environment = _canonicalize_family_environment(environment)
    a_expression = expr.substitute_word(case.a_word, normalized_environment)
    b_expression = expr.substitute_word(case.b_word, normalized_environment)
    all_exponents = []
    seen = set()
    for value in list(normalized_environment.values()) + [a_expression, b_expression]:
        for name in expr.exponent_parameters(value):
            if name not in seen:
                seen.add(name)
                all_exponents.append(name)
    minimums = tuple(
        (name, int(exponent_minimums.get(name, 0)))
        for name in all_exponents
    )
    return ExactFormalFamily(
        family_id=family_id,
        kind=_family_kind(a_expression, b_expression),
        environment=tuple(sorted(normalized_environment.items())),
        a_expression=a_expression,
        b_expression=b_expression,
        exponent_minimums=minimums,
        trace=tuple(trace),
    )


def _expanded_environment_key(
    family: ExactFormalFamily,
    assignment: Mapping[str, int],
    *,
    length_cap: int,
) -> Optional[str]:
    expanded_environment: Dict[str, expr.WordExpression] = {}
    total_length = 0
    for variable, value in family.environment:
        remaining = max(0, length_cap - total_length)
        value_length = expr.expanded_length(value, assignment, cap=remaining)
        total_length += value_length
        if total_length > length_cap:
            return None
        expanded_environment[variable] = expr.from_word(expr.expand(value, assignment))
    return _family_key(_canonicalize_family_environment(expanded_environment))


def _finite_family_size(family: ExactFormalFamily) -> int:
    if family.parametric:
        raise ValueError("Expected a finite family")
    return sum(
        expr.expanded_length(value, {})
        for _variable, value in family.environment
    )


def _remove_parametric_specializations(
    families: Sequence[ExactFormalFamily],
) -> Tuple[List[ExactFormalFamily], int]:
    """Remove finite outputs already represented by a parametric family.

    Because every word atom is nonempty, any exponent relevant to a finite
    target of total size ``L`` can be searched exactly in the interval from its
    declared minimum through ``L``.  Nested repeats are length-checked before
    expansion, so this guard never materializes enormous words.
    """
    finite = [family for family in families if not family.parametric]
    parametric = [family for family in families if family.parametric]
    if not finite or not parametric:
        return list(families), 0

    finite_sizes = {family.family_id: _finite_family_size(family) for family in finite}
    maximum_size = max(finite_sizes.values())
    specialization_keys = set()
    for family in parametric:
        minimums = dict(family.exponent_minimums)
        names = tuple(sorted(minimums))
        ranges = [range(minimums[name], maximum_size + 1) for name in names]
        for values in product(*ranges):
            assignment = dict(zip(names, values))
            key = _expanded_environment_key(
                family,
                assignment,
                length_cap=maximum_size,
            )
            if key is not None:
                specialization_keys.add(key)

    retained: List[ExactFormalFamily] = []
    suppressed = 0
    for family in families:
        if family.parametric:
            retained.append(family)
            continue
        key = _expanded_environment_key(family, {}, length_cap=finite_sizes[family.family_id])
        if key in specialization_keys:
            suppressed += 1
            continue
        retained.append(family)

    return [replace(family, family_id=index) for index, family in enumerate(retained)], suppressed


def solve_case(
    case: base.PlacementCase,
    *,
    max_nodes: Optional[int],
    max_edges: Optional[int],
    max_families: int,
    representative_exponent_value: Optional[int] = None,
) -> ExactCaseResult:
    # Kept as a source-compatible argument for callers of v1.  Formal solving
    # no longer performs any exponent specialization.
    if representative_exponent_value is not None and representative_exponent_value < 0:
        raise ValueError("representative_exponent_value must be nonnegative")
    if max_families <= 0:
        raise ValueError("max_families must be positive")

    graph = graph_layer.build_graph(
        case.equations,
        initial_a=case.a_word,
        initial_b=case.b_word,
        case_id=case.case_id,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
    summary = graph_layer.graph_summary(graph)
    if not graph.complete:
        return ExactCaseResult(
            case_id=case.case_id,
            status=UNRESOLVED_GRAPH_LIMIT,
            graph_complete=False,
            graph_summary=summary,
            families=(),
            unsupported_families=(),
            unsupported_reasons=(graph.truncation_reason or "graph construction truncated",),
        )
    if graph.root < 0 or not graph.nodes:
        return ExactCaseResult(
            case_id=case.case_id,
            status=EXACT_UNSAT,
            graph_complete=True,
            graph_summary=summary,
            families=(),
            unsupported_families=(),
            unsupported_reasons=(),
        )

    reaches_terminal = _nodes_reaching_terminal(graph)
    if graph.root not in reaches_terminal:
        return ExactCaseResult(
            case_id=case.case_id,
            status=EXACT_UNSAT,
            graph_complete=True,
            graph_summary=summary,
            families=(),
            unsupported_families=(),
            unsupported_reasons=(),
        )

    reachable = _reachable_from_root(graph)
    component_of, components = _component_maps(graph)
    cycle_infos: Dict[int, _CycleInfo] = {}
    commutation_components: set[int] = set()
    unsupported_by_component: Dict[int, str] = {}

    for component_id, component in components.items():
        if not any(node in reachable and node in reaches_terminal for node in component):
            continue
        if not _is_cyclic_component(component, graph):
            continue
        info = _simple_cycle_info(component, graph)
        if info is None:
            if len(component) == 1:
                probe = _commutation_mapping(graph, component[0], _NameFactory())
                if probe is not None:
                    commutation_components.add(component_id)
                    continue
            unsupported_by_component[component_id] = (
                f"SCC {component_id} has branching or multiple internal cycles: "
                f"nodes {list(component)}"
            )
            continue
        for entry in component:
            _mapping, reason = _cycle_repeat_mapping(
                graph, info, entry, "n_probe"
            )
            if reason is not None:
                unsupported_by_component[component_id] = (
                    f"SCC {component_id} is not a fixed-context power cycle at entry {entry}: {reason}"
                )
                break
        else:
            cycle_infos[component_id] = info

    adjacency = _adjacency(graph)
    initial_environment = {
        variable: expr.from_word(word)
        for variable, word in graph.initial_environment
    }
    families: List[ExactFormalFamily] = []
    seen_family_keys = set()
    unsupported_families: List[UnsupportedFormalFamily] = []
    seen_unsupported_keys = set()

    def add_family(
        environment: Mapping[str, expr.WordExpression],
        minimums: Mapping[str, int],
        trace: Sequence[str],
    ) -> None:
        normalized = _canonicalize_family_environment(environment)
        key = _family_key(normalized)
        if key in seen_family_keys:
            return
        if len(families) >= max_families:
            raise _FamilyLimitReached
        seen_family_keys.add(key)
        families.append(
            _finalize_family(
                case=case,
                environment=environment,
                exponent_minimums=minimums,
                trace=trace,
                family_id=len(families),
            )
        )

    def park_unsupported_family(
        *,
        node_id: int,
        component_id: int,
        environment: Mapping[str, expr.WordExpression],
        trace: Sequence[str],
    ) -> None:
        normalized = _canonicalize_family_environment(environment)
        key = (component_id, node_id, _family_key(normalized))
        if key in seen_unsupported_keys:
            return
        seen_unsupported_keys.add(key)
        unsupported_families.append(
            UnsupportedFormalFamily(
                frontier_id=len(unsupported_families),
                component_id=component_id,
                entry_node=node_id,
                component_nodes=components[component_id],
                reason=unsupported_by_component[component_id],
                entry_environment=tuple(sorted(normalized.items())),
                trace=tuple(trace),
            )
        )

    def walk(
        node_id: int,
        environment: Mapping[str, expr.WordExpression],
        minimums: Mapping[str, int],
        trace: Tuple[str, ...],
        names: _NameFactory,
    ) -> None:
        node = graph.nodes[node_id]
        if node.terminal:
            add_family(environment, minimums, trace)
            return
        component_id = component_of[node_id]
        if component_id in unsupported_by_component:
            park_unsupported_family(
                node_id=node_id,
                component_id=component_id,
                environment=environment,
                trace=trace,
            )
            return
        if component_id in commutation_components:
            branch_names = names.clone()
            result = _commutation_mapping(graph, node_id, branch_names)
            if result is None:
                raise AssertionError("Commutation component recognition became inconsistent")
            mapping, added_minimums, label = result
            new_minimums = dict(minimums)
            new_minimums.update(added_minimums)
            add_family(
                _apply_mapping(environment, mapping),
                new_minimums,
                trace + (label,),
            )
            return
        if component_id in cycle_infos:
            info = cycle_infos[component_id]
            members = set(info.component)
            external_edges = [
                edge for source in info.component
                for edge in adjacency.get(source, ())
                if edge.target not in members and edge.target in reaches_terminal
            ]
            for exit_edge in external_edges:
                branch_names = names.clone()
                exponent = branch_names.exponent()
                repeat_mapping, reason = _cycle_repeat_mapping(
                    graph, info, node_id, exponent
                )
                if repeat_mapping is None:
                    raise AssertionError(reason)
                new_environment = _apply_mapping(environment, repeat_mapping)
                new_minimums = dict(minimums)
                if any(
                    isinstance(value, expr.Repeat)
                    or exponent in expr.exponent_parameters(value)
                    for value in repeat_mapping.values()
                ):
                    new_minimums[exponent] = 0
                new_trace = trace + (
                    f"repeat SCC {component_id} with exponent {exponent}",
                )
                for internal_edge in _partial_cycle_edges(
                    info, node_id, exit_edge.source
                ):
                    mapping = _edge_expression_mapping(internal_edge, branch_names)
                    new_environment = _apply_mapping(new_environment, mapping)
                    new_trace += (f"edge {internal_edge.edge_id}:{internal_edge.branch}",)
                mapping = _edge_expression_mapping(exit_edge, branch_names)
                new_environment = _apply_mapping(new_environment, mapping)
                walk(
                    exit_edge.target,
                    new_environment,
                    new_minimums,
                    new_trace + (f"edge {exit_edge.edge_id}:{exit_edge.branch}",),
                    branch_names,
                )
            return

        for edge in adjacency.get(node_id, ()):
            if edge.target not in reaches_terminal:
                continue
            branch_names = names.clone()
            mapping = _edge_expression_mapping(edge, branch_names)
            walk(
                edge.target,
                _apply_mapping(environment, mapping),
                minimums,
                trace + (f"edge {edge.edge_id}:{edge.branch}",),
                branch_names,
            )

    try:
        walk(graph.root, initial_environment, {}, (), _NameFactory())
    except _FamilyLimitReached:
        return ExactCaseResult(
            case_id=case.case_id,
            status=UNRESOLVED_FAMILY_LIMIT,
            graph_complete=True,
            graph_summary=summary,
            families=(),
            unsupported_families=(),
            unsupported_reasons=(f"max_families={max_families} reached",),
        )

    families, suppressed_finite_specializations = _remove_parametric_specializations(
        families
    )

    if unsupported_families and families:
        status = EXACT_MIXED_SUPPORTED_AND_UNSUPPORTED
    elif unsupported_families:
        status = EXACT_GRAPH_UNSUPPORTED
    elif not families:
        status = EXACT_UNSAT
    else:
        kinds = {family.kind for family in families}
        if EXACT_NESTED_POWER in kinds or len(kinds - {EXACT_FINITE}) > 1:
            status = EXACT_NESTED_POWER
        elif EXACT_POWER in kinds:
            status = EXACT_POWER
        else:
            status = EXACT_FINITE
    return ExactCaseResult(
        case_id=case.case_id,
        status=status,
        graph_complete=True,
        graph_summary=summary,
        families=tuple(families),
        unsupported_families=tuple(unsupported_families),
        unsupported_reasons=tuple(
            dict.fromkeys(family.reason for family in unsupported_families)
        ),
        suppressed_finite_specialization_count=suppressed_finite_specializations,
    )


def summarize(results: Sequence[ExactCaseResult]) -> Dict[str, object]:
    counts = Counter(result.status for result in results)
    return {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(results),
        "status_counts": dict(sorted(counts.items())),
        "exact_case_count": sum(result.exact for result in results),
        "downstream_supported_case_count": sum(
            result.downstream_supported for result in results
        ),
        "family_count": sum(len(result.families) for result in results),
        "finite_family_count": sum(
            not family.parametric
            for result in results
            for family in result.families
        ),
        "parametric_family_count": sum(
            family.parametric
            for result in results
            for family in result.families
        ),
        "unsupported_family_frontier_count": sum(
            len(result.unsupported_families) for result in results
        ),
        "suppressed_finite_specialization_count": sum(
            result.suppressed_finite_specialization_count for result in results
        ),
        "unresolved_or_unsupported_case_count": sum(
            bool(result.unsupported_families)
            or result.status in (UNRESOLVED_GRAPH_LIMIT, UNRESOLVED_FAMILY_LIMIT)
            for result in results
        ),
    }
