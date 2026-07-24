#!/usr/bin/env python3
"""
Finite derivation-graph layer for symbolic contour word equations.

This module does not pre-discretize A or B. Placement points create the initial
factorization, and word-equation resolution creates every later split.

A graph node is a canonical system of word equations. An edge is one exhaustive
Levi/Nielsen prefix branch, labelled by a substitution morphism. Variables that
become unconstrained on an edge are represented by edge-local nonempty
parameters K0, K1, ... . Every traversal of an edge creates fresh instances of
those local parameters.

Cycles are not unrolled. A path language through this graph is therefore a
finite parametric representation whenever graph construction terminates. A
cycle may be traversed n >= 0 times; equivalently, its edge morphism is iterated.

Important limitation: this is a transparent Nielsen derivation graph, not the
full recompression algorithm. It is exact for the graph it constructs, but
termination/finite-state completeness for arbitrary word-equation systems is
not claimed. Full generality requires a recompression/EDT0L implementation.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import symbolic_enumerator as base


Word = base.Word
Equation = base.Equation
Literal = base.Literal


@dataclass(frozen=True)
class GraphNode:
    node_id: int
    equations: Tuple[Equation, ...]

    @property
    def terminal(self) -> bool:
        return not self.equations

    def to_dict(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id,
            "terminal": self.terminal,
            "equations": [equation.to_text() for equation in self.equations],
        }


@dataclass(frozen=True)
class GraphEdge:
    edge_id: int
    source: int
    target: int
    branch: str
    morphism: Tuple[Tuple[str, Word], ...]
    local_parameters: Tuple[str, ...]

    def morphism_map(self) -> Dict[str, Word]:
        return dict(self.morphism)

    def to_dict(self) -> Dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "branch": self.branch,
            "morphism": {
                variable: base.word_to_text(word)
                for variable, word in self.morphism
            },
            "local_parameters": list(self.local_parameters),
            "iteration": (
                "n >= 0 when this edge belongs to a directed cycle"
            ),
        }


@dataclass(frozen=True)
class DerivationGraph:
    case_id: Optional[int]
    initial_a: Word
    initial_b: Word
    nodes: Tuple[GraphNode, ...]
    edges: Tuple[GraphEdge, ...]
    root: int
    complete: bool
    truncation_reason: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        cyclic_edges = set(edge_ids_in_cycles(self.nodes, self.edges))
        return {
            "case_id": self.case_id,
            "root": self.root,
            "complete": self.complete,
            "truncation_reason": self.truncation_reason,
            "initial_A": base.word_to_text(self.initial_a),
            "initial_B": base.word_to_text(self.initial_b),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [
                {
                    **edge.to_dict(),
                    "cyclic": edge.edge_id in cyclic_edges,
                }
                for edge in self.edges
            ],
            "semantics": {
                "path": (
                    "Compose edge morphisms from root to a terminal node."
                ),
                "cycle": (
                    "A directed cycle may be traversed an arbitrary number "
                    "n >= 0 of times; edge-local K variables are fresh on "
                    "each traversal."
                ),
            },
        }


def variables_in_equations(equations: Sequence[Equation]) -> Tuple[str, ...]:
    return tuple(
        sorted(
            {
                literal.variable
                for equation in equations
                for word in (equation.left, equation.right)
                for literal in word
            }
        )
    )


def canonicalize_core(
    equations: Sequence[Equation],
) -> Optional[Tuple[Tuple[Equation, ...], Dict[str, str]]]:
    """
    Canonicalize only the active equation system modulo:

    - equation permutation;
    - swapping the two sides of any equation;
    - alpha-renaming variables by first occurrence.

    Return the canonical equations and the old-to-canonical renaming used by
    the selected representative.
    """
    simplified = base.simplify_system(equations)
    if simplified is None:
        return None

    equation_count = len(simplified)
    best_serialization: Optional[Tuple[object, ...]] = None
    best_equations: Optional[Tuple[Equation, ...]] = None
    best_renaming: Optional[Dict[str, str]] = None

    orders: Iterable[Tuple[int, ...]]
    orders = permutations(range(equation_count)) if equation_count else [()]

    for order in orders:
        for flips in product((False, True), repeat=equation_count):
            renaming: Dict[str, str] = {}
            next_index = 0
            canonical_equations: List[Equation] = []
            serialization: List[object] = []

            def rename_word(word: Word) -> Word:
                nonlocal next_index
                output: List[Literal] = []
                for literal in word:
                    if literal.variable not in renaming:
                        renaming[literal.variable] = f"V{next_index}"
                        next_index += 1
                    output.append(
                        Literal(
                            renaming[literal.variable],
                            literal.inverse,
                        )
                    )
                return tuple(output)

            for equation_index, flip in zip(order, flips):
                equation = simplified[equation_index]
                left, right = (
                    (equation.right, equation.left)
                    if flip
                    else (equation.left, equation.right)
                )
                canonical_left = rename_word(left)
                canonical_right = rename_word(right)
                canonical_equations.append(
                    Equation(canonical_left, canonical_right)
                )
                serialization.append(
                    (
                        tuple(
                            (literal.variable, literal.inverse)
                            for literal in canonical_left
                        ),
                        tuple(
                            (literal.variable, literal.inverse)
                            for literal in canonical_right
                        ),
                    )
                )

            serialized = tuple(serialization)
            if (
                best_serialization is None
                or serialized < best_serialization
            ):
                best_serialization = serialized
                best_equations = tuple(canonical_equations)
                best_renaming = dict(renaming)

    assert best_equations is not None
    assert best_renaming is not None
    return best_equations, best_renaming


def fresh_residual(equations: Sequence[Equation]) -> str:
    used = set(variables_in_equations(equations))
    index = 0
    while f"R{index}" in used:
        index += 1
    return f"R{index}"


def prefix_branches(
    equations: Tuple[Equation, ...],
) -> Iterator[Tuple[str, Dict[str, Word]]]:
    """
    Exhaustive first-prefix branches.

    For distinct leading variables X^e and Y^f, exactly one holds in any
    concrete solution:

    - equal lengths;
    - X^e is a strict prefix of Y^f;
    - Y^f is a strict prefix of X^e.

    These are precisely the branches that recover relative orders between an
    already placed marker and a split point introduced later by resolution.
    """
    equation = equations[0]
    left = equation.left[0]
    right = equation.right[0]
    residual_name = fresh_residual(equations)
    residual = Literal(residual_name)

    if left.variable == right.variable:
        if left.inverse == right.inverse:
            raise RuntimeError("Identical prefixes should have been canceled")

        # Formal fixed-point-free path involution model.
        yield "involutive_fixed_word", {
            left.variable: (residual, residual.flipped())
        }
        return

    equal_orientation = left.inverse ^ right.inverse
    yield "same_cut", {
        left.variable: (
            Literal(right.variable, equal_orientation),
        )
    }

    # |left| < |right|: right-oriented word = left-oriented word + residual.
    oriented_right = (left, residual)
    right_positive = (
        base.inverse_word(oriented_right)
        if right.inverse
        else oriented_right
    )
    yield "left_cut_before_right_cut", {
        right.variable: right_positive
    }

    # |right| < |left|.
    oriented_left = (right, residual)
    left_positive = (
        base.inverse_word(oriented_left)
        if left.inverse
        else oriented_left
    )
    yield "right_cut_before_left_cut", {
        left.variable: left_positive
    }


def apply_branch(
    source_equations: Tuple[Equation, ...],
    substitution: Mapping[str, Word],
) -> Optional[Tuple[Tuple[Equation, ...], Tuple[Tuple[str, Word], ...], Tuple[str, ...]]]:
    """
    Apply a branch and build its edge morphism.

    Active variables are renamed to child-state variables V0, V1, ... . Any
    variable no longer occurring in the child equations is unconstrained and
    becomes an edge-local nonempty parameter K0, K1, ... .
    """
    substituted = base.substitute_equations(source_equations, substitution)
    canonical = canonicalize_core(substituted)
    if canonical is None:
        return None

    child_equations, active_renaming = canonical
    parent_variables = variables_in_equations(source_equations)

    expanded_images: Dict[str, Word] = {}
    free_after: List[str] = []

    for parent in parent_variables:
        image = substitution.get(parent, (Literal(parent),))
        expanded_images[parent] = image
        for literal in image:
            if (
                literal.variable not in active_renaming
                and literal.variable not in free_after
            ):
                free_after.append(literal.variable)

    free_renaming = {
        variable: f"K{index}"
        for index, variable in enumerate(free_after)
    }

    morphism: List[Tuple[str, Word]] = []
    for parent in parent_variables:
        output: List[Literal] = []
        for literal in expanded_images[parent]:
            if literal.variable in active_renaming:
                target_name = active_renaming[literal.variable]
            else:
                target_name = free_renaming[literal.variable]
            output.append(Literal(target_name, literal.inverse))
        morphism.append((parent, tuple(output)))

    return child_equations, tuple(morphism), tuple(free_renaming.values())


def build_graph(
    equations: Sequence[Equation],
    initial_a: Word,
    initial_b: Word,
    case_id: Optional[int] = None,
    max_nodes: Optional[int] = None,
    max_edges: Optional[int] = None,
) -> DerivationGraph:
    canonical = canonicalize_core(equations)
    if canonical is None:
        return DerivationGraph(
            case_id=case_id,
            initial_a=initial_a,
            initial_b=initial_b,
            nodes=(),
            edges=(),
            root=-1,
            complete=True,
            truncation_reason="initial contradiction",
        )

    root_equations, root_renaming = canonical

    # Rewrite initial A/B expressions into root variable names. Variables absent
    # from equations are already free; retain them as root-local K parameters.
    root_free: Dict[str, str] = {}

    def rewrite_initial(word: Word) -> Word:
        output: List[Literal] = []
        for literal in word:
            if literal.variable in root_renaming:
                name = root_renaming[literal.variable]
            else:
                if literal.variable not in root_free:
                    root_free[literal.variable] = f"Kroot{len(root_free)}"
                name = root_free[literal.variable]
            output.append(Literal(name, literal.inverse))
        return tuple(output)

    canonical_a = rewrite_initial(initial_a)
    canonical_b = rewrite_initial(initial_b)

    node_ids: Dict[Tuple[Equation, ...], int] = {root_equations: 0}
    node_equations: List[Tuple[Equation, ...]] = [root_equations]
    queue = deque([root_equations])
    edges: List[GraphEdge] = []
    complete = True
    truncation_reason: Optional[str] = None

    while queue:
        source_equations = queue.popleft()
        source_id = node_ids[source_equations]

        if not source_equations:
            continue

        for branch_name, substitution in prefix_branches(source_equations):
            result = apply_branch(source_equations, substitution)
            if result is None:
                continue
            child_equations, morphism, local_parameters = result

            if child_equations not in node_ids:
                if max_nodes is not None and len(node_ids) >= max_nodes:
                    complete = False
                    truncation_reason = f"max_nodes={max_nodes} reached"
                    queue.clear()
                    break
                node_ids[child_equations] = len(node_ids)
                node_equations.append(child_equations)
                queue.append(child_equations)

            if max_edges is not None and len(edges) >= max_edges:
                complete = False
                truncation_reason = f"max_edges={max_edges} reached"
                queue.clear()
                break

            edges.append(
                GraphEdge(
                    edge_id=len(edges),
                    source=source_id,
                    target=node_ids[child_equations],
                    branch=branch_name,
                    morphism=morphism,
                    local_parameters=local_parameters,
                )
            )

    nodes = tuple(
        GraphNode(node_id=index, equations=equations_value)
        for index, equations_value in enumerate(node_equations)
    )

    return DerivationGraph(
        case_id=case_id,
        initial_a=canonical_a,
        initial_b=canonical_b,
        nodes=nodes,
        edges=tuple(edges),
        root=0,
        complete=complete,
        truncation_reason=truncation_reason,
    )


def strongly_connected_components(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
) -> List[List[int]]:
    adjacency: Dict[int, List[int]] = {node.node_id: [] for node in nodes}
    for edge in edges:
        adjacency.setdefault(edge.source, []).append(edge.target)

    index = 0
    stack: List[int] = []
    on_stack: set[int] = set()
    indices: Dict[int, int] = {}
    lowlink: Dict[int, int] = {}
    components: List[List[int]] = []

    def visit(vertex: int) -> None:
        nonlocal index
        indices[vertex] = index
        lowlink[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack.add(vertex)

        for neighbor in adjacency.get(vertex, []):
            if neighbor not in indices:
                visit(neighbor)
                lowlink[vertex] = min(lowlink[vertex], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[vertex] = min(lowlink[vertex], indices[neighbor])

        if lowlink[vertex] == indices[vertex]:
            component: List[int] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == vertex:
                    break
            components.append(component)

    for node in nodes:
        if node.node_id not in indices:
            visit(node.node_id)

    return components


def edge_ids_in_cycles(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
) -> List[int]:
    components = strongly_connected_components(nodes, edges)
    component_of: Dict[int, int] = {}
    component_size: Dict[int, int] = {}
    for component_index, component in enumerate(components):
        component_size[component_index] = len(component)
        for node_id in component:
            component_of[node_id] = component_index

    cyclic: List[int] = []
    for edge in edges:
        component_index = component_of[edge.source]
        if component_index != component_of[edge.target]:
            continue
        if component_size[component_index] > 1 or edge.source == edge.target:
            cyclic.append(edge.edge_id)
    return cyclic


def graph_summary(graph: DerivationGraph) -> Dict[str, object]:
    cyclic_edges = edge_ids_in_cycles(graph.nodes, graph.edges)
    terminal_nodes = [node.node_id for node in graph.nodes if node.terminal]
    return {
        "case_id": graph.case_id,
        "complete": graph.complete,
        "truncation_reason": graph.truncation_reason,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "terminal_nodes": terminal_nodes,
        "cyclic_edge_count": len(cyclic_edges),
        "cyclic_edges": cyclic_edges,
    }


def write_dot(graph: DerivationGraph, path: Path) -> None:
    cyclic = set(edge_ids_in_cycles(graph.nodes, graph.edges))
    lines = ["digraph derivation_graph {"]
    for node in graph.nodes:
        label = "terminal" if node.terminal else "\\n".join(
            equation.to_text().replace('"', '\\"')
            for equation in node.equations
        )
        shape = "doublecircle" if node.terminal else "box"
        lines.append(f'  n{node.node_id} [shape={shape}, label="{label}"];')

    for edge in graph.edges:
        morphism = ", ".join(
            f"{variable}->{base.word_to_text(word)}"
            for variable, word in edge.morphism
        )
        star = " [repeat n>=0]" if edge.edge_id in cyclic else ""
        label = f"{edge.branch}: {morphism}{star}".replace('"', '\\"')
        lines.append(
            f'  n{edge.source} -> n{edge.target} '
            f'[label="{label}"];'
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def commuting_demo_graph() -> DerivationGraph:
    x = Literal("X")
    y = Literal("Y")
    equation = Equation((x, y), (y, x))
    return build_graph(
        equations=(equation,),
        initial_a=(x,),
        initial_b=(y,),
        case_id=None,
    )


def command_case(args: argparse.Namespace) -> int:
    case = base.find_case(args.case_id)
    graph = build_graph(
        equations=case.equations,
        initial_a=case.a_word,
        initial_b=case.b_word,
        case_id=case.case_id,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
    )
    args.output.write_text(
        json.dumps(graph.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if args.dot:
        write_dot(graph, args.dot)
    print(json.dumps(graph_summary(graph), indent=2))
    return 0


def command_commute(args: argparse.Namespace) -> int:
    graph = commuting_demo_graph()
    args.output.write_text(
        json.dumps(graph.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if args.dot:
        write_dot(graph, args.dot)
    print(json.dumps(graph_summary(graph), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cycle-preserving parametric derivation graph for "
            "symbolic contour word equations."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    case_parser = subparsers.add_parser("case")
    case_parser.add_argument("case_id", type=int)
    case_parser.add_argument("--output", type=Path, required=True)
    case_parser.add_argument("--dot", type=Path)
    case_parser.add_argument("--max-nodes", type=int)
    case_parser.add_argument("--max-edges", type=int)
    case_parser.set_defaults(function=command_case)

    commute_parser = subparsers.add_parser("commute-demo")
    commute_parser.add_argument("--output", type=Path, required=True)
    commute_parser.add_argument("--dot", type=Path)
    commute_parser.set_defaults(function=command_commute)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
