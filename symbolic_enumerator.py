#!/usr/bin/env python3
"""
Exact symbolic placement enumerator and lazy word-equation solution enumerator.

The initial contour is never discretized. It starts as exactly:

    P0 -- A -- P1 -- B -- P0

Four labeled image points are placed on that contour:

    A_start : image of P0 for the copy covering A
    A_end   : image of P1 for the copy covering A
    B_start : image of P1 for the copy covering B
    B_end   : image of P0 for the copy covering B

A or B is split only when one of these image points lies strictly inside it.
Coincident image points create only one split. Further splits are introduced only
by the word-equation solver through strict prefix comparisons or involutive
palindrome constraints.

The placement layer is finite and exhaustive. The solution layer is a fair lazy
enumerator based on Levi/Nielsen transformations. It may produce infinitely many
solution schemes, and it is intentionally not bounded by a preselected contour
resolution.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import projection_topology as topology
import settings


FORWARD = settings.FORWARD
REVERSE = settings.REVERSE
DIRECT = settings.DIRECT
REFLECTED = settings.REFLECTED
MARKERS = settings.MARKERS
LOCI = settings.LOCI


@dataclass(frozen=True, order=True)
class Literal:
    variable: str
    inverse: bool = False

    def flipped(self) -> "Literal":
        return Literal(self.variable, not self.inverse)

    def to_text(self) -> str:
        return f"{self.variable}^-1" if self.inverse else self.variable


Word = Tuple[Literal, ...]


@dataclass(frozen=True)
class Equation:
    left: Word
    right: Word

    def to_text(self) -> str:
        return f"{word_to_text(self.left)} = {word_to_text(self.right)}"


@dataclass(frozen=True)
class PlacementCase:
    case_id: int
    marker_loci: Tuple[Tuple[str, str], ...]
    a_interior_blocks: Tuple[Tuple[str, ...], ...]
    b_interior_blocks: Tuple[Tuple[str, ...], ...]
    marker_boundaries: Tuple[Tuple[str, int], ...]
    a_word: Word
    b_word: Word
    cycle_word: Word
    a_direction: int
    b_direction: int
    a_mirror_sign: int
    b_mirror_sign: int
    a_target: Word
    b_target: Word
    equations: Tuple[Equation, Equation]

    def __post_init__(self) -> None:
        expected_a = topology.required_mirror_sign(self.a_direction)
        expected_b = topology.required_mirror_sign(self.b_direction)
        if self.a_mirror_sign != expected_a or self.b_mirror_sign != expected_b:
            raise ValueError(
                "Placement parity must be fixed by opposite-side contact at construction time"
            )

    def marker_boundary_map(self) -> Dict[str, int]:
        return dict(self.marker_boundaries)

    def marker_locus_map(self) -> Dict[str, str]:
        return dict(self.marker_loci)

    @property
    def parity_label(self) -> str:
        return topology.parity_label(self.a_mirror_sign, self.b_mirror_sign)

    @property
    def requires_reflection(self) -> bool:
        return self.a_mirror_sign == REFLECTED or self.b_mirror_sign == REFLECTED

    def to_dict(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "marker_loci": dict(self.marker_loci),
            "a_interior_blocks": [list(block) for block in self.a_interior_blocks],
            "b_interior_blocks": [list(block) for block in self.b_interior_blocks],
            "marker_boundaries": dict(self.marker_boundaries),
            "a_direction": direction_name(self.a_direction),
            "b_direction": direction_name(self.b_direction),
            "a_isometry": topology.isometry_name(self.a_mirror_sign),
            "b_isometry": topology.isometry_name(self.b_mirror_sign),
            "a_mirror_sign": self.a_mirror_sign,
            "b_mirror_sign": self.b_mirror_sign,
            "contact_parity": self.parity_label,
            "a_word": word_to_text(self.a_word),
            "b_word": word_to_text(self.b_word),
            "cycle_word": word_to_text(self.cycle_word),
            "a_target": word_to_text(self.a_target),
            "b_target": word_to_text(self.b_target),
            "equations": [equation.to_text() for equation in self.equations],
        }


@dataclass(frozen=True)
class SolverState:
    equations: Tuple[Equation, ...]
    environment: Tuple[Tuple[str, Word], ...]
    depth: int

    def environment_map(self) -> Dict[str, Word]:
        return dict(self.environment)


@dataclass(frozen=True)
class SolutionScheme:
    case_id: int
    depth: int
    a_expression: Word
    b_expression: Word
    parameters: Tuple[str, ...]
    derivation: Tuple[str, ...]

    def key(self) -> Tuple[Word, Word]:
        return self.a_expression, self.b_expression

    def to_dict(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "depth": self.depth,
            "A": word_to_text(self.a_expression),
            "B": word_to_text(self.b_expression),
            "contour": f"P0 {word_to_text(self.a_expression)} P1 {word_to_text(self.b_expression)}",
            "parameters": list(self.parameters),
            "derivation": list(self.derivation),
        }


def direction_name(direction: int) -> str:
    return "forward" if direction == FORWARD else "reverse"


def word_to_text(word: Sequence[Literal]) -> str:
    return "1" if not word else " ".join(literal.to_text() for literal in word)


def inverse_word(word: Sequence[Literal]) -> Word:
    return tuple(literal.flipped() for literal in reversed(word))


def ordered_partitions(items: Sequence[str]) -> Iterator[Tuple[Tuple[str, ...], ...]]:
    """Enumerate all weak orders of labeled items as ordered coincidence blocks."""
    items = tuple(items)
    if not items:
        yield ()
        return

    size = len(items)
    seen: set[Tuple[int, ...]] = set()

    for raw_ranks in product(range(size), repeat=size):
        values = sorted(set(raw_ranks))
        normalization = {value: index for index, value in enumerate(values)}
        ranks = tuple(normalization[value] for value in raw_ranks)
        if ranks in seen:
            continue
        seen.add(ranks)

        blocks: List[List[str]] = [[] for _ in range(max(ranks) + 1)]
        for item, rank in zip(items, ranks):
            blocks[rank].append(item)
        yield tuple(tuple(block) for block in blocks)


def build_refined_contour(
    marker_loci: Mapping[str, str],
    a_blocks: Tuple[Tuple[str, ...], ...],
    b_blocks: Tuple[Tuple[str, ...], ...],
) -> Tuple[Word, Word, Word, Dict[str, int]]:
    """Split A and B only at distinct interior marker positions."""
    a_word = tuple(Literal(f"A{index}") for index in range(len(a_blocks) + 1))
    b_word = tuple(Literal(f"B{index}") for index in range(len(b_blocks) + 1))
    cycle = a_word + b_word
    p1_boundary = len(a_word)

    boundaries: Dict[str, int] = {}

    for marker, locus in marker_loci.items():
        if locus == "P0":
            boundaries[marker] = 0
        elif locus == "P1":
            boundaries[marker] = p1_boundary

    for block_index, block in enumerate(a_blocks):
        for marker in block:
            boundaries[marker] = block_index + 1

    for block_index, block in enumerate(b_blocks):
        for marker in block:
            boundaries[marker] = p1_boundary + block_index + 1

    if set(boundaries) != set(MARKERS):
        raise RuntimeError("Internal marker placement error")

    return a_word, b_word, cycle, boundaries


def cyclic_factor(cycle: Word, start: int, end: int, direction: int) -> Word:
    """Read the oriented contour factor from one boundary to another."""
    if start == end:
        return ()

    size = len(cycle)
    current = start
    output: List[Literal] = []

    if direction == FORWARD:
        while current != end:
            output.append(cycle[current])
            current = (current + 1) % size
    else:
        while current != end:
            segment_index = (current - 1) % size
            output.append(cycle[segment_index].flipped())
            current = segment_index

    return tuple(output)


def enumerate_placement_cases(
    allow_reflections: bool = settings.DEFAULT_ALLOW_REFLECTIONS,
) -> Iterator[PlacementCase]:
    """Enumerate every symbolic order type with contact parity already fixed.

    If ``allow_reflections`` is false, placements requiring a reflected copy are
    pruned immediately, before any word equation is solved.
    """
    case_id = 0

    for assigned_loci in product(LOCI, repeat=len(MARKERS)):
        marker_loci = dict(zip(MARKERS, assigned_loci))
        a_markers = [marker for marker in MARKERS if marker_loci[marker] == "A"]
        b_markers = [marker for marker in MARKERS if marker_loci[marker] == "B"]

        for a_blocks in ordered_partitions(a_markers):
            for b_blocks in ordered_partitions(b_markers):
                a_word, b_word, cycle, boundaries = build_refined_contour(
                    marker_loci,
                    a_blocks,
                    b_blocks,
                )

                for a_direction, b_direction in product(
                    (FORWARD, REVERSE), repeat=2
                ):
                    a_topology = topology.ProjectionTopology.from_direction(
                        "A", a_direction
                    )
                    b_topology = topology.ProjectionTopology.from_direction(
                        "B", b_direction
                    )

                    a_target = cyclic_factor(
                        cycle,
                        boundaries["A_start"],
                        boundaries["A_end"],
                        a_direction,
                    )
                    b_target = cyclic_factor(
                        cycle,
                        boundaries["B_start"],
                        boundaries["B_end"],
                        b_direction,
                    )

                    # A and B are nonempty. A zero target factor is impossible.
                    if not a_target or not b_target:
                        continue

                    current_case_id = case_id
                    case_id += 1

                    if not allow_reflections and (
                        a_topology.requires_reflection or b_topology.requires_reflection
                    ):
                        continue

                    equations = (
                        Equation(a_word, a_target),
                        Equation(b_word, b_target),
                    )

                    yield PlacementCase(
                        case_id=current_case_id,
                        marker_loci=tuple((marker, marker_loci[marker]) for marker in MARKERS),
                        a_interior_blocks=a_blocks,
                        b_interior_blocks=b_blocks,
                        marker_boundaries=tuple((marker, boundaries[marker]) for marker in MARKERS),
                        a_word=a_word,
                        b_word=b_word,
                        cycle_word=cycle,
                        a_direction=a_direction,
                        b_direction=b_direction,
                        a_mirror_sign=a_topology.mirror_sign,
                        b_mirror_sign=b_topology.mirror_sign,
                        a_target=a_target,
                        b_target=b_target,
                        equations=equations,
                    )


def substitute_word(word: Word, substitution: Mapping[str, Word]) -> Word:
    output: List[Literal] = []
    for literal in word:
        replacement = substitution.get(literal.variable, (Literal(literal.variable),))
        if literal.inverse:
            replacement = inverse_word(replacement)
        output.extend(replacement)
    return tuple(output)


def substitute_equations(
    equations: Sequence[Equation], substitution: Mapping[str, Word]
) -> Tuple[Equation, ...]:
    return tuple(
        Equation(
            substitute_word(equation.left, substitution),
            substitute_word(equation.right, substitution),
        )
        for equation in equations
    )


def simplify_equation(equation: Equation) -> Optional[Equation | bool]:
    """
    Cancel identical word prefixes and suffixes.

    Return:
      None  for a tautology;
      False for a contradiction;
      Equation otherwise.
    """
    left = list(equation.left)
    right = list(equation.right)

    while left and right and left[0] == right[0]:
        left.pop(0)
        right.pop(0)

    while left and right and left[-1] == right[-1]:
        left.pop()
        right.pop()

    if not left and not right:
        return None
    if not left or not right:
        return False
    return Equation(tuple(left), tuple(right))


def simplify_system(equations: Sequence[Equation]) -> Optional[Tuple[Equation, ...]]:
    simplified: List[Equation] = []
    seen: set[Tuple[Word, Word]] = set()

    for equation in equations:
        result = simplify_equation(equation)
        if result is False:
            return None
        if result is None:
            continue
        assert isinstance(result, Equation)

        direct = (result.left, result.right)
        reverse = (result.right, result.left)
        key = min(direct, reverse)
        if key not in seen:
            seen.add(key)
            simplified.append(result)

    return tuple(simplified)


def serialize_word(word: Word, renaming: Mapping[str, str]) -> Tuple[Tuple[int, bool], ...]:
    return tuple((int(renaming[literal.variable][1:]), literal.inverse) for literal in word)


def canonicalize_state(
    equations: Sequence[Equation],
    environment: Mapping[str, Word],
) -> Optional[Tuple[Tuple[Equation, ...], Tuple[Tuple[str, Word], ...]]]:
    """Canonicalize current variable names modulo alpha-renaming."""
    simplified = simplify_system(equations)
    if simplified is None:
        return None

    environment_items = tuple(sorted(environment.items()))
    equation_count = len(simplified)

    best_serialization: Optional[Tuple[object, ...]] = None
    best_equations: Optional[Tuple[Equation, ...]] = None
    best_environment: Optional[Tuple[Tuple[str, Word], ...]] = None

    equation_orders = permutations(range(equation_count)) if equation_count else [()]

    for order in equation_orders:
        for flips in product((False, True), repeat=equation_count):
            renaming: Dict[str, str] = {}
            next_index = 0
            canonical_equations: List[Equation] = []
            equation_serialization: List[object] = []

            def rename_word(word: Word) -> Word:
                nonlocal next_index
                output: List[Literal] = []
                for literal in word:
                    if literal.variable not in renaming:
                        renaming[literal.variable] = f"V{next_index}"
                        next_index += 1
                    output.append(Literal(renaming[literal.variable], literal.inverse))
                return tuple(output)

            for equation_index, flip in zip(order, flips):
                equation = simplified[equation_index]
                left, right = (
                    (equation.right, equation.left) if flip else (equation.left, equation.right)
                )
                canonical_left = rename_word(left)
                canonical_right = rename_word(right)
                canonical_equations.append(Equation(canonical_left, canonical_right))
                equation_serialization.append(
                    (
                        tuple((int(literal.variable[1:]), literal.inverse) for literal in canonical_left),
                        tuple((int(literal.variable[1:]), literal.inverse) for literal in canonical_right),
                    )
                )

            canonical_environment: List[Tuple[str, Word]] = []
            environment_serialization: List[object] = []
            for initial_variable, word in environment_items:
                canonical_word = rename_word(word)
                canonical_environment.append((initial_variable, canonical_word))
                environment_serialization.append(
                    (
                        initial_variable,
                        tuple((int(literal.variable[1:]), literal.inverse) for literal in canonical_word),
                    )
                )

            serialization = (
                tuple(equation_serialization),
                tuple(environment_serialization),
            )

            if best_serialization is None or serialization < best_serialization:
                best_serialization = serialization
                best_equations = tuple(canonical_equations)
                best_environment = tuple(canonical_environment)

    assert best_equations is not None
    assert best_environment is not None
    return best_equations, best_environment


def current_variables(
    equations: Sequence[Equation], environment: Mapping[str, Word]
) -> set[str]:
    variables: set[str] = set()
    for equation in equations:
        for word in (equation.left, equation.right):
            variables.update(literal.variable for literal in word)
    for word in environment.values():
        variables.update(literal.variable for literal in word)
    return variables


def fresh_variable(
    equations: Sequence[Equation], environment: Mapping[str, Word]
) -> str:
    used = current_variables(equations, environment)
    index = 0
    while f"R{index}" in used or f"V{index}" in used:
        index += 1
    return f"R{index}"


def branch_substitutions(
    equations: Tuple[Equation, ...],
    environment: Mapping[str, Word],
) -> Iterator[Tuple[str, Dict[str, Word]]]:
    """Generate the complete Levi/Nielsen prefix branches for one state."""
    equation = equations[0]
    left_literal = equation.left[0]
    right_literal = equation.right[0]
    residual_name = fresh_variable(equations, environment)
    residual = Literal(residual_name)

    if left_literal.variable == right_literal.variable:
        # Identical orientations would already have been canceled.
        if left_literal.inverse == right_literal.inverse:
            raise RuntimeError("Uncanceled identical literals")

        # W = W^-1. With a fixed-point-free orientation involution,
        # every nonempty solution is U U^-1.
        substitution = {
            left_literal.variable: (residual, residual.flipped())
        }
        yield "involutive_palindrome", substitution
        return

    # Equal prefix lengths: X^e = Y^f.
    equal_orientation = left_literal.inverse ^ right_literal.inverse
    yield "equal_length", {
        left_literal.variable: (
            Literal(right_literal.variable, equal_orientation),
        )
    }

    # The left prefix is strictly shorter: Y^f = X^e R.
    oriented_right = (left_literal, residual)
    right_positive = (
        inverse_word(oriented_right)
        if right_literal.inverse
        else oriented_right
    )
    yield "left_strictly_shorter", {
        right_literal.variable: right_positive
    }

    # The right prefix is strictly shorter: X^e = Y^f R.
    oriented_left = (right_literal, residual)
    left_positive = (
        inverse_word(oriented_left)
        if left_literal.inverse
        else oriented_left
    )
    yield "right_strictly_shorter", {
        left_literal.variable: left_positive
    }


def initial_solver_state(case: PlacementCase) -> Optional[SolverState]:
    initial_variables = sorted(
        {literal.variable for literal in case.cycle_word}
    )
    environment = {
        variable: (Literal(variable),)
        for variable in initial_variables
    }
    canonical = canonicalize_state(case.equations, environment)
    if canonical is None:
        return None
    equations, canonical_environment = canonical
    return SolverState(equations, canonical_environment, depth=0)


def advance_state(
    state: SolverState,
    substitution: Mapping[str, Word],
) -> Optional[SolverState]:
    environment = state.environment_map()
    new_equations = substitute_equations(state.equations, substitution)
    new_environment = {
        initial_variable: substitute_word(word, substitution)
        for initial_variable, word in environment.items()
    }
    canonical = canonicalize_state(new_equations, new_environment)
    if canonical is None:
        return None
    equations, canonical_environment = canonical
    return SolverState(equations, canonical_environment, state.depth + 1)


def terminal_scheme(
    case: PlacementCase,
    state: SolverState,
    derivation: Tuple[str, ...],
) -> SolutionScheme:
    environment = state.environment_map()

    a_expression: List[Literal] = []
    for literal in case.a_word:
        a_expression.extend(environment[literal.variable])

    b_expression: List[Literal] = []
    for literal in case.b_word:
        b_expression.extend(environment[literal.variable])

    # Canonicalize free parameter names by first occurrence in A then B.
    renaming: Dict[str, str] = {}
    next_index = 0

    def rename_expression(expression: Sequence[Literal]) -> Word:
        nonlocal next_index
        output: List[Literal] = []
        for literal in expression:
            if literal.variable not in renaming:
                renaming[literal.variable] = f"X{next_index}"
                next_index += 1
            output.append(Literal(renaming[literal.variable], literal.inverse))
        return tuple(output)

    canonical_a = rename_expression(a_expression)
    canonical_b = rename_expression(b_expression)

    return SolutionScheme(
        case_id=case.case_id,
        depth=state.depth,
        a_expression=canonical_a,
        b_expression=canonical_b,
        parameters=tuple(f"X{index}" for index in range(next_index)),
        derivation=derivation,
    )


def terminal_state_satisfies_case(
    case: PlacementCase,
    state: SolverState,
) -> bool:
    """Return whether a terminal environment satisfies the original case equations."""
    if state.equations:
        return False
    substituted = substitute_equations(case.equations, state.environment_map())
    simplified = simplify_system(substituted)
    return simplified == ()


def enumerate_terminal_states(
    case: PlacementCase,
    max_depth: Optional[int] = None,
    max_states: Optional[int] = None,
) -> Iterator[Tuple[SolverState, Tuple[str, ...]]]:
    """Fairly enumerate terminal solver states while preserving substitutions."""
    initial = initial_solver_state(case)
    if initial is None:
        return

    queue = deque([(initial, tuple())])
    seen_at_depth: set[
        Tuple[int, Tuple[Equation, ...], Tuple[Tuple[str, Word], ...]]
    ] = set()
    visited_states = 0

    while queue:
        state, derivation = queue.popleft()
        visited_states += 1
        if max_states is not None and visited_states > max_states:
            return

        state_key = (state.depth, state.equations, state.environment)
        if state_key in seen_at_depth:
            continue
        seen_at_depth.add(state_key)

        if not state.equations:
            yield state, derivation
            continue

        if max_depth is not None and state.depth >= max_depth:
            continue

        environment = state.environment_map()
        for branch_name, substitution in branch_substitutions(
            state.equations, environment
        ):
            child = advance_state(state, substitution)
            if child is not None:
                queue.append((child, derivation + (branch_name,)))


def enumerate_solution_schemes(
    case: PlacementCase,
    max_depth: Optional[int] = None,
    max_states: Optional[int] = None,
) -> Iterator[SolutionScheme]:
    """
    Fairly enumerate all terminal symbolic schemes for one placement case.

    Without max_depth/max_states this is an unbounded lazy stream. Every
    concrete nonempty solution determines a finite Levi/Nielsen derivation,
    so it eventually appears in the stream, although duplicate schemes may
    arise through different derivations.
    """
    initial = initial_solver_state(case)
    if initial is None:
        return
    queue = deque([(initial, tuple())])
    seen_at_depth: set[Tuple[int, Tuple[Equation, ...], Tuple[Tuple[str, Word], ...]]] = set()
    emitted: set[Tuple[Word, Word]] = set()
    visited_states = 0

    while queue:
        state, derivation = queue.popleft()
        visited_states += 1
        if max_states is not None and visited_states > max_states:
            return

        state_key = (state.depth, state.equations, state.environment)
        if state_key in seen_at_depth:
            continue
        seen_at_depth.add(state_key)

        if not state.equations:
            scheme = terminal_scheme(case, state, derivation)
            if scheme.key() not in emitted:
                emitted.add(scheme.key())
                yield scheme
            continue

        if max_depth is not None and state.depth >= max_depth:
            continue

        environment = state.environment_map()
        for branch_name, substitution in branch_substitutions(
            state.equations, environment
        ):
            child = advance_state(state, substitution)
            if child is not None:
                queue.append((child, derivation + (branch_name,)))


def find_case(case_id: int) -> PlacementCase:
    for case in enumerate_placement_cases():
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown case id: {case_id}")


def write_placements(
    path: Path,
    allow_reflections: bool = settings.DEFAULT_ALLOW_REFLECTIONS,
) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for case in enumerate_placement_cases(allow_reflections=allow_reflections):
            handle.write(json.dumps(case.to_dict(), ensure_ascii=True) + "\n")
            count += 1
    return count


def write_schemes(
    case: PlacementCase,
    path: Path,
    max_depth: Optional[int],
    max_states: Optional[int],
    max_schemes: Optional[int],
) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for scheme in enumerate_solution_schemes(
            case,
            max_depth=max_depth,
            max_states=max_states,
        ):
            handle.write(json.dumps(scheme.to_dict(), ensure_ascii=True) + "\n")
            count += 1
            if max_schemes is not None and count >= max_schemes:
                break
    return count


def count_raw_order_types() -> int:
    count = 0
    for assigned_loci in product(LOCI, repeat=len(MARKERS)):
        marker_loci = dict(zip(MARKERS, assigned_loci))
        a_markers = [marker for marker in MARKERS if marker_loci[marker] == "A"]
        b_markers = [marker for marker in MARKERS if marker_loci[marker] == "B"]
        a_count = sum(1 for _ in ordered_partitions(a_markers))
        b_count = sum(1 for _ in ordered_partitions(b_markers))
        count += a_count * b_count
    return count


def command_summary(_: argparse.Namespace) -> int:
    cases = list(enumerate_placement_cases())
    direct_only = list(enumerate_placement_cases(allow_reflections=False))
    viable_order_types = {
        (
            case.marker_loci,
            case.a_interior_blocks,
            case.b_interior_blocks,
        )
        for case in cases
    }
    print(f"Raw symbolic marker order types: {count_raw_order_types()}")
    print(f"Order types with at least one nonzero orientation: {len(viable_order_types)}")
    print(f"Nonzero oriented placement cases: {len(cases)}")
    print(f"Direct-copy-only placement cases: {len(direct_only)}")
    return 0


def command_placements(args: argparse.Namespace) -> int:
    count = write_placements(
        args.output,
        allow_reflections=not args.direct_copies_only,
    )
    print(f"Wrote {count} placement cases to {args.output}")
    return 0


def command_show_case(args: argparse.Namespace) -> int:
    case = find_case(args.case_id)
    print(json.dumps(case.to_dict(), indent=2, ensure_ascii=True))
    return 0


def command_solve_case(args: argparse.Namespace) -> int:
    case = find_case(args.case_id)

    if args.output:
        count = write_schemes(
            case=case,
            path=args.output,
            max_depth=args.max_depth,
            max_states=args.max_states,
            max_schemes=args.max_schemes,
        )
        print(f"Wrote {count} solution schemes to {args.output}")
        return 0

    count = 0
    for scheme in enumerate_solution_schemes(
        case,
        max_depth=args.max_depth,
        max_states=args.max_states,
    ):
        print(json.dumps(scheme.to_dict(), ensure_ascii=True))
        count += 1
        if args.max_schemes is not None and count >= args.max_schemes:
            break

    print(f"Emitted schemes: {count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact symbolic contour placement and word-equation enumerator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary")
    summary_parser.set_defaults(function=command_summary)

    placements_parser = subparsers.add_parser("placements")
    placements_parser.add_argument("--output", type=Path, required=True)
    placements_parser.add_argument(
        "--direct-copies-only",
        action="store_true",
        help="Prune reflected-copy placements during generation.",
    )
    placements_parser.set_defaults(function=command_placements)

    show_parser = subparsers.add_parser("show-case")
    show_parser.add_argument("case_id", type=int)
    show_parser.set_defaults(function=command_show_case)

    solve_parser = subparsers.add_parser("solve-case")
    solve_parser.add_argument("case_id", type=int)
    solve_parser.add_argument("--output", type=Path)
    solve_parser.add_argument("--max-depth", type=int)
    solve_parser.add_argument("--max-states", type=int)
    solve_parser.add_argument("--max-schemes", type=int)
    solve_parser.set_defaults(function=command_solve_case)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
