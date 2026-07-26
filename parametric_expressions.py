#!/usr/bin/env python3
"""Small AST for exact finite and nested-power word families.

This module is deliberately independent from geometry, point angles and copy
parity.  It only represents words built from nonempty word parameters,
concatenation, path inversion and integer repetition parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Union

import symbolic_enumerator as base


@dataclass(frozen=True)
class Atom:
    name: str
    inverse: bool = False


@dataclass(frozen=True)
class Concat:
    parts: Tuple["WordExpression", ...]


@dataclass(frozen=True)
class Repeat:
    body: "WordExpression"
    parameter: str
    minimum: int = 0


WordExpression = Union[Atom, Concat, Repeat]


def atom(name: str, inverse: bool = False) -> WordExpression:
    return Atom(str(name), bool(inverse))


def concat(*parts: WordExpression) -> WordExpression:
    flat = []
    for part in parts:
        if isinstance(part, Concat):
            flat.extend(part.parts)
        else:
            flat.append(part)
    if not flat:
        return Concat(())
    if len(flat) == 1:
        return flat[0]
    return Concat(tuple(flat))


def from_word(word: Sequence[base.Literal]) -> WordExpression:
    return concat(*(atom(literal.variable, literal.inverse) for literal in word))


def inverse(expression: WordExpression) -> WordExpression:
    if isinstance(expression, Atom):
        return Atom(expression.name, not expression.inverse)
    if isinstance(expression, Concat):
        return concat(*(inverse(part) for part in reversed(expression.parts)))
    if isinstance(expression, Repeat):
        return Repeat(inverse(expression.body), expression.parameter, expression.minimum)
    raise TypeError(type(expression).__name__)


def substitute(
    expression: WordExpression,
    mapping: Mapping[str, WordExpression],
) -> WordExpression:
    if isinstance(expression, Atom):
        if expression.name not in mapping:
            return expression
        replacement = mapping[expression.name]
        return inverse(replacement) if expression.inverse else replacement
    if isinstance(expression, Concat):
        return concat(*(substitute(part, mapping) for part in expression.parts))
    if isinstance(expression, Repeat):
        return Repeat(
            substitute(expression.body, mapping),
            expression.parameter,
            expression.minimum,
        )
    raise TypeError(type(expression).__name__)


def substitute_word(
    word: Sequence[base.Literal],
    mapping: Mapping[str, WordExpression],
) -> WordExpression:
    return substitute(from_word(word), mapping)


def expand(
    expression: WordExpression,
    exponents: Mapping[str, int],
) -> base.Word:
    if isinstance(expression, Atom):
        return (base.Literal(expression.name, expression.inverse),)
    if isinstance(expression, Concat):
        output = []
        for part in expression.parts:
            output.extend(expand(part, exponents))
        return tuple(output)
    if isinstance(expression, Repeat):
        count = int(exponents[expression.parameter])
        if count < expression.minimum:
            raise ValueError(
                f"Exponent {expression.parameter}={count} is below minimum {expression.minimum}"
            )
        body = expand(expression.body, exponents)
        return tuple(body * count)
    raise TypeError(type(expression).__name__)


def expanded_length(
    expression: WordExpression,
    exponents: Mapping[str, int],
    *,
    cap: int | None = None,
) -> int:
    """Return the expanded word length without constructing the word.

    When ``cap`` is provided, any value greater than the cap is returned as
    ``cap + 1``.  This prevents accidental construction of enormous nested
    powers in inclusion checks and expansion planning.
    """
    if cap is not None and cap < 0:
        raise ValueError("cap must be nonnegative or None")
    if isinstance(expression, Atom):
        return 1
    if isinstance(expression, Concat):
        total = 0
        for part in expression.parts:
            remaining = None if cap is None else max(0, cap - total)
            total += expanded_length(part, exponents, cap=remaining)
            if cap is not None and total > cap:
                return cap + 1
        return total
    if isinstance(expression, Repeat):
        count = int(exponents[expression.parameter])
        if count < expression.minimum:
            raise ValueError(
                f"Exponent {expression.parameter}={count} is below minimum {expression.minimum}"
            )
        body_length = expanded_length(expression.body, exponents, cap=cap)
        if cap is not None:
            if body_length > cap:
                return cap + 1
            if body_length and count > cap // body_length:
                return cap + 1
        return body_length * count
    raise TypeError(type(expression).__name__)


def exponent_parameters(expression: WordExpression) -> Tuple[str, ...]:
    output = []
    seen = set()

    def visit(item: WordExpression) -> None:
        if isinstance(item, Atom):
            return
        if isinstance(item, Concat):
            for part in item.parts:
                visit(part)
            return
        if isinstance(item, Repeat):
            if item.parameter not in seen:
                seen.add(item.parameter)
                output.append(item.parameter)
            visit(item.body)
            return
        raise TypeError(type(item).__name__)

    visit(expression)
    return tuple(output)


def repeat_nesting_depth(expression: WordExpression) -> int:
    if isinstance(expression, Atom):
        return 0
    if isinstance(expression, Concat):
        return max((repeat_nesting_depth(part) for part in expression.parts), default=0)
    if isinstance(expression, Repeat):
        return 1 + repeat_nesting_depth(expression.body)
    raise TypeError(type(expression).__name__)


def atom_names(expression: WordExpression) -> Tuple[str, ...]:
    output = []
    seen = set()

    def visit(item: WordExpression) -> None:
        if isinstance(item, Atom):
            if item.name not in seen:
                seen.add(item.name)
                output.append(item.name)
            return
        if isinstance(item, Concat):
            for part in item.parts:
                visit(part)
            return
        if isinstance(item, Repeat):
            visit(item.body)
            return
        raise TypeError(type(item).__name__)

    visit(expression)
    return tuple(output)


def rename_atoms(
    expression: WordExpression,
    renaming: Mapping[str, str],
) -> WordExpression:
    if isinstance(expression, Atom):
        return Atom(renaming.get(expression.name, expression.name), expression.inverse)
    if isinstance(expression, Concat):
        return concat(*(rename_atoms(part, renaming) for part in expression.parts))
    if isinstance(expression, Repeat):
        return Repeat(
            rename_atoms(expression.body, renaming),
            expression.parameter,
            expression.minimum,
        )
    raise TypeError(type(expression).__name__)


def to_text(expression: WordExpression) -> str:
    if isinstance(expression, Atom):
        return expression.name + ("^-1" if expression.inverse else "")
    if isinstance(expression, Concat):
        if not expression.parts:
            return "1"
        return " ".join(to_text(part) for part in expression.parts)
    if isinstance(expression, Repeat):
        body = to_text(expression.body)
        if isinstance(expression.body, Atom) and not expression.body.inverse:
            return f"{body}^{expression.parameter}"
        return f"({body})^{expression.parameter}"
    raise TypeError(type(expression).__name__)


def to_dict(expression: WordExpression) -> Dict[str, object]:
    if isinstance(expression, Atom):
        return {
            "kind": "atom",
            "name": expression.name,
            "inverse": expression.inverse,
        }
    if isinstance(expression, Concat):
        return {
            "kind": "concat",
            "parts": [to_dict(part) for part in expression.parts],
        }
    if isinstance(expression, Repeat):
        return {
            "kind": "repeat",
            "parameter": expression.parameter,
            "minimum": expression.minimum,
            "body": to_dict(expression.body),
        }
    raise TypeError(type(expression).__name__)
