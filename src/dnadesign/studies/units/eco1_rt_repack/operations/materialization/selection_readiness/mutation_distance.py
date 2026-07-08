"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/mutation_distance.py

Mutation-set distance helpers for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from collections.abc import Iterable


def canonical_mutation_tokens(value: object) -> frozenset[str]:
    """Return normalized canonical substitution tokens from a candidate row value."""

    return frozenset(_canonical_mutation_list(value))


def canonical_mutation_positions(value: object) -> frozenset[int]:
    """Return mutated positions from canonical substitution tokens."""

    return frozenset(_mutation_position(mutation) for mutation in _canonical_mutation_list(value))


def nearest_jaccard_distance[T](value: frozenset[T], selected_values: Iterable[frozenset[T]]) -> float | None:
    """Return the nearest Jaccard distance to already selected mutation sets."""

    distances = [jaccard_distance(value, selected) for selected in selected_values]
    return round(min(distances), 6) if distances else None


def nearest_shared_count[T](value: frozenset[T], selected_values: Iterable[frozenset[T]]) -> int | None:
    """Return the largest overlap count with already selected mutation sets."""

    shared_counts = [len(value & selected) for selected in selected_values]
    return max(shared_counts) if shared_counts else None


def jaccard_distance[T](left: frozenset[T], right: frozenset[T]) -> float:
    """Return Jaccard distance for two finite sets."""

    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def _canonical_mutation_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(entry) for entry in value]
    if isinstance(value, tuple):
        return [str(entry) for entry in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(loaded, (list, tuple)):
        return [str(entry) for entry in loaded]
    return [str(loaded)]


def _mutation_position(mutation: str) -> int:
    if len(mutation) < 3:
        raise ValueError(f"Malformed canonical mutation token: {mutation!r}")
    position_text = mutation[1:-1]
    if not position_text.isdigit():
        raise ValueError(f"Malformed canonical mutation token: {mutation!r}")
    return int(position_text)
