"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/sequence_constraints/sampler.py

Deterministic constrained DNA sequence generator with hard kmer constraints.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import random

import numpy as np


class ConstrainedSequenceError(ValueError):
    """Raised when constrained sequence generation is infeasible."""


def _normalize_dna(value: str, *, label: str) -> str:
    seq = str(value or "").strip().upper()
    if not seq:
        return ""
    if any(ch not in {"A", "C", "G", "T"} for ch in seq):
        raise ConstrainedSequenceError(f"{label} must contain only A/C/G/T characters.")
    return seq


def _normalize_patterns(forbid_kmers: list[str] | None) -> list[str]:
    if not forbid_kmers:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in forbid_kmers:
        motif = _normalize_dna(raw, label="forbid_kmers entry")
        if not motif:
            continue
        if motif in seen:
            continue
        seen.add(motif)
        ordered.append(motif)
    return ordered


def _gc_bounds(*, length: int, gc_min: float, gc_max: float) -> tuple[int, int]:
    if not (0.0 <= float(gc_min) <= 1.0 and 0.0 <= float(gc_max) <= 1.0):
        raise ConstrainedSequenceError("gc_min and gc_max must be within [0, 1].")
    if float(gc_min) > float(gc_max):
        raise ConstrainedSequenceError("gc_min must be <= gc_max.")
    return int(math.ceil(length * float(gc_min))), int(math.floor(length * float(gc_max)))


def _random_base_order(rng, bases: list[str]) -> list[str]:
    ordered = list(bases)
    if isinstance(rng, random.Random):
        rng.shuffle(ordered)
        return ordered
    if isinstance(rng, np.random.Generator):
        perm = rng.permutation(len(ordered))
        return [ordered[int(i)] for i in perm]
    random.shuffle(ordered)
    return ordered


def _boundary_violation(*, seq: str, right_context: str, patterns: list[str]) -> tuple[str, int] | None:
    if not right_context or not patterns:
        return None
    combined = seq + right_context
    seq_len = len(seq)
    for motif in patterns:
        k = len(motif)
        start_lo = max(0, seq_len - k + 1)
        start_hi = min(seq_len - 1, len(combined) - k)
        if start_hi < start_lo:
            continue
        for start in range(start_lo, start_hi + 1):
            if combined[start : start + k] == motif:
                return motif, int(start)
    return None


def generate_constrained_sequence(
    *,
    length: int,
    gc_min: float = 0.0,
    gc_max: float = 1.0,
    forbid_kmers: list[str] | None = None,
    left_context: str = "",
    right_context: str = "",
    rng=None,
    max_search_nodes: int = 200_000,
) -> str:
    seq_len = int(length)
    if seq_len < 0:
        raise ConstrainedSequenceError("length must be >= 0.")
    if seq_len == 0:
        return ""
    if int(max_search_nodes) <= 0:
        raise ConstrainedSequenceError("max_search_nodes must be > 0.")

    patterns = _normalize_patterns(forbid_kmers)
    left = _normalize_dna(left_context, label="left_context")
    right = _normalize_dna(right_context, label="right_context")
    gc_min_count, gc_max_count = _gc_bounds(length=seq_len, gc_min=gc_min, gc_max=gc_max)
    if gc_min_count > gc_max_count:
        raise ConstrainedSequenceError(
            f"infeasible constrained sequence generation: length={seq_len} "
            f"gc_min_count={gc_min_count} gc_max_count={gc_max_count}."
        )

    max_pattern_len = max((len(motif) for motif in patterns), default=0)
    nodes = 0
    bases = ["A", "C", "G", "T"]
    generator = rng if rng is not None else random.Random(0)

    def _suffix_hits_forbidden(candidate_prefix: str) -> bool:
        if not patterns:
            return False
        context = (left + candidate_prefix)[-max_pattern_len:]
        for motif in patterns:
            if context.endswith(motif):
                return True
        return False

    def _search(*, pos: int, prefix: str, gc_count: int) -> str | None:
        nonlocal nodes
        nodes += 1
        if nodes > int(max_search_nodes):
            raise ConstrainedSequenceError(
                "infeasible constrained sequence generation: search-node limit reached "
                f"(nodes>{int(max_search_nodes)})."
            )
        if pos == seq_len:
            hit = _boundary_violation(seq=prefix, right_context=right, patterns=patterns)
            if hit is not None:
                return None
            return prefix

        remaining = seq_len - (pos + 1)
        for base in _random_base_order(generator, bases):
            next_gc = gc_count + (1 if base in {"G", "C"} else 0)
            if next_gc > gc_max_count:
                continue
            if next_gc + remaining < gc_min_count:
                continue
            candidate = prefix + base
            if _suffix_hits_forbidden(candidate):
                continue
            solved = _search(pos=pos + 1, prefix=candidate, gc_count=next_gc)
            if solved is not None:
                return solved
        return None

    solved = _search(pos=0, prefix="", gc_count=0)
    if solved is None:
        raise ConstrainedSequenceError(
            "infeasible constrained sequence generation: no sequence satisfies constraints "
            f"(length={seq_len}, gc_min={gc_min}, gc_max={gc_max}, "
            f"forbidden={len(patterns)}, left_context={left!r}, right_context={right!r})."
        )
    return solved
