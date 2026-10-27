"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/sequence_constraints/test_constrained_sampler.py

Tests for constrained sequence generation with forbidden kmer constraints.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import random

import pytest

from dnadesign.densegen.src.core.sequence_constraints.sampler import (
    ConstrainedSequenceError,
    generate_constrained_sequence,
)


def test_constrained_sampler_generates_sequence_without_forbidden_kmers() -> None:
    seq = generate_constrained_sequence(
        length=24,
        gc_min=0.40,
        gc_max=0.60,
        forbid_kmers=["TTGACA", "TATAAT"],
        left_context="",
        right_context="",
        rng=random.Random(42),
    )
    assert len(seq) == 24
    assert "TTGACA" not in seq
    assert "TATAAT" not in seq


def test_constrained_sampler_respects_left_boundary_context() -> None:
    seq = generate_constrained_sequence(
        length=1,
        gc_min=0.0,
        gc_max=1.0,
        forbid_kmers=["TTGACA"],
        left_context="TTGAC",
        right_context="",
        rng=random.Random(7),
    )
    assert seq != "A"


def test_constrained_sampler_raises_explicit_error_when_constraints_infeasible() -> None:
    with pytest.raises(ConstrainedSequenceError, match="infeasible"):
        generate_constrained_sequence(
            length=1,
            gc_min=1.0,
            gc_max=1.0,
            forbid_kmers=["G", "C"],
            left_context="",
            right_context="",
            rng=random.Random(3),
        )
