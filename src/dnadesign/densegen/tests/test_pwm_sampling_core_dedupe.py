"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_pwm_sampling_core_dedupe.py

Core dedupe helpers for Stage-A PWM sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import (
    FimoCandidate,
    _core_sequence,
    _dedupe_by_core,
    _hamming_distance,
)


def test_core_sequence_prefers_matched_sequence() -> None:
    cand = FimoCandidate(seq="TTTAAA", score=5.0, start=4, stop=6, strand="+", matched_sequence="AAA")
    assert _core_sequence(cand) == "AAA"


def test_core_sequence_uses_start_stop_slice() -> None:
    cand = FimoCandidate(seq="TTTAAA", score=5.0, start=4, stop=6, strand="+")
    assert _core_sequence(cand) == "AAA"


def test_core_sequence_invalid_bounds_raises() -> None:
    cand = FimoCandidate(seq="TTTAAA", score=5.0, start=7, stop=8, strand="+")
    with pytest.raises(ValueError, match="Core sequence bounds"):
        _core_sequence(cand)


def test_dedupe_by_core_prefers_highest_score() -> None:
    ranked = [
        FimoCandidate(seq="AAAAAA", score=9.0, start=1, stop=3, strand="+"),
        FimoCandidate(seq="TTTAAA", score=8.0, start=4, stop=6, strand="+"),
        FimoCandidate(seq="AAATTT", score=7.0, start=1, stop=3, strand="+"),
    ]
    deduped = _dedupe_by_core(ranked, min_core_hamming_distance=None)
    assert [cand.seq for cand in deduped] == ["AAAAAA"]


def test_min_core_hamming_distance_filters_neighbors() -> None:
    ranked = [
        FimoCandidate(seq="AAA", score=9.0, start=1, stop=3, strand="+"),
        FimoCandidate(seq="AAT", score=8.0, start=1, stop=3, strand="+"),
        FimoCandidate(seq="TTT", score=7.0, start=1, stop=3, strand="+"),
    ]
    deduped = _dedupe_by_core(ranked, min_core_hamming_distance=2)
    assert [cand.seq for cand in deduped] == ["AAA", "TTT"]


def test_hamming_distance_requires_equal_length() -> None:
    with pytest.raises(ValueError, match="length"):
        _hamming_distance("AAA", "AAAA")
