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
    _collapse_by_core_identity,
    _core_sequence,
)


def test_core_sequence_prefers_matched_sequence() -> None:
    cand = FimoCandidate(seq="TTTAAA", score=5.0, start=4, stop=6, strand="+", matched_sequence="AAA")
    assert _core_sequence(cand) == "AAA"


def test_core_sequence_requires_matched_sequence() -> None:
    cand = FimoCandidate(seq="TTTAAA", score=5.0, start=4, stop=6, strand="+")
    with pytest.raises(ValueError, match="matched_sequence"):
        _core_sequence(cand)


def test_core_identity_collapse_prefers_highest_score() -> None:
    ranked = [
        FimoCandidate(seq="AAAAAA", score=9.0, start=1, stop=3, strand="+", matched_sequence="AAA"),
        FimoCandidate(seq="TTTAAA", score=8.0, start=4, stop=6, strand="+", matched_sequence="AAA"),
        FimoCandidate(seq="AAATTT", score=7.0, start=1, stop=3, strand="+", matched_sequence="AAA"),
    ]
    deduped, collapsed = _collapse_by_core_identity(ranked)
    assert [cand.seq for cand in deduped] == ["AAAAAA"]
    assert collapsed == 2
