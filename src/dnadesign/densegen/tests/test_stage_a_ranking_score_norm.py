"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_ranking_score_norm.py

Stage-A ranking behavior for length-normalized selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.adapters.sources.stage_a import stage_a_pipeline
from dnadesign.densegen.src.adapters.sources.stage_a.stage_a_types import FimoCandidate


def _cand(seq: str, score: float) -> FimoCandidate:
    return FimoCandidate(
        seq=seq,
        score=score,
        start=1,
        stop=len(seq),
        strand="+",
        matched_sequence=seq,
    )


def test_rank_candidates_by_score_norm_prefers_higher_norm() -> None:
    candidates = [_cand("AAAAA", 10.0), _cand("AAA", 9.0)]
    denom_by_seq = {"AAAAA": 20.0, "AAA": 9.0}
    ranked = stage_a_pipeline._rank_candidates(
        candidates,
        rank_by="score_norm",
        score_norm_denominator_by_seq=denom_by_seq,
    )
    assert [cand.seq for cand in ranked] == ["AAA", "AAAAA"]


def test_rank_candidates_by_score_uses_raw_score() -> None:
    candidates = [_cand("AAAAA", 10.0), _cand("AAA", 9.0)]
    denom_by_seq = {"AAAAA": 20.0, "AAA": 9.0}
    ranked = stage_a_pipeline._rank_candidates(
        candidates,
        rank_by="score",
        score_norm_denominator_by_seq=denom_by_seq,
    )
    assert [cand.seq for cand in ranked] == ["AAAAA", "AAA"]
