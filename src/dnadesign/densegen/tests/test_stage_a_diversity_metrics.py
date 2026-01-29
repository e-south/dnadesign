"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_diversity_metrics.py

Unit tests for Stage-A diversity metric helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import (
    FimoCandidate,
    _core_entropy,
    _core_hamming_nnd,
    _diversity_summary,
    _select_diversity_baseline_candidates,
)


def test_core_hamming_nnd_counts_and_median() -> None:
    cores = ["AAAA", "AAAT", "AATT"]
    summary = _core_hamming_nnd(cores, max_n=2500)
    assert summary is not None
    counts = summary.get("counts")
    assert isinstance(counts, list)
    assert counts[1] == 3
    assert summary.get("median") == 1.0
    assert summary.get("frac_le_1") == 1.0
    assert summary.get("p05") is not None
    assert summary.get("p95") is not None


def test_core_hamming_nnd_subsample_flag() -> None:
    cores = ["AAAA", "AAAT", "AATT"]
    summary = _core_hamming_nnd(cores, max_n=2)
    assert summary is not None
    assert summary.get("n") == 2
    assert summary.get("subsampled") is True


def test_core_entropy_values() -> None:
    cores = ["AAAA", "AAAT"]
    entropies = _core_entropy(cores)
    assert entropies == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_diversity_summary_scores() -> None:
    cores = ["AAAA", "AAAT"]
    scores = [1.0, 2.0]
    summary = _diversity_summary(
        baseline_cores=cores,
        actual_cores=cores,
        baseline_scores=scores,
        actual_scores=scores,
        max_n=2500,
    )
    assert summary is not None
    overlap = summary.get("overlap_actual_fraction")
    assert overlap == 1.0
    pairwise = summary.get("pairwise_median")
    assert isinstance(pairwise, dict)
    assert pairwise.get("baseline") is not None
    assert pairwise.get("actual") is not None
    score_block = summary.get("score_baseline_vs_actual")
    assert isinstance(score_block, dict)
    assert score_block.get("baseline_median") == 1.5
    assert score_block.get("actual_median") == 1.5


def _cand(seq: str, score: float) -> FimoCandidate:
    return FimoCandidate(
        seq=seq,
        score=score,
        start=1,
        stop=len(seq),
        strand="+",
        matched_sequence=seq,
    )


def test_baseline_candidates_use_shortlist_k() -> None:
    ranked = [_cand("AAAA", 4.0), _cand("AAAT", 3.0), _cand("AATT", 2.0), _cand("TTTT", 1.0)]
    diag = {"shortlist_k": 2, "tier_limit": 4}
    baseline = _select_diversity_baseline_candidates(
        ranked,
        selection_policy="mmr",
        selection_diag=diag,
        n_sites=3,
    )
    assert [cand.seq for cand in baseline] == ["AAAA", "AAAT"]


def test_baseline_candidates_use_tier_limit_when_shortlist_missing() -> None:
    ranked = [_cand("AAAA", 4.0), _cand("AAAT", 3.0), _cand("AATT", 2.0), _cand("TTTT", 1.0)]
    diag = {"shortlist_k": None, "tier_limit": 2}
    baseline = _select_diversity_baseline_candidates(
        ranked,
        selection_policy="mmr",
        selection_diag=diag,
        n_sites=3,
    )
    assert [cand.seq for cand in baseline] == ["AAAA", "AAAT"]
