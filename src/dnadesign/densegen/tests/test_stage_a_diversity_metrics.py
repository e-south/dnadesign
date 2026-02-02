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

from dnadesign.densegen.src.adapters.sources.stage_a_diversity import (
    _core_entropy,
    _core_hamming_knn,
    _core_hamming_nnd,
    _diversity_summary,
)
from dnadesign.densegen.src.adapters.sources.stage_a_metrics import (
    KnnSummary,
    PairwiseSummary,
    _mmr_objective,
)
from dnadesign.densegen.src.adapters.sources.stage_a_selection import (
    SelectionDiagnostics,
    _score_percentile_norm,
    _select_diversity_top_candidates,
    _select_diversity_upper_bound_candidates,
)
from dnadesign.densegen.src.adapters.sources.stage_a_types import FimoCandidate


def test_core_hamming_nnd_counts_and_median() -> None:
    cores = ["AAAA", "AAAT", "AATT"]
    summary = _core_hamming_nnd(cores, max_n=2500)
    assert summary is not None
    counts = summary.counts
    assert counts[1] == 3
    assert summary.median == 1.0
    assert summary.frac_le_1 == 1.0
    assert summary.p05 is not None
    assert summary.p95 is not None


def test_core_hamming_nnd_subsample_flag() -> None:
    cores = ["AAAA", "AAAT", "AATT"]
    summary = _core_hamming_nnd(cores, max_n=2)
    assert summary is not None
    assert summary.n == 2
    assert summary.subsampled is True


def test_core_hamming_knn_counts_and_median() -> None:
    cores = ["AAAA", "AAAT", "AATT", "TTTT"]
    summary = _core_hamming_knn(cores, k=2, max_n=2500)
    assert summary is not None
    assert summary.median == 2.0
    assert summary.frac_le_1 == 0.25


def test_core_entropy_values() -> None:
    cores = ["AAAA", "AAAT"]
    entropies = _core_entropy(cores)
    assert entropies == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_diversity_summary_scores() -> None:
    cores = ["AAAA", "AAAT"]
    scores = [1.0, 2.0]
    summary = _diversity_summary(
        top_candidates_cores=cores,
        diversified_candidates_cores=cores,
        top_candidates_scores=scores,
        diversified_candidates_scores=scores,
        max_diversity_upper_bound_cores=cores,
        max_diversity_upper_bound_scores=scores,
        pwm_theoretical_max_score=2.0,
        max_n=2500,
    )
    assert summary is not None
    overlap = summary.set_overlap_fraction
    assert overlap == 1.0
    n_swaps = summary.set_overlap_swaps
    assert n_swaps == 0
    core_hamming = summary.core_hamming
    pairwise = core_hamming.pairwise
    assert pairwise is not None
    base_pair = pairwise.top_candidates
    upper_pair = pairwise.max_diversity_upper_bound
    assert upper_pair is not None
    assert base_pair.bins
    assert base_pair.counts
    score_block = summary.score_quantiles
    base = score_block.top_candidates
    diversified = score_block.diversified_candidates
    assert base is not None
    assert diversified is not None
    assert base.p50 == 0.75
    assert diversified.p50 == 0.75
    assert summary.nnd_unweighted_median_top == 1.0
    assert summary.nnd_unweighted_median_diversified == 1.0
    assert summary.delta_nnd_unweighted_median == 0.0
    score_norm_summary = summary.score_norm_summary
    assert score_norm_summary is not None
    assert score_norm_summary.top_candidates is not None
    assert score_norm_summary.diversified_candidates is not None
    assert score_norm_summary.top_candidates.min == pytest.approx(0.5)
    assert score_norm_summary.top_candidates.median == pytest.approx(0.75)
    assert score_norm_summary.top_candidates.max == pytest.approx(1.0)


def test_diversity_summary_allows_zero_pwm_theoretical_max_score_for_zero_scores() -> None:
    cores = ["AAAA", "AAAT"]
    scores = [0.0, 0.0]
    summary = _diversity_summary(
        top_candidates_cores=cores,
        diversified_candidates_cores=cores,
        top_candidates_scores=scores,
        diversified_candidates_scores=scores,
        max_diversity_upper_bound_cores=cores,
        max_diversity_upper_bound_scores=scores,
        pwm_theoretical_max_score=0.0,
        max_n=2500,
    )
    assert summary is not None
    score_block = summary.score_quantiles
    base = score_block.top_candidates
    diversified = score_block.diversified_candidates
    assert base is not None
    assert diversified is not None
    assert base.p50 == 0.0
    assert diversified.p50 == 0.0


def test_diversity_summary_rejects_zero_pwm_theoretical_max_score_with_nonzero_scores() -> None:
    cores = ["AAAA", "AAAT"]
    scores = [0.0, 1.0]
    with pytest.raises(ValueError, match="pwm_theoretical_max_score"):
        _diversity_summary(
            top_candidates_cores=cores,
            diversified_candidates_cores=cores,
            top_candidates_scores=scores,
            diversified_candidates_scores=scores,
            max_diversity_upper_bound_cores=cores,
            max_diversity_upper_bound_scores=scores,
            pwm_theoretical_max_score=0.0,
            max_n=2500,
        )


def _base4_sequence(index: int, *, length: int = 6) -> str:
    bases = ["A", "C", "G", "T"]
    digits = []
    value = int(index)
    for _ in range(int(length)):
        digits.append(bases[value % 4])
        value //= 4
    return "".join(digits)


def test_diversity_summary_pairwise_is_exact_for_retained_sets() -> None:
    cores = [_base4_sequence(i, length=6) for i in range(200)]
    scores = [float(i) for i in range(200)]
    summary = _diversity_summary(
        top_candidates_cores=cores,
        diversified_candidates_cores=cores,
        top_candidates_scores=scores,
        diversified_candidates_scores=scores,
        max_diversity_upper_bound_cores=cores,
        max_diversity_upper_bound_scores=scores,
        pwm_theoretical_max_score=100.0,
        max_n=2500,
    )
    assert summary is not None
    pairwise = summary.core_hamming.pairwise
    assert pairwise is not None
    total_pairs = len(cores) * (len(cores) - 1) // 2
    assert pairwise.top_candidates.total_pairs == total_pairs
    assert pairwise.top_candidates.n_pairs == total_pairs
    assert pairwise.top_candidates.subsampled is False
    assert pairwise.diversified_candidates.n_pairs == total_pairs
    assert pairwise.diversified_candidates.subsampled is False


def _cand(seq: str, score: float) -> FimoCandidate:
    return FimoCandidate(
        seq=seq,
        score=score,
        start=1,
        stop=len(seq),
        strand="+",
        matched_sequence=seq,
    )


def test_mmr_objective_mean_utility() -> None:
    cores = ["AAAA", "AAAT", "TTTT"]
    scores = [3.0, 2.0, 1.0]
    scores_norm_map = _score_percentile_norm(scores)
    scores_norm = [scores_norm_map[score] for score in scores]
    top_candidates = _mmr_objective(
        cores=cores,
        scores=scores,
        scores_norm=scores_norm,
        alpha=0.5,
    )
    assert top_candidates == pytest.approx(0.125)
    diversified = _mmr_objective(
        cores=["AAAA", "TTTT", "AAAT"],
        scores=[3.0, 1.0, 2.0],
        scores_norm=[scores_norm_map[3.0], scores_norm_map[1.0], scores_norm_map[2.0]],
        alpha=0.5,
    )
    assert diversified == pytest.approx(0.13333333333333333)


def test_top_candidates_use_selection_pool_size() -> None:
    ranked = [_cand("AAAA", 4.0), _cand("AAAT", 3.0), _cand("AATT", 2.0), _cand("TTTT", 1.0)]
    diag = SelectionDiagnostics(
        selection_pool_size_final=2,
        selection_pool_rung_fraction_used=1.0,
        selection_pool_min_score_norm_used=None,
        selection_pool_capped=False,
        selection_pool_cap_value=None,
    )
    top_candidates = _select_diversity_top_candidates(
        ranked,
        selection_policy="mmr",
        selection_diag=diag,
        n_sites=3,
    )
    assert [cand.seq for cand in top_candidates] == ["AAAA", "AAAT"]


def test_top_candidates_use_full_pool_when_uncapped() -> None:
    ranked = [_cand("AAAA", 4.0), _cand("AAAT", 3.0), _cand("AATT", 2.0), _cand("TTTT", 1.0)]
    diag = SelectionDiagnostics(
        selection_pool_size_final=4,
        selection_pool_rung_fraction_used=1.0,
        selection_pool_min_score_norm_used=None,
        selection_pool_capped=False,
        selection_pool_cap_value=None,
    )
    top_candidates = _select_diversity_top_candidates(
        ranked,
        selection_policy="mmr",
        selection_diag=diag,
        n_sites=3,
    )
    assert [cand.seq for cand in top_candidates] == ["AAAA", "AAAT", "AATT"]


def test_upper_bound_candidates_prefer_diverse_cores() -> None:
    ranked = [_cand("AAAA", 4.0), _cand("AAAT", 3.0), _cand("AATT", 2.0), _cand("TTTT", 1.0)]
    diag = SelectionDiagnostics(
        selection_pool_size_final=4,
        selection_pool_rung_fraction_used=1.0,
        selection_pool_min_score_norm_used=None,
        selection_pool_capped=False,
        selection_pool_cap_value=None,
    )
    selected = _select_diversity_upper_bound_candidates(
        ranked,
        selection_policy="mmr",
        selection_diag=diag,
        n_sites=2,
        weights=None,
    )
    assert len(selected) == 2
    picked = {cand.seq for cand in selected}
    assert {"AAAA", "TTTT"} <= picked


def test_selection_diagnostics_rejects_negative_pool_size() -> None:
    with pytest.raises(ValueError):
        SelectionDiagnostics(
            selection_pool_size_final=-1,
            selection_pool_rung_fraction_used=1.0,
            selection_pool_min_score_norm_used=None,
            selection_pool_capped=False,
            selection_pool_cap_value=None,
        )


def test_knn_summary_rejects_mismatched_bins() -> None:
    with pytest.raises(ValueError):
        KnnSummary(
            bins=[0.0, 1.0],
            counts=[1],
            median=1.0,
            p05=1.0,
            p95=1.0,
            frac_le_1=1.0,
            n=1,
            subsampled=False,
            k=1,
        )


def test_pairwise_summary_rejects_mismatched_bins() -> None:
    with pytest.raises(ValueError):
        PairwiseSummary(
            bins=[0.0, 1.0],
            counts=[1, 2, 3],
            median=1.0,
            mean=1.0,
            p10=1.0,
            p90=1.0,
            n_pairs=1,
            total_pairs=1,
            subsampled=False,
        )
