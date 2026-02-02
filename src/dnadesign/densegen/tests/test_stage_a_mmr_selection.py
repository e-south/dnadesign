"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_mmr_selection.py

MMR selection behavior for Stage-A PWM sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import itertools
import logging

import pytest

from dnadesign.densegen.src.adapters.sources import stage_a_selection
from dnadesign.densegen.src.adapters.sources.stage_a_types import FimoCandidate


def _cand(seq: str, score: float) -> FimoCandidate:
    return FimoCandidate(
        seq=seq,
        score=score,
        start=1,
        stop=len(seq),
        strand="+",
        matched_sequence=seq,
    )


def _seqs(count: int, length: int = 3) -> list[str]:
    bases = ["A", "C", "G", "T"]
    seqs: list[str] = []
    for core in itertools.product(bases, repeat=length):
        seqs.append("".join(core))
        if len(seqs) >= count:
            break
    if len(seqs) < count:
        raise ValueError("Insufficient unique sequences for requested count.")
    return seqs


def test_mmr_prefers_low_ic_mismatch_when_scores_equal() -> None:
    matrix = [
        {"A": 0.97, "C": 0.01, "G": 0.01, "T": 0.01},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    ranked = [
        _cand("AA", 10.0),
        _cand("AT", 9.0),
        _cand("TA", 9.0),
    ]
    selected, meta, diag = stage_a_selection._select_by_mmr(
        ranked,
        matrix=matrix,
        background=background,
        n_sites=2,
        alpha=0.5,
        pool_min_score_norm=None,
        pool_max_candidates=None,
        relevance_norm="minmax_raw_score",
        tier_fractions=None,
        pwm_theoretical_max_score=10.0,
    )
    assert [cand.seq for cand in selected] == ["AA", "AT"]
    assert meta["AA"].selection_rank == 1
    assert meta["AT"].selection_rank == 2
    assert diag.selection_pool_size_final == 3


def test_score_norm_uses_percentile_rank() -> None:
    values = [10.0, 10.0, 30.0, 40.0]
    norm = stage_a_selection._score_percentile_norm(values)
    assert norm[10.0] == pytest.approx(0.166666, rel=1e-5)
    assert norm[30.0] == pytest.approx(0.666666, rel=1e-5)
    assert norm[40.0] == pytest.approx(1.0, rel=1e-5)


def test_pwm_tolerant_weights_use_background_information() -> None:
    matrix = [
        {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1},
        {"A": 0.97, "C": 0.01, "G": 0.01, "T": 0.01},
    ]
    background = {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1}
    weights = stage_a_selection._pwm_tolerant_weights(matrix, background=background)
    assert weights[0] == pytest.approx(1.0)
    assert weights[0] > weights[1]


def test_mmr_records_score_norm_and_distance_norm() -> None:
    matrix = [
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    ranked = [
        _cand("AA", 10.0),
        _cand("AT", 9.0),
        _cand("TA", 9.0),
    ]
    selected, meta, _diag = stage_a_selection._select_by_mmr(
        ranked,
        matrix=matrix,
        background=background,
        n_sites=2,
        alpha=0.5,
        pool_min_score_norm=None,
        pool_max_candidates=None,
        relevance_norm="minmax_raw_score",
        tier_fractions=None,
        pwm_theoretical_max_score=10.0,
    )
    assert [cand.seq for cand in selected] == ["AA", "AT"]
    assert meta["AA"].selection_score_norm == pytest.approx(1.0)
    assert meta["AT"].selection_score_norm == pytest.approx(0.9)
    assert meta["AT"].nearest_selected_distance_norm == pytest.approx(0.5)


def test_mmr_uses_per_candidate_score_norm_denominator() -> None:
    matrix = [
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    ranked = [
        _cand("AAAA", 10.0),
        _cand("CCCC", 10.0),
        _cand("GGGG", 10.0),
    ]
    score_norm_denominator_by_seq = {"AAAA": 20.0, "CCCC": 10.0, "GGGG": 40.0}
    selected, meta, _diag = stage_a_selection._select_by_mmr(
        ranked,
        matrix=matrix,
        background=background,
        n_sites=2,
        alpha=1.0,
        pool_min_score_norm=None,
        pool_max_candidates=None,
        relevance_norm="minmax_raw_score",
        tier_fractions=None,
        pwm_theoretical_max_score=20.0,
        score_norm_denominator_by_seq=score_norm_denominator_by_seq,
    )
    assert [cand.seq for cand in selected] == ["CCCC", "AAAA"]
    assert meta["CCCC"].selection_score_norm == pytest.approx(1.0)
    assert meta["AAAA"].selection_score_norm == pytest.approx(0.5)


def test_mmr_selection_score_norm_uses_pwm_ratio_with_percentile_relevance() -> None:
    matrix = [
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    ranked = [
        _cand("AA", 100.0),
        _cand("AC", 90.0),
    ]
    selected, meta, _diag = stage_a_selection._select_by_mmr(
        ranked,
        matrix=matrix,
        background=background,
        n_sites=2,
        alpha=0.5,
        pool_min_score_norm=0.1,
        pool_max_candidates=None,
        relevance_norm="percentile",
        tier_fractions=[0.1, 0.2, 0.3],
        pwm_theoretical_max_score=100.0,
    )
    assert [cand.seq for cand in selected] == ["AA", "AC"]
    assert meta["AC"].selection_score_norm == pytest.approx(0.9)


def test_mmr_pool_includes_full_rung_without_cap() -> None:
    matrix = [
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    seqs = _seqs(12, length=3)
    ranked = [_cand(seqs[i], float(100 - i)) for i in range(12)]
    selected, _meta, diag = stage_a_selection._select_by_mmr(
        ranked,
        matrix=matrix,
        background=background,
        n_sites=3,
        alpha=0.5,
        pool_min_score_norm=None,
        pool_max_candidates=None,
        relevance_norm="minmax_raw_score",
        tier_fractions=[0.1, 0.2, 0.3],
        pwm_theoretical_max_score=100.0,
    )
    assert len(selected) == 3
    assert diag.selection_pool_size_final == 4
    assert diag.selection_pool_capped is False


def test_mmr_pool_min_score_norm_is_report_only() -> None:
    matrix = [
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    seqs = _seqs(12, length=3)
    ranked = [_cand(seqs[i], float(100 - i)) for i in range(12)]
    selected, _meta, diag = stage_a_selection._select_by_mmr(
        ranked,
        matrix=matrix,
        background=background,
        n_sites=3,
        alpha=0.5,
        pool_min_score_norm=0.99,
        pool_max_candidates=None,
        relevance_norm="minmax_raw_score",
        tier_fractions=[0.1, 0.2, 0.3],
        pwm_theoretical_max_score=100.0,
    )
    assert len(selected) == 3
    assert diag.selection_pool_size_final == 4
    assert diag.selection_pool_rung_fraction_used == pytest.approx(0.3)


def test_mmr_pool_shortfall_warns(caplog: pytest.LogCaptureFixture) -> None:
    matrix = [
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    seqs = _seqs(12, length=3)
    ranked = [_cand(seqs[i], float(100 - i)) for i in range(12)]
    with caplog.at_level(logging.WARNING):
        selected, _meta, _diag = stage_a_selection._select_by_mmr(
            ranked,
            matrix=matrix,
            background=background,
            n_sites=5,
            alpha=0.5,
            pool_min_score_norm=None,
            pool_max_candidates=3,
            relevance_norm="minmax_raw_score",
            tier_fractions=[0.1, 0.2, 0.3],
            pwm_theoretical_max_score=100.0,
        )
    assert len(selected) == 3
    assert "MMR degenerate" in caplog.text


def test_mmr_pool_degenerate_warns_when_equal_to_n_sites(caplog: pytest.LogCaptureFixture) -> None:
    matrix = [
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    seqs = _seqs(12, length=3)
    ranked = [_cand(seqs[i], float(100 - i)) for i in range(12)]
    with caplog.at_level(logging.WARNING):
        selected, _meta, _diag = stage_a_selection._select_by_mmr(
            ranked,
            matrix=matrix,
            background=background,
            n_sites=3,
            alpha=0.5,
            pool_min_score_norm=None,
            pool_max_candidates=None,
            relevance_norm="minmax_raw_score",
            tier_fractions=[0.2, 0.3, 0.4],
            pwm_theoretical_max_score=100.0,
        )
    assert [cand.seq for cand in selected] == seqs[:3]
    assert "MMR degenerate" in caplog.text


def test_mmr_pool_cap_is_deterministic() -> None:
    matrix = [
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    seqs = _seqs(10, length=3)
    ranked = [_cand(seqs[i], float(100 - i)) for i in range(10)]
    selected, _meta, diag = stage_a_selection._select_by_mmr(
        ranked,
        matrix=matrix,
        background=background,
        n_sites=5,
        alpha=0.5,
        pool_min_score_norm=None,
        pool_max_candidates=4,
        relevance_norm="minmax_raw_score",
        tier_fractions=[0.1, 0.2, 0.3],
        pwm_theoretical_max_score=100.0,
    )
    assert len(selected) == 4
    assert diag.selection_pool_size_final == 4
    assert diag.selection_pool_capped is True
    assert diag.selection_pool_cap_value == 4


def test_selection_score_norm_clips_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    matrix = [
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    ranked = [
        _cand("AA", 10.0),
        _cand("AC", 9.0),
    ]
    with caplog.at_level(logging.ERROR):
        selected, meta, diag = stage_a_selection._select_by_mmr(
            ranked,
            matrix=matrix,
            background=background,
            n_sites=1,
            alpha=0.5,
            pool_min_score_norm=None,
            pool_max_candidates=None,
            relevance_norm="minmax_raw_score",
            tier_fractions=None,
            pwm_theoretical_max_score=5.0,
        )
    assert [cand.seq for cand in selected] == ["AA"]
    assert meta["AA"].selection_score_norm == pytest.approx(1.0)
    assert diag.selection_score_norm_clipped is True
    assert diag.selection_score_norm_max_raw == pytest.approx(2.0)
    assert "score_norm exceeded" in caplog.text
