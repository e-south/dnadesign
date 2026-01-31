"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_mmr_selection.py

MMR selection behavior for Stage-A PWM sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

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
        shortlist_min=3,
        shortlist_factor=1,
        shortlist_max=None,
        tier_widening=None,
    )
    assert [cand.seq for cand in selected] == ["AA", "AT"]
    assert meta["AA"].selection_rank == 1
    assert meta["AT"].selection_rank == 2
    assert diag.shortlist_k == 3


def test_score_norm_uses_percentile_rank() -> None:
    values = [10.0, 10.0, 30.0, 40.0]
    norm = stage_a_selection._score_norm(values)
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


def test_mmr_records_score_percentile_and_distance_norm() -> None:
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
        shortlist_min=3,
        shortlist_factor=1,
        shortlist_max=None,
        tier_widening=None,
    )
    assert [cand.seq for cand in selected] == ["AA", "AT"]
    assert meta["AA"].selection_score_percentile == pytest.approx(1.0)
    assert meta["AT"].selection_score_percentile == pytest.approx(0.25)
    assert meta["AT"].nearest_selected_distance_norm == pytest.approx(0.5)
