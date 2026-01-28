"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_mmr_selection.py

MMR selection behavior for Stage-A PWM sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.adapters.sources import pwm_sampling


def _motif_with_log_odds() -> pwm_sampling.PWMMotif:
    matrix = [
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]
    log_odds = [
        {"A": 2.0, "C": 0.0, "G": 0.0, "T": 0.0},
        {"A": 2.0, "C": 0.0, "G": 0.0, "T": 0.0},
        {"A": 2.0, "C": 5.0, "G": 0.0, "T": 1.0},
    ]
    return pwm_sampling.PWMMotif(
        motif_id="M1",
        matrix=matrix,
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        log_odds=log_odds,
    )


def _cand(seq: str, score: float) -> pwm_sampling.FimoCandidate:
    return pwm_sampling.FimoCandidate(
        seq=seq,
        score=score,
        start=1,
        stop=len(seq),
        strand="+",
        matched_sequence=seq,
    )


def test_mmr_prefers_diverse_when_scores_equal() -> None:
    motif = _motif_with_log_odds()
    ranked = [
        _cand("AAA", 6.0),
        _cand("AAT", 5.0),
        _cand("CCC", 5.0),
    ]
    selected, meta, diag = pwm_sampling._select_by_mmr(
        ranked,
        motif=motif,
        n_sites=2,
        alpha=0.9,
        shortlist_min=3,
        shortlist_factor=1,
        shortlist_max=None,
        tier_widening=None,
    )
    assert [cand.seq for cand in selected] == ["AAA", "CCC"]
    assert meta["AAA"]["selection_rank"] == 1
    assert meta["CCC"]["selection_rank"] == 2
    assert diag["shortlist_k"] == 3
