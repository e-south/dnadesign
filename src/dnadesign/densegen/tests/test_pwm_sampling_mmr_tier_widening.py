"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_pwm_sampling_mmr_tier_widening.py

MMR tier widening behavior when early rungs are too small.

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
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]
    log_odds = [
        {"A": 0.1, "C": 0.0, "G": 0.0, "T": 0.0},
        {"A": 0.0, "C": 0.1, "G": 0.0, "T": 0.0},
        {"A": 0.0, "C": 0.0, "G": 0.1, "T": 0.0},
        {"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.1},
    ]
    return pwm_sampling.PWMMotif(
        motif_id="test",
        matrix=matrix,
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        log_odds=log_odds,
    )


def test_mmr_tier_widening_widens_instead_of_crashing() -> None:
    motif = _motif_with_log_odds()
    ranked = []
    for idx in range(200):
        core = "ACGT"
        seq = f"TT{core}AA{idx:03d}"
        ranked.append(
            pwm_sampling.FimoCandidate(
                seq=seq,
                score=float(200 - idx),
                start=3,
                stop=6,
                strand="+",
                matched_sequence=core,
            )
        )

    selected, meta, diag = pwm_sampling._select_by_mmr(
        ranked,
        motif=motif,
        n_sites=20,
        alpha=0.9,
        shortlist_min=50,
        shortlist_factor=5,
        shortlist_max=None,
        tier_widening=[0.01, 0.1, 1.0],
    )

    assert len(selected) == 20
    assert diag["tier_fraction_used"] in (0.1, 1.0)
    assert all(cand.seq in meta for cand in selected)


def test_mmr_tier_widening_honors_shortlist_target() -> None:
    motif = _motif_with_log_odds()
    ranked = []
    for idx in range(100):
        core = "ACGT"
        seq = f"TT{core}AA{idx:03d}"
        ranked.append(
            pwm_sampling.FimoCandidate(
                seq=seq,
                score=float(100 - idx),
                start=3,
                stop=6,
                strand="+",
                matched_sequence=core,
            )
        )

    selected, _meta, diag = pwm_sampling._select_by_mmr(
        ranked,
        motif=motif,
        n_sites=10,
        alpha=0.9,
        shortlist_min=10,
        shortlist_factor=5,
        shortlist_max=None,
        tier_widening=[0.2, 0.5, 1.0],
        ensure_shortlist_target=True,
    )

    assert len(selected) == 10
    assert diag["tier_limit"] >= 50
