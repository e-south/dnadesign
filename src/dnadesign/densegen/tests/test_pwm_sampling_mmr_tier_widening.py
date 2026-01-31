"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_pwm_sampling_mmr_tier_widening.py

MMR tier widening behavior when early rungs are too small.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.adapters.sources import stage_a_selection
from dnadesign.densegen.src.adapters.sources.stage_a_types import FimoCandidate


def _motif_with_pwm() -> list[dict[str, float]]:
    return [
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]


def test_mmr_tier_widening_widens_instead_of_crashing() -> None:
    matrix = _motif_with_pwm()
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    ranked = []
    for idx in range(200):
        core = "ACGT"
        seq = f"TT{core}AA{idx:03d}"
        ranked.append(
            FimoCandidate(
                seq=seq,
                score=float(200 - idx),
                start=3,
                stop=6,
                strand="+",
                matched_sequence=core,
            )
        )

    selected, meta, diag = stage_a_selection._select_by_mmr(
        ranked,
        matrix=matrix,
        background=background,
        n_sites=20,
        alpha=0.9,
        shortlist_min=50,
        shortlist_factor=5,
        shortlist_max=None,
        tier_widening=[0.01, 0.1, 1.0],
    )

    assert len(selected) == 20
    assert diag.tier_fraction_used in (0.1, 1.0)
    assert all(cand.seq in meta for cand in selected)


def test_mmr_tier_widening_honors_shortlist_target() -> None:
    matrix = _motif_with_pwm()
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    ranked = []
    for idx in range(100):
        core = "ACGT"
        seq = f"TT{core}AA{idx:03d}"
        ranked.append(
            FimoCandidate(
                seq=seq,
                score=float(100 - idx),
                start=3,
                stop=6,
                strand="+",
                matched_sequence=core,
            )
        )

    selected, _meta, diag = stage_a_selection._select_by_mmr(
        ranked,
        matrix=matrix,
        background=background,
        n_sites=10,
        alpha=0.9,
        shortlist_min=10,
        shortlist_factor=5,
        shortlist_max=None,
        tier_widening=[0.2, 0.5, 1.0],
    )

    assert len(selected) == 10
    assert diag.tier_limit >= 50
