"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/pwm/test_pwm_sampling_mining.py

FIMO mining behavior for Stage-A PWM sampling.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import sample_pwm_sites
from dnadesign.densegen.src.core.stage_a.stage_a_types import PWMMotif
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.tests.pwm_sampling_fixtures import fixed_candidates_mining, selection_top_score

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_pwm_sampling_fimo_mining_consensus_includes_score_metadata() -> None:
    motif = PWMMotif(
        motif_id="M1",
        matrix=[
            {"A": 0.9, "C": 0.05, "G": 0.03, "T": 0.02},
            {"A": 0.9, "C": 0.05, "G": 0.03, "T": 0.02},
            {"A": 0.9, "C": 0.05, "G": 0.03, "T": 0.02},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )

    rng = np.random.default_rng(0)
    selected, meta = sample_pwm_sites(
        rng,
        motif,
        strategy="consensus",
        n_sites=1,
        mining=fixed_candidates_mining(batch_size=1, candidates=1, log_every_batches=1),
        selection=selection_top_score(),
        include_matched_sequence=True,
        return_metadata=True,
    )

    assert len(selected) == 1
    info = meta[selected[0]]
    assert info.best_hit_score > 0
    assert info.rank_within_regulator == 1
    assert info.tier in {0, 1, 2}
    assert info.fimo_matched_sequence is not None


def test_pwm_sampling_fimo_mining_shortfall_warns(caplog) -> None:
    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")
    motif = PWMMotif(
        motif_id="M2",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )

    rng = np.random.default_rng(0)
    with caplog.at_level(logging.WARNING):
        selected = sample_pwm_sites(
            rng,
            motif,
            strategy="stochastic",
            n_sites=5,
            mining=fixed_candidates_mining(batch_size=2, candidates=2, log_every_batches=1),
            selection=selection_top_score(),
        )
    assert 0 <= len(selected) < 5
    assert "shortfall" in caplog.text.lower()
