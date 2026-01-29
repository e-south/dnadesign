"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/tests/test_pwm_log_odds_smoothing.py

Checks PWM log-odds smoothing and sampling stability.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import (
    PWMMotif,
    build_log_odds,
    sample_pwm_sites,
    score_sequence,
)
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.tests.pwm_sampling_fixtures import fixed_candidates_mining, selection_top_score

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


def test_pwm_log_odds_smoothing_finite() -> None:
    matrix = [
        {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
        {"A": 0.0, "C": 1.0, "G": 0.0, "T": 0.0},
        {"A": 0.0, "C": 0.0, "G": 1.0, "T": 0.0},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    log_odds = build_log_odds(matrix, background)
    assert all(np.isfinite(val) for row in log_odds for val in row.values())

    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")

    motif = PWMMotif(motif_id="M1", matrix=matrix, background=background)
    rng = np.random.default_rng(0)
    sites = sample_pwm_sites(
        rng,
        motif,
        strategy="stochastic",
        n_sites=1,
        mining=fixed_candidates_mining(batch_size=5, candidates=5, log_every_batches=1),
        selection=selection_top_score(),
    )
    assert len(sites) == 1
    core = sites[0][: len(matrix)]
    score = score_sequence(core, matrix, background=background)
    assert np.isfinite(score)
