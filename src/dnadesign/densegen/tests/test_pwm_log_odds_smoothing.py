from __future__ import annotations

import numpy as np

from dnadesign.densegen.src.adapters.sources.pwm_sampling import (
    PWMMotif,
    build_log_odds,
    sample_pwm_sites,
    score_sequence,
)


def test_pwm_log_odds_smoothing_finite() -> None:
    matrix = [
        {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
        {"A": 0.0, "C": 1.0, "G": 0.0, "T": 0.0},
        {"A": 0.0, "C": 0.0, "G": 1.0, "T": 0.0},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    log_odds = build_log_odds(matrix, background)
    assert all(np.isfinite(val) for row in log_odds for val in row.values())

    motif = PWMMotif(motif_id="M1", matrix=matrix, background=background)
    rng = np.random.default_rng(0)
    sites = sample_pwm_sites(
        rng,
        motif,
        strategy="stochastic",
        n_sites=1,
        oversample_factor=3,
        score_threshold=None,
        score_percentile=50.0,
    )
    assert len(sites) == 1
    core = sites[0][: len(matrix)]
    score = score_sequence(core, matrix, background=background)
    assert np.isfinite(score)
