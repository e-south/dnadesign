from __future__ import annotations

import logging

import numpy as np

from dnadesign.densegen.src.adapters.sources.pwm_sampling import PWMMotif, sample_pwm_sites


def test_pwm_sampling_time_limit_warns(caplog) -> None:
    motif = PWMMotif(
        motif_id="M1",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )

    rng = np.random.default_rng(0)
    with caplog.at_level(logging.WARNING):
        sample_pwm_sites(
            rng,
            motif,
            strategy="stochastic",
            n_sites=1,
            oversample_factor=1,
            scoring_backend="fimo",
            mining={"batch_size": 10, "max_seconds": 0, "log_every_batches": 1},
            length_policy="exact",
            length_range=None,
        )
    assert "increase mining.max_seconds" in caplog.text
