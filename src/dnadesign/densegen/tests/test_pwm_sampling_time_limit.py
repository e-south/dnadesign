from __future__ import annotations

import logging

import numpy as np

from dnadesign.densegen.src.adapters.sources import pwm_sampling
from dnadesign.densegen.src.adapters.sources.pwm_sampling import PWMMotif, sample_pwm_sites


def test_pwm_sampling_time_limit_warns(monkeypatch, caplog) -> None:
    motif = PWMMotif(
        motif_id="M1",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )

    clock = {"t": 0.0}

    def fake_monotonic() -> float:
        clock["t"] += 0.1
        return clock["t"]

    monkeypatch.setattr(pwm_sampling.time, "monotonic", fake_monotonic)
    rng = np.random.default_rng(0)
    with caplog.at_level(logging.WARNING):
        sample_pwm_sites(
            rng,
            motif,
            strategy="stochastic",
            n_sites=1,
            oversample_factor=100,
            max_candidates=None,
            max_seconds=0.05,
            score_threshold=-100.0,
            score_percentile=None,
            length_policy="exact",
            length_range=None,
        )
    assert "max_seconds" in caplog.text
