from __future__ import annotations

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import PWMMotif, sample_pwm_sites


def test_pwm_sampling_trim_window_selects_max_info() -> None:
    motif = PWMMotif(
        motif_id="trim",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
            {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    rng = np.random.default_rng(0)
    seqs = sample_pwm_sites(
        rng,
        motif,
        strategy="consensus",
        n_sites=1,
        oversample_factor=1,
        max_candidates=None,
        max_seconds=None,
        score_threshold=-1e6,
        score_percentile=None,
        length_policy="exact",
        length_range=None,
        trim_window_length=2,
        trim_window_strategy="max_info",
    )
    assert seqs == ["AA"]


def test_pwm_sampling_trim_length_rejects_too_long() -> None:
    motif = PWMMotif(
        motif_id="trim",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="exceeds motif width"):
        sample_pwm_sites(
            rng,
            motif,
            strategy="consensus",
            n_sites=1,
            oversample_factor=1,
            max_candidates=None,
            max_seconds=None,
            score_threshold=-1e6,
            score_percentile=None,
            length_policy="exact",
            length_range=None,
            trim_window_length=3,
            trim_window_strategy="max_info",
        )
