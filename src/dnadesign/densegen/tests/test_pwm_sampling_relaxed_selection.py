"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_pwm_sampling_relaxed_selection.py

Stage-A PWM sampling relaxed selection rules.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import PWMMotif, sample_pwm_sites


def _motif() -> PWMMotif:
    return PWMMotif(
        motif_id="M1",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )


def test_densegen_accepts_score_percentile() -> None:
    rng = np.random.default_rng(0)
    sites = sample_pwm_sites(
        rng,
        _motif(),
        strategy="stochastic",
        n_sites=1,
        oversample_factor=2,
        max_candidates=None,
        max_seconds=None,
        score_threshold=None,
        score_percentile=90.0,
    )
    assert len(sites) == 1


def test_densegen_allows_shortfall_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    rng = np.random.default_rng(0)
    with caplog.at_level(logging.WARNING):
        sites = sample_pwm_sites(
            rng,
            _motif(),
            strategy="stochastic",
            n_sites=10,
            oversample_factor=1,
            max_candidates=None,
            max_seconds=None,
            score_threshold=1e6,
            score_percentile=None,
        )
    assert len(sites) < 10
    assert "shortfall" in caplog.text
