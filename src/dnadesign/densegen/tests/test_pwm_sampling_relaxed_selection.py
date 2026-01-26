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


def test_stage_a_requires_fimo_backend() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="scoring_backend"):
        sample_pwm_sites(
            rng,
            _motif(),
            strategy="stochastic",
            n_sites=1,
            oversample_factor=2,
            scoring_backend="densegen",
        )


def test_densegen_allows_shortfall_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    rng = np.random.default_rng(0)
    with caplog.at_level(logging.WARNING):
        sites = sample_pwm_sites(
            rng,
            _motif(),
            strategy="stochastic",
            n_sites=10,
            oversample_factor=1,
            scoring_backend="fimo",
            mining={"batch_size": 2, "max_seconds": 0, "log_every_batches": 1},
        )
    assert len(sites) < 10
    assert "shortfall" in caplog.text
