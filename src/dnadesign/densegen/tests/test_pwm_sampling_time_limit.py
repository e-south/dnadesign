from __future__ import annotations

import logging

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import PWMMotif, sample_pwm_sites
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.tests.pwm_sampling_fixtures import selection_top_score, tier_target_mining

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


def test_pwm_sampling_time_limit_warns(caplog) -> None:
    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")
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
            mining=tier_target_mining(
                batch_size=10,
                target_tier_fraction=0.001,
                max_seconds=0.000001,
                log_every_batches=1,
            ),
            selection=selection_top_score(),
            length_policy="exact",
            length_range=None,
        )
    assert "increase mining.budget.max_seconds" in caplog.text
