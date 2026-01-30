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

from dnadesign.densegen.src.adapters.sources.pwm_sampling import sample_pwm_sites
from dnadesign.densegen.src.adapters.sources.stage_a_types import PWMMotif
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.tests.pwm_sampling_fixtures import fixed_candidates_mining, selection_top_score

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


def _motif() -> PWMMotif:
    return PWMMotif(
        motif_id="M1",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )


def test_stage_a_rejects_untyped_selection() -> None:
    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="pwm.sampling.selection"):
        sample_pwm_sites(
            rng,
            _motif(),
            strategy="stochastic",
            n_sites=1,
            mining=fixed_candidates_mining(batch_size=5, candidates=5),
            selection={"policy": "top_score"},
        )


def test_densegen_allows_shortfall_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")
    rng = np.random.default_rng(0)
    with caplog.at_level(logging.WARNING):
        sites = sample_pwm_sites(
            rng,
            _motif(),
            strategy="stochastic",
            n_sites=10,
            mining=fixed_candidates_mining(batch_size=2, candidates=2, log_every_batches=1),
            selection=selection_top_score(),
        )
    assert len(sites) < 10
    assert "shortfall" in caplog.text
