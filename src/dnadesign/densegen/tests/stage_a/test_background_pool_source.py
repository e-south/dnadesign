"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_a/test_background_pool_source.py

Tests for background_pool Stage-A source validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources.background_pool import (
    BackgroundPoolDataSource,
    _run_fimo_exclusion,
)
from dnadesign.densegen.src.config import (
    BackgroundPoolFiltersConfig,
    BackgroundPoolFimoExcludeConfig,
    BackgroundPoolLengthConfig,
    BackgroundPoolMiningBudgetConfig,
    BackgroundPoolMiningConfig,
    BackgroundPoolSamplingConfig,
)
from dnadesign.densegen.src.core.stage_a.stage_a_types import PWMMotif
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable

pytestmark = pytest.mark.fimo

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


def test_background_pool_requires_pwm_inputs(tmp_path: Path) -> None:
    sampling = BackgroundPoolSamplingConfig(
        n_sites=1,
        mining=BackgroundPoolMiningConfig(
            batch_size=1,
            budget=BackgroundPoolMiningBudgetConfig(mode="fixed_candidates", candidates=2),
        ),
        length=BackgroundPoolLengthConfig(policy="range", range=(4, 4)),
        filters=BackgroundPoolFiltersConfig(
            fimo_exclude=BackgroundPoolFimoExcludeConfig(
                pwms_input=["tf_pwms"],
                allow_zero_hit_only=True,
            )
        ),
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("densegen: {}\n")
    src = BackgroundPoolDataSource(
        cfg_path=cfg_path,
        sampling=sampling,
        input_name="neutral_bg",
        pwm_inputs=[],
    )
    with pytest.raises(ValueError, match="background_pool.*pwms_input"):
        src.load_data(rng=np.random.default_rng(0))


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_background_pool_fimo_zero_hit_rejects_any_hit() -> None:
    motif = PWMMotif(
        motif_id="test_motif",
        matrix=[
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    accepted, _scores = _run_fimo_exclusion(
        motifs=[motif],
        sequences=["AAAAAAAA", "TTTTTTTT", "CCCCCCCC"],
        allow_zero_hit_only=True,
        max_score_norm=None,
    )
    assert accepted == ["CCCCCCCC"]


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_background_pool_fimo_zero_hit_requires_no_hits_across_all_input_pwms() -> None:
    motif_a = PWMMotif(
        motif_id="motif_A",
        matrix=[
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
            {"A": 0.99, "C": 0.0033, "G": 0.0033, "T": 0.0034},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    motif_t = PWMMotif(
        motif_id="motif_T",
        matrix=[
            {"A": 0.0034, "C": 0.0033, "G": 0.0033, "T": 0.99},
            {"A": 0.0034, "C": 0.0033, "G": 0.0033, "T": 0.99},
            {"A": 0.0034, "C": 0.0033, "G": 0.0033, "T": 0.99},
            {"A": 0.0034, "C": 0.0033, "G": 0.0033, "T": 0.99},
            {"A": 0.0034, "C": 0.0033, "G": 0.0033, "T": 0.99},
            {"A": 0.0034, "C": 0.0033, "G": 0.0033, "T": 0.99},
            {"A": 0.0034, "C": 0.0033, "G": 0.0033, "T": 0.99},
            {"A": 0.0034, "C": 0.0033, "G": 0.0033, "T": 0.99},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    accepted, scores = _run_fimo_exclusion(
        motifs=[motif_a, motif_t],
        sequences=["AAAAAAAA", "TTTTTTTT", "CCCCCCCC"],
        allow_zero_hit_only=True,
        max_score_norm=None,
    )
    assert accepted == ["CCCCCCCC"]
    assert scores == {"CCCCCCCC": None}
