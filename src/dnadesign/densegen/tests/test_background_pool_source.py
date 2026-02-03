"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_background_pool_source.py

Tests for background_pool Stage-A source validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources.background_pool import BackgroundPoolDataSource
from dnadesign.densegen.src.config import (
    BackgroundPoolFiltersConfig,
    BackgroundPoolFimoExcludeConfig,
    BackgroundPoolLengthConfig,
    BackgroundPoolMiningBudgetConfig,
    BackgroundPoolMiningConfig,
    BackgroundPoolSamplingConfig,
)


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
