"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_sample_early_stop_validation.py

Validates early-stop configuration invariants.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.cruncher.config.schema_v2 import (
    AutoOptConfig,
    InitConfig,
    OptimizersConfig,
    OptimizerSelectionConfig,
    SampleBudgetConfig,
    SampleConfig,
    SampleEarlyStopConfig,
    SampleObjectiveConfig,
)


def _build_sample_config(*, min_delta: float) -> SampleConfig:
    return SampleConfig(
        budget=SampleBudgetConfig(draws=2, tune=1, restarts=1),
        early_stop=SampleEarlyStopConfig(enabled=True, patience=10, min_delta=min_delta),
        init=InitConfig(kind="random", length=6),
        objective=SampleObjectiveConfig(score_scale="normalized-llr"),
        optimizer=OptimizerSelectionConfig(name="pt"),
        optimizers=OptimizersConfig(),
        auto_opt=AutoOptConfig(enabled=False),
    )


def test_early_stop_min_delta_rejects_large_normalized_llr() -> None:
    with pytest.raises(ValueError, match="early_stop.min_delta"):
        _build_sample_config(min_delta=0.5)


def test_early_stop_min_delta_accepts_small_normalized_llr() -> None:
    cfg = _build_sample_config(min_delta=0.05)
    assert cfg.early_stop.min_delta == 0.05


def test_early_stop_requires_min_unique_when_enabled() -> None:
    with pytest.raises(ValueError, match="min_unique"):
        SampleEarlyStopConfig(
            enabled=True,
            patience=10,
            min_delta=0.01,
            require_min_unique=True,
            min_unique=0,
            success_min_per_tf_norm=0.8,
        )
