"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_elites_selection_config.py

Validates elite selection configuration defaults and constraints.

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
    SampleElitesSelectionConfig,
    SampleObjectiveConfig,
)


def _base_sample_config(*, elites_selection: SampleElitesSelectionConfig | None = None) -> SampleConfig:
    payload = {
        "budget": SampleBudgetConfig(draws=2, tune=1),
        "early_stop": SampleEarlyStopConfig(enabled=True, patience=10, min_delta=0.05),
        "init": InitConfig(kind="random", length=6),
        "objective": SampleObjectiveConfig(score_scale="normalized-llr"),
        "optimizer": OptimizerSelectionConfig(name="pt"),
        "optimizers": OptimizersConfig(),
        "auto_opt": AutoOptConfig(enabled=False),
    }
    if elites_selection is not None:
        payload["elites"] = {"selection": elites_selection}
    return SampleConfig(**payload)


def test_elites_selection_defaults() -> None:
    cfg = _base_sample_config()
    selection = cfg.elites.selection
    assert selection.policy == "mmr"
    assert selection.pool_size == 1000
    assert selection.alpha == pytest.approx(0.85)
    assert selection.relevance == "min_per_tf_norm"


def test_elites_selection_alpha_must_be_positive() -> None:
    with pytest.raises(ValueError, match="selection.alpha"):
        _base_sample_config(elites_selection=SampleElitesSelectionConfig(alpha=0.0))
