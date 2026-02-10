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

from dnadesign.cruncher.config.schema_v3 import (
    SampleBudgetConfig,
    SampleConfig,
    SampleElitesConfig,
    SampleEliteSelectConfig,
    SampleObjectiveConfig,
)


def _base_sample_config(*, elites: SampleElitesConfig | None = None) -> SampleConfig:
    payload = {
        "seed": 7,
        "sequence_length": 6,
        "budget": SampleBudgetConfig(tune=1, draws=2),
        "objective": SampleObjectiveConfig(score_scale="normalized-llr"),
    }
    if elites is not None:
        payload["elites"] = elites
    return SampleConfig(**payload)


def test_elites_defaults() -> None:
    cfg = _base_sample_config()
    elites = cfg.elites
    assert elites.k == 10
    assert elites.filter.min_per_tf_norm is None
    assert elites.filter.require_all_tfs is True
    assert elites.select.alpha == pytest.approx(0.85)


def test_elites_mmr_alpha_must_be_positive() -> None:
    with pytest.raises(ValueError, match="elites.select.alpha"):
        _base_sample_config(elites=SampleElitesConfig(select=SampleEliteSelectConfig(alpha=0.0)))


def test_elites_filter_rejects_auto_threshold_value() -> None:
    with pytest.raises(ValueError, match="filter.min_per_tf_norm"):
        _base_sample_config(
            elites=SampleElitesConfig(
                filter={"min_per_tf_norm": "auto"},
            )
        )
