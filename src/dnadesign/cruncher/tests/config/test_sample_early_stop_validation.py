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
from pydantic import ValidationError

from dnadesign.cruncher.config.schema_v3 import (
    SampleBudgetConfig,
    SampleConfig,
    SampleObjectiveConfig,
)


def _build_sample_payload() -> dict[str, object]:
    return {
        "seed": 7,
        "sequence_length": 6,
        "budget": {"tune": 1, "draws": 2},
        "objective": {"score_scale": "normalized-llr"},
    }


def test_legacy_early_stop_config_is_rejected() -> None:
    payload = _build_sample_payload()
    payload["early_stop"] = {"enabled": True, "patience": 10, "min_delta": 0.05}
    with pytest.raises(ValidationError) as exc:
        SampleConfig.model_validate(payload)
    assert any(err.get("type") == "extra_forbidden" for err in exc.value.errors())


def test_budget_draws_must_be_positive() -> None:
    with pytest.raises(ValueError, match="sample.budget.draws must be >= 1"):
        SampleBudgetConfig(tune=1, draws=0)


def test_objective_config_accepts_normalized_llr_defaults() -> None:
    cfg = SampleConfig(
        seed=7,
        sequence_length=6,
        budget=SampleBudgetConfig(tune=1, draws=2),
        objective=SampleObjectiveConfig(score_scale="normalized-llr"),
    )
    assert cfg.objective.score_scale == "normalized-llr"


def test_optimizer_early_stop_config_is_accepted() -> None:
    cfg = SampleConfig.model_validate(
        {
            **_build_sample_payload(),
            "optimizer": {
                "kind": "gibbs_anneal",
                "chains": 2,
                "cooling": {"kind": "fixed", "beta": 1.0},
                "early_stop": {
                    "enabled": True,
                    "patience": 10,
                    "min_delta": 0.05,
                    "require_min_unique": True,
                    "min_unique": 3,
                    "success_min_per_tf_norm": 0.4,
                },
            },
        }
    )
    assert cfg.optimizer.early_stop.enabled is True
    assert cfg.optimizer.early_stop.patience == 10
    assert cfg.optimizer.early_stop.min_delta == pytest.approx(0.05)
    assert cfg.optimizer.early_stop.require_min_unique is True
    assert cfg.optimizer.early_stop.min_unique == 3
    assert cfg.optimizer.early_stop.success_min_per_tf_norm == pytest.approx(0.4)


def test_optimizer_early_stop_enabled_requires_positive_patience() -> None:
    payload = _build_sample_payload()
    payload["optimizer"] = {
        "kind": "gibbs_anneal",
        "chains": 1,
        "cooling": {"kind": "fixed", "beta": 1.0},
        "early_stop": {"enabled": True, "patience": 0},
    }
    with pytest.raises(ValidationError, match="sample.optimizer.early_stop.patience must be >= 1 when enabled=true"):
        SampleConfig.model_validate(payload)
