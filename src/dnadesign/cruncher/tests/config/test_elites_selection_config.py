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
from pydantic import ValidationError

from dnadesign.cruncher.config.schema_v3 import (
    SampleBudgetConfig,
    SampleConfig,
    SampleElitesConfig,
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
    assert elites.select.pool_size == "auto"
    assert elites.select.diversity == pytest.approx(0.0)


def test_elites_pool_size_accepts_all_literal() -> None:
    cfg = _base_sample_config(elites=SampleElitesConfig(select={"pool_size": "all"}))
    assert cfg.elites.select.pool_size == "all"


def test_elites_pool_size_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="elites.select.pool_size"):
        _base_sample_config(elites=SampleElitesConfig(select={"pool_size": 0}))
    with pytest.raises(ValidationError):
        _base_sample_config(elites=SampleElitesConfig(select={"pool_size": "huge"}))


def test_elites_diversity_must_be_in_closed_unit_interval() -> None:
    with pytest.raises(ValueError, match="elites.select.diversity"):
        _base_sample_config(elites=SampleElitesConfig(select={"diversity": -0.01}))
    with pytest.raises(ValueError, match="elites.select.diversity"):
        _base_sample_config(elites=SampleElitesConfig(select={"diversity": 1.01}))


@pytest.mark.parametrize(
    "key,value",
    [
        ("filter", {"min_per_tf_norm": 0.6}),
        ("select", {"alpha": 0.8}),
        ("select", {"relevance": "min_tf_score"}),
        ("select", {"distance_metric": "hybrid"}),
        ("select", {"constraint_policy": "relax"}),
        ("select", {"min_hamming_bp": 3}),
        ("select", {"min_core_hamming_bp": 2}),
        ("select", {"pool_strategy": "stratified"}),
    ],
)
def test_removed_elite_keys_are_rejected(key: str, value: object) -> None:
    payload = {"k": 10}
    payload[key] = value
    with pytest.raises(ValidationError):
        _base_sample_config(elites=SampleElitesConfig(**payload))
