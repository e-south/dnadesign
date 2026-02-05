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
    InitConfig,
    SampleComputeConfig,
    SampleConfig,
    SampleEarlyStopConfig,
    SampleElitesConfig,
    SampleObjectiveConfig,
)


def _base_sample_config(*, elites: SampleElitesConfig | None = None) -> SampleConfig:
    payload = {
        "sequence_length": 6,
        "compute": SampleComputeConfig(total_sweeps=3, adapt_sweep_frac=0.34),
        "early_stop": SampleEarlyStopConfig(enabled=True, patience=10, min_delta=0.05),
        "init": InitConfig(kind="random"),
        "objective": SampleObjectiveConfig(score_scale="normalized-llr"),
    }
    if elites is not None:
        payload["elites"] = elites
    return SampleConfig(**payload)


def test_elites_defaults() -> None:
    cfg = _base_sample_config()
    elites = cfg.elites
    assert elites.k == 10
    assert elites.min_per_tf_norm is None
    assert elites.require_all_tfs_over_min_norm is True
    assert elites.mmr_alpha == pytest.approx(0.85)


def test_elites_mmr_alpha_must_be_positive() -> None:
    with pytest.raises(ValueError, match="mmr_alpha"):
        _base_sample_config(elites=SampleElitesConfig(mmr_alpha=0.0))
