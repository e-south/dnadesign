"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_run_set_strictness.py

Validates strict preflight checks in sample run setup.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.cruncher.app.sample.run_set import (
    ConfigError,
    RunError,
    _assert_sequence_buffers_aligned,
    _validate_objective_preflight,
)
from dnadesign.cruncher.config.schema_v3 import (
    SampleBudgetConfig,
    SampleConfig,
    SampleObjectiveConfig,
    SampleObjectiveSoftminConfig,
)


def _sample_cfg(
    *,
    score_scale: str = "normalized-llr",
    combine: str = "min",
    softmin_enabled: bool = True,
) -> SampleConfig:
    return SampleConfig(
        seed=7,
        sequence_length=12,
        budget=SampleBudgetConfig(tune=1, draws=2),
        objective=SampleObjectiveConfig(
            score_scale=score_scale,  # type: ignore[arg-type]
            combine=combine,  # type: ignore[arg-type]
            softmin=SampleObjectiveSoftminConfig(enabled=softmin_enabled, schedule="fixed", beta_end=2.0),
        ),
    )


def test_multi_tf_llr_fails_fast_with_actionable_message() -> None:
    sample_cfg = _sample_cfg(score_scale="llr", combine="min", softmin_enabled=False)
    with pytest.raises(
        ConfigError,
        match=(
            "cruncher.sample.objective.score_scale.*"
            "Multi-TF runs require comparable scales\\. Use normalized-llr, logp, z, or consensus-neglop-sum\\."
        ),
    ):
        _validate_objective_preflight(sample_cfg, n_tfs=2)


def test_single_tf_llr_is_allowed() -> None:
    sample_cfg = _sample_cfg(score_scale="llr", combine="min", softmin_enabled=False)
    _validate_objective_preflight(sample_cfg, n_tfs=1)


def test_sum_combine_with_softmin_enabled_is_rejected() -> None:
    sample_cfg = _sample_cfg(score_scale="normalized-llr", combine="sum", softmin_enabled=True)
    with pytest.raises(
        ConfigError,
        match="Disable softmin or switch combine to min/softmin-compatible mode\\.",
    ):
        _validate_objective_preflight(sample_cfg, n_tfs=2)


def test_consensus_sum_scale_with_softmin_enabled_is_rejected() -> None:
    sample_cfg = _sample_cfg(score_scale="consensus-neglop-sum", combine="min", softmin_enabled=True)
    with pytest.raises(
        ConfigError,
        match="Disable softmin or switch combine to min/softmin-compatible mode\\.",
    ):
        _validate_objective_preflight(sample_cfg, n_tfs=2)


class _BufferOptimizer:
    def __init__(self, *, meta: int, samples: int, scores: int) -> None:
        self.all_meta = [(0, idx) for idx in range(meta)]
        self.all_samples = [object() for _ in range(samples)]
        self.all_scores = [object() for _ in range(scores)]


def test_sequence_buffers_mismatch_fails_fast() -> None:
    optimizer = _BufferOptimizer(meta=3, samples=2, scores=3)
    with pytest.raises(RunError, match="inconsistent lengths"):
        _assert_sequence_buffers_aligned(optimizer)


def test_sequence_buffers_match_is_accepted() -> None:
    optimizer = _BufferOptimizer(meta=2, samples=2, scores=2)
    meta, samples, scores = _assert_sequence_buffers_aligned(optimizer)
    assert len(meta) == 2
    assert len(samples) == 2
    assert len(scores) == 2
