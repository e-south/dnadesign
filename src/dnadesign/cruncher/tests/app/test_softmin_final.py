"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_softmin_final.py

Validates deterministic final softmin beta resolution from schedule + executed sweeps.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.cruncher.app.sample.run_set import ConfigError, _resolve_final_softmin_beta
from dnadesign.cruncher.config.schema_v3 import (
    SampleBudgetConfig,
    SampleConfig,
    SampleObjectiveConfig,
    SampleObjectiveSoftminConfig,
)


class _DummyOptimizer:
    def __init__(self, sweep_indices: list[int]) -> None:
        self.all_meta = [(0, idx) for idx in sweep_indices]

    def final_softmin_beta(self) -> float:
        return 99.0

    def stats(self) -> dict[str, object]:
        return {"final_softmin_beta": 77.0}


def _sample_cfg(
    *,
    schedule: str,
    beta_start: float,
    beta_end: float,
    tune: int = 2,
    draws: int = 5,
) -> SampleConfig:
    return SampleConfig(
        seed=7,
        sequence_length=8,
        budget=SampleBudgetConfig(tune=tune, draws=draws),
        objective=SampleObjectiveConfig(
            softmin=SampleObjectiveSoftminConfig(
                enabled=True,
                schedule=schedule,  # type: ignore[arg-type]
                beta_start=beta_start,
                beta_end=beta_end,
            )
        ),
    )


def test_resolve_final_softmin_beta_uses_executed_sweep_count_for_linear_schedule() -> None:
    sample_cfg = _sample_cfg(schedule="linear", beta_start=1.0, beta_end=5.0, tune=2, draws=5)
    optimizer = _DummyOptimizer([0, 1, 2, 3])
    beta = _resolve_final_softmin_beta(optimizer, sample_cfg)
    assert beta == pytest.approx(3.0)


def test_resolve_final_softmin_beta_uses_executed_sweep_count_for_fixed_schedule() -> None:
    sample_cfg = _sample_cfg(schedule="fixed", beta_start=0.5, beta_end=2.5, tune=2, draws=5)
    optimizer = _DummyOptimizer([0, 1, 2])
    beta = _resolve_final_softmin_beta(optimizer, sample_cfg)
    assert beta == pytest.approx(2.5)


def test_resolve_final_softmin_beta_rejects_unknown_schedule_kind() -> None:
    sample_cfg = _sample_cfg(schedule="fixed", beta_start=1.0, beta_end=2.0)
    sample_cfg.objective.softmin.schedule = "bogus"  # type: ignore[assignment]

    with pytest.raises(ConfigError, match="objective.softmin.schedule"):
        _resolve_final_softmin_beta(_DummyOptimizer([0, 1]), sample_cfg)
