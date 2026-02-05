"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_length_ladder.py

Auto-opt pilot budget helper tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.cruncher.app.sample.auto_opt import _budget_to_tune_draws, _pilot_budget_levels
from dnadesign.cruncher.config.schema_v2 import AutoOptConfig, InitConfig, SampleBudgetConfig, SampleConfig


def test_pilot_budget_levels_clamp_to_base_total() -> None:
    base_cfg = SampleConfig(
        budget=SampleBudgetConfig(tune=10, draws=10, restarts=1),
        init=InitConfig(kind="random", length=12),
    )
    auto_cfg = AutoOptConfig(budget_levels=[4, 10, 25])
    budgets = _pilot_budget_levels(base_cfg, auto_cfg)
    assert budgets == [4, 10, 20]


def test_budget_to_tune_draws_reserves_min_draws() -> None:
    base_cfg = SampleConfig(
        budget=SampleBudgetConfig(tune=1, draws=1, restarts=1),
        init=InitConfig(kind="random", length=12),
    )
    tune, draws = _budget_to_tune_draws(base_cfg, total_sweeps=6)
    assert draws == 4
    assert tune == 2


def test_pilot_budget_levels_rejects_tiny_base_budget() -> None:
    base_cfg = SampleConfig(
        budget=SampleBudgetConfig(tune=1, draws=1, restarts=1),
        init=InitConfig(kind="random", length=12),
    )
    auto_cfg = AutoOptConfig(budget_levels=[4])
    with pytest.raises(ValueError, match=r"tune\+draws"):
        _pilot_budget_levels(base_cfg, auto_cfg)
