"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_plan_execution_round_robin.py

Tests round-robin scheduling behavior for plan-level start-time reuse.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from dnadesign.densegen.src.core.pipeline.plan_context import PlanExecutionState
from dnadesign.densegen.src.core.pipeline.plan_execution import run_plan_schedule


@dataclass(frozen=True)
class _PlanItem:
    name: str
    quota: int


def test_round_robin_reuses_plan_started_at_per_plan() -> None:
    plan_items = [_PlanItem(name="plan_a", quota=2), _PlanItem(name="plan_b", quota=2)]
    plan_pools = {
        "plan_a": SimpleNamespace(pool_name="pool_a", include_inputs=["a"], pool=object()),
        "plan_b": SimpleNamespace(pool_name="pool_b", include_inputs=["b"], pool=object()),
    }
    plan_pool_sources = {
        "plan_a": SimpleNamespace(name="source_a"),
        "plan_b": SimpleNamespace(name="source_b"),
    }

    starts_by_plan: dict[str, list[float]] = {"plan_a": [], "plan_b": []}

    def _process_plan(
        source_cfg,
        plan_item,
        _plan_context,
        _execution_state,
        plan_started_at: float,
        *,
        one_subsample_only: bool,
        already_generated: int,
    ) -> tuple[int, dict]:
        assert one_subsample_only is True
        starts_by_plan[plan_item.name].append(plan_started_at)
        produced = 1 if already_generated < int(plan_item.quota) else 0
        return produced, {}

    result = run_plan_schedule(
        plan_items=plan_items,
        plan_pools=plan_pools,
        plan_pool_sources=plan_pool_sources,
        existing_counts={},
        round_robin=True,
        process_plan=_process_plan,
        plan_context=SimpleNamespace(),
        execution_state=PlanExecutionState(inputs_manifest={}),
        accumulate_stats=lambda _k, _s: None,
        plan_pool_input_meta=lambda _spec: {},
        existing_usage_by_plan=None,
    )

    assert result.total == 4
    assert result.per_plan[("pool_a", "plan_a")] == 2
    assert result.per_plan[("pool_b", "plan_b")] == 2
    assert len(starts_by_plan["plan_a"]) >= 2
    assert len(starts_by_plan["plan_b"]) >= 2
    assert len(set(starts_by_plan["plan_a"])) == 1
    assert len(set(starts_by_plan["plan_b"])) == 1
