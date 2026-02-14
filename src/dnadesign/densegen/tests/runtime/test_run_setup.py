"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_run_setup.py

Tests for DenseGen run setup helpers that seed per-plan run statistics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

from dnadesign.densegen.src.core.pipeline.run_setup import init_plan_stats


def test_init_plan_stats_seeds_existing_library_build_counts() -> None:
    plan_items = [SimpleNamespace(name="controls")]
    plan_pools = {
        "controls": SimpleNamespace(pool_name="plan_pool__controls"),
    }
    existing_counts = {("plan_pool__controls", "controls"): 14}
    existing_library_build_counts = {("plan_pool__controls", "controls"): 7}

    plan_stats, _plan_order = init_plan_stats(
        plan_items=plan_items,
        plan_pools=plan_pools,
        existing_counts=existing_counts,
        existing_library_build_counts=existing_library_build_counts,
    )

    key = ("plan_pool__controls", "controls")
    assert plan_stats[key]["generated"] == 14
    assert plan_stats[key]["libraries_built"] == 7
