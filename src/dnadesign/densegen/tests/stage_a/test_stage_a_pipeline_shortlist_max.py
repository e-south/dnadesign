"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_a/test_stage_a_pipeline_shortlist_max.py

Stage-A selection pool cap coercion regression tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.core.stage_a import stage_a_pipeline


def test_coerce_pool_cap_allows_none_for_mmr() -> None:
    assert stage_a_pipeline._coerce_pool_max_candidates("mmr", None) is None


def test_coerce_pool_cap_ignores_non_mmr_policy() -> None:
    assert stage_a_pipeline._coerce_pool_max_candidates("top_score", 10) is None


def test_coerce_pool_cap_casts_values() -> None:
    assert stage_a_pipeline._coerce_pool_max_candidates("mmr", 12) == 12
