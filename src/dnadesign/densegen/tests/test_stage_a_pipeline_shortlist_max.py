"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_pipeline_shortlist_max.py

Stage-A shortlist max coercion regression tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.adapters.sources import stage_a_pipeline


def test_coerce_shortlist_max_allows_none_for_mmr() -> None:
    assert stage_a_pipeline._coerce_shortlist_max("mmr", None) is None


def test_coerce_shortlist_max_ignores_non_mmr_policy() -> None:
    assert stage_a_pipeline._coerce_shortlist_max("top_score", 10) is None


def test_coerce_shortlist_max_casts_values() -> None:
    assert stage_a_pipeline._coerce_shortlist_max("mmr", 12) == 12
