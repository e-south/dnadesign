"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_tier_targeting.py

Tier targeting helpers for Stage-A PWM sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.adapters.sources.stage_a_pipeline import _evaluate_tier_target, _score_norm_by_tier


def test_tier_target_requirement_computation() -> None:
    required, met = _evaluate_tier_target(
        n_sites=200,
        target_tier_fraction=0.001,
        eligible_unique=150_000,
    )
    assert required == 200_000
    assert met is False


def test_score_norm_by_tier_summary() -> None:
    scores_by_seq = {"a": 10.0, "b": 8.0, "c": 6.0, "d": 4.0}
    tier_by_seq = {"a": 0, "b": 1, "c": 2, "d": 3}
    summary = _score_norm_by_tier(scores_by_seq, tier_by_seq, denominator=10.0)
    assert summary is not None
    assert summary["tier0"] == {"min": 1.0, "median": 1.0, "max": 1.0}
    assert summary["tier1"] == {"min": 0.8, "median": 0.8, "max": 0.8}
    assert summary["tier2"] == {"min": 0.6, "median": 0.6, "max": 0.6}
    assert summary["rest"] == {"min": 0.4, "median": 0.4, "max": 0.4}
