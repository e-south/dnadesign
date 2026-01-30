"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_tier_targeting.py

Tier targeting helpers for Stage-A PWM sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.adapters.sources.stage_a_pipeline import _evaluate_tier_target


def test_tier_target_requirement_computation() -> None:
    required, met = _evaluate_tier_target(
        n_sites=200,
        target_tier_fraction=0.001,
        eligible_unique=150_000,
    )
    assert required == 200_000
    assert met is False
