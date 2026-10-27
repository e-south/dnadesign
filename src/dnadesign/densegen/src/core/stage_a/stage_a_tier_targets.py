"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/stage_a/stage_a_tier_targets.py

Tier-target calculations for Stage-A PWM mining.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math


def required_unique_for_tier_target(*, n_sites: int, target_tier_fraction: float) -> int:
    if target_tier_fraction <= 0 or target_tier_fraction > 1:
        raise ValueError("target_tier_fraction must be in (0, 1].")
    return int(math.ceil(float(n_sites) / float(target_tier_fraction)))


def evaluate_tier_target(*, n_sites: int, target_tier_fraction: float, eligible_unique: int) -> tuple[int, bool]:
    required_unique = required_unique_for_tier_target(
        n_sites=n_sites,
        target_tier_fraction=target_tier_fraction,
    )
    return required_unique, int(eligible_unique) >= required_unique
