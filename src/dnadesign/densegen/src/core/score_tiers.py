"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/score_tiers.py

Tier sizing helpers for Stage-A score-ranked sampling.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Sequence

TIER_FRACTIONS: tuple[float, float, float] = (0.001, 0.01, 0.09)


def normalize_tier_fractions(
    fractions: Sequence[float] | None,
) -> tuple[float, float, float]:
    if fractions is None:
        return TIER_FRACTIONS
    if len(fractions) != 3:
        raise ValueError("Tier fractions must contain exactly three values.")
    values = [float(val) for val in fractions]
    if any(not math.isfinite(val) for val in values):
        raise ValueError("Tier fractions must be finite.")
    if any(val <= 0.0 or val > 1.0 for val in values):
        raise ValueError("Tier fractions must be in (0, 1].")
    if values != sorted(values):
        raise ValueError("Tier fractions must be non-decreasing.")
    if sum(values) > 1.0:
        raise ValueError("Tier fractions must sum to <= 1.0.")
    return values[0], values[1], values[2]


def resolve_tier_fractions(
    ladder: Sequence[float] | None,
    *,
    default: Sequence[float] | None = None,
) -> tuple[float, float, float]:
    if ladder is not None:
        values = [float(val) for val in ladder if float(val) < 1.0]
        if len(values) < 3:
            raise ValueError("Tier ladder must include at least three fractions below 1.0.")
        return normalize_tier_fractions(values[:3])
    if default is None:
        return normalize_tier_fractions(TIER_FRACTIONS)
    return normalize_tier_fractions(default)


def score_tier_counts(total: int, *, fractions: Sequence[float] | None = None) -> tuple[int, int, int, int]:
    if total <= 0:
        return 0, 0, 0, 0
    frac0, frac1, frac2 = normalize_tier_fractions(fractions)
    n0 = max(1, int(math.ceil(frac0 * total)))
    n1 = min(total - n0, int(math.ceil(frac1 * total)))
    n2 = min(total - n0 - n1, int(math.ceil(frac2 * total)))
    n3 = total - n0 - n1 - n2
    return n0, n1, n2, n3
