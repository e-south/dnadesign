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

TIER_FRACTIONS: tuple[float, float, float] = (0.01, 0.09, 0.90)


def score_tier_counts(total: int) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    n0 = max(1, int(math.ceil(TIER_FRACTIONS[0] * total)))
    n1 = min(total - n0, int(math.ceil(TIER_FRACTIONS[1] * total)))
    n2 = total - n0 - n1
    return n0, n1, n2
