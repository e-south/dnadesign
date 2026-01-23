"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/pvalue_bins.py

Canonical p-value bin edges for Stage-A FIMO-based PWM sampling.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

CANONICAL_PVALUE_BINS: tuple[float, ...] = (
    1e-10,
    1e-8,
    1e-6,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
)


def resolve_pvalue_bins(pvalue_bins: Sequence[float] | None) -> list[float]:
    if pvalue_bins is None:
        return list(CANONICAL_PVALUE_BINS)
    return [float(v) for v in pvalue_bins]
