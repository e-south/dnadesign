"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/stats/__init__.py

Reusable statistical primitives for LatentDNA analyses.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .rank import kendall_tau_b, linear_r2, rankdata_average, spearman_correlation

__all__ = [
    "kendall_tau_b",
    "linear_r2",
    "rankdata_average",
    "spearman_correlation",
]
