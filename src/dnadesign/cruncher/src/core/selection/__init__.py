"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/selection/__init__.py

Selection utilities for elite extraction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.cruncher.core.selection.mmr import (
    MmrCandidate,
    MmrSelectionResult,
    compute_core_distance,
    compute_position_weights,
    select_mmr_elites,
    select_score_elites,
)

__all__ = [
    "MmrCandidate",
    "MmrSelectionResult",
    "compute_core_distance",
    "compute_position_weights",
    "select_score_elites",
    "select_mmr_elites",
]
