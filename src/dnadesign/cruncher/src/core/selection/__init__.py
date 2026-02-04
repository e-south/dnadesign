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
    compute_sequence_distance,
    select_mmr_elites,
)

__all__ = [
    "MmrCandidate",
    "MmrSelectionResult",
    "compute_core_distance",
    "compute_position_weights",
    "compute_sequence_distance",
    "select_mmr_elites",
]
