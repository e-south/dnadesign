"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/__init__.py

Public API for pairwise and multiple sequence alignment utilities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.aligner.pairwise import (
    build_score_matrix,
    compute_alignment_scores,
    global_alignment,
    matrix_to_condensed,
    mean_pairwise,
    score_pairwise,
)

__all__ = [
    "build_score_matrix",
    "compute_alignment_scores",
    "global_alignment",
    "matrix_to_condensed",
    "mean_pairwise",
    "score_pairwise",
]
