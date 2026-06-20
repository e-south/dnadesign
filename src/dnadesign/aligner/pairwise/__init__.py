"""Public pairwise alignment scoring API."""

from dnadesign.aligner.pairwise.matrix import build_score_matrix, matrix_to_condensed
from dnadesign.aligner.pairwise.scoring import compute_alignment_scores, global_alignment, mean_pairwise, score_pairwise
from dnadesign.aligner.pairwise.validation import extract_sequence, validate_sequence

__all__ = [
    "build_score_matrix",
    "compute_alignment_scores",
    "extract_sequence",
    "global_alignment",
    "matrix_to_condensed",
    "mean_pairwise",
    "score_pairwise",
    "validate_sequence",
]
