"""
--------------------------------------------------------------------------------
<dnadesign project>
aligner/matrix.py

Matrix module for constructing full and condensed pairwise score matrices.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from typing import List

import numpy as np

from .align import global_alignment


def build_score_matrix(
    sequences: List[str], match: int = 2, mismatch: int = -1, gap_open: int = 10, gap_extend: int = 1
) -> np.ndarray:
    """
    Build a full symmetric score matrix for a list of sequences using global alignment.

    Parameters:
        sequences: List of nucleotide sequences (already validated and uppercase).
        match, mismatch, gap_open, gap_extend: Alignment parameters.

    Returns:
        An N x N NumPy array of pairwise alignment scores.
    """
    n = len(sequences)
    score_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score_matrix[i, j] = match * len(sequences[i])
            else:
                score = global_alignment(
                    sequences[i], sequences[j], match=match, mismatch=mismatch, gap_open=gap_open, gap_extend=gap_extend
                )
                score_matrix[i, j] = score
                score_matrix[j, i] = score
    return score_matrix


def matrix_to_condensed(score_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a full square matrix to a SciPy-style condensed (upper-triangular) vector.

    Parameters:
        score_matrix: An N x N symmetric NumPy array.

    Returns:
        A 1D NumPy array containing the upper-triangular elements (excluding the diagonal).
    """
    n = score_matrix.shape[0]
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(score_matrix[i, j])
    return np.array(condensed, dtype=score_matrix.dtype)
