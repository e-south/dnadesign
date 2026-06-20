"""Pairwise alignment score matrix helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dnadesign.aligner.pairwise.scoring import global_alignment


def build_score_matrix(
    sequences: Sequence[str],
    match: int = 2,
    mismatch: int = -1,
    gap_open: int = 10,
    gap_extend: int = 1,
) -> np.ndarray:
    """Build a full symmetric score matrix for a list of sequences."""

    n = len(sequences)
    score_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score_matrix[i, j] = match * len(sequences[i])
                continue
            score = global_alignment(
                sequences[i],
                sequences[j],
                match=match,
                mismatch=mismatch,
                gap_open=gap_open,
                gap_extend=gap_extend,
            )
            score_matrix[i, j] = score
            score_matrix[j, i] = score
    return score_matrix


def matrix_to_condensed(score_matrix: np.ndarray) -> np.ndarray:
    """Convert a full square matrix to a SciPy-style condensed vector."""

    n = score_matrix.shape[0]
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(score_matrix[i, j])
    return np.array(condensed, dtype=score_matrix.dtype)
