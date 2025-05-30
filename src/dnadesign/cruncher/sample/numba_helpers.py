"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/numba_helpers.py

Numba-accelerated PWM sliding-window scoring.

This is the core inner loop for scanning a sequence:
Given an integer-encoded DNA array and a log-odds matrix,
compute the maximum log-odds sum over all alignments.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import numpy as np
from numba import njit


@njit
def best_score_pwm(seq: np.ndarray, lom: np.ndarray) -> float:
    """
    Compute the best (max) log-odds score of `lom` over `seq`.

    Args:
      seq: 1D integer array (0=A,1=C,2=G,3=T)
      lom: 2D float array (motif_length × 4) log-odds

    Returns:
      best_score: float maximum sum of lom[j, seq[offset+j]]
    """
    L = seq.shape[0]
    m = lom.shape[0]
    best_score = -1e12
    for offset in range(L - m + 1):
        s = 0.0
        for j in range(m):
            s += lom[j, seq[offset + j]]
        if s > best_score:
            best_score = s
    return best_score
