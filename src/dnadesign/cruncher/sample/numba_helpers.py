"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/numba_helpers.py

Numba-accelerated helper for PWM sliding-window scoring.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""
# cruncher/sample/numba_helpers.py

"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/numba_helpers.py

Numba-accelerated helper for PWM sliding-window scoring.

Why Numba?
----------
  Sliding a log-odds matrix over every possible subsequence is the
  hottest inner loop of our sampler.  We use @njit to compile this
  to native code, eliminating Python overhead and making each
  scoring call as fast as possible.

Core functionality:
  Given a sequence encoded as integers 0..3 and a log-odds matrix
  of shape (motif_length x 4), return the best (maximum) dot-product
  score over all valid alignments of the motif within the sequence.
--------------------------------------------------------------------------------
"""

import numpy as np
from numba import njit

@njit
def best_score_pwm(seq: np.ndarray, lom: np.ndarray) -> float:
    """
    Slide the log-odds matrix 'lom' over every window of 'seq' and
    return the maximum sum of log-odds:
    
      - seq: array of ints (0=A,1=C,2=G,3=T), length L
      - lom: array of floats, shape (m,4) for a PWM of length m

    We assume L >= m.  We loop over each offset, compute the inner
    sum for that window, and track the best score.
    """
    L = seq.shape[0]
    m = lom.shape[0]
    best = -1e12            # start lower than any realistic log-odds
    # for each possible alignment of motif within sequence:
    for offset in range(L - m + 1):
        s = 0.0
        # dot-product: sum log-odds for each position
        for j in range(m):
            s += lom[j, seq[offset + j]]
        # keep the highest-scoring alignment
        if s > best:
            best = s
    return best