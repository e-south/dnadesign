"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/parse/pvalue.py

Exact p-value lookup for PWM log-odds scores.

This module answers:
    "If I generate random DNA of this length (zero-order uniform),
     what is the probability of observing a motif window scoring at least s?"

We implement FIMO's dynamic-programming approach at 0.001-bit resolution,
building once per PWM a full null distribution and corresponding tail probabilities.

Inspired by Grant et al. 2011 (DOI: 10.1093/bioinformatics/btr064).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import numpy as np
from numba import njit


@njit
def _dp_convolve(lom_int: np.ndarray, bg: np.ndarray, offset: int, length: int) -> np.ndarray:
    """
    Build the null distribution by dynamic‐programming convolution, fully in Numba.
    """
    w = lom_int.shape[0]
    # default zeros() dtype is float64 under Numba
    probs = np.zeros(length)
    probs[0] = 1.0

    for i in range(w):
        col = lom_int[i]
        new = np.zeros(length)
        # each base in this column
        for b in range(col.shape[0]):
            shift = col[b] + offset
            # accumulate old probs into new shifted by 'shift'
            for j in range(length - shift):
                new[j + shift] += probs[j] * bg[b]
        probs = new
    return probs


def logodds_to_p_lookup(lom: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute exact tail p-values for a log-odds matrix under a zero-order background,
    with the DP convolution accelerated by Numba.
    """
    # 1) Scale log-odds to integer resolution
    scale = 1000.0 / np.log(2.0)
    lom_int = np.round(lom * scale).astype(np.int32)

    # 2) Determine total score bounds
    min_per_col = lom_int.min(axis=1)
    max_per_col = lom_int.max(axis=1)
    total_min = int(min_per_col.sum())
    total_max = int(max_per_col.sum())

    offset = -int(min_per_col.min())
    length = total_max - total_min + 1

    # 3) Build null distribution in one Numba call
    probs = _dp_convolve(lom_int, bg, offset, length)

    # 4) Compute tail probabilities P(X >= k)
    tail_p = probs[::-1].cumsum()[::-1]

    # 5) Map integer indices back to float scores
    scores = np.arange(total_min, total_max + 1) / scale

    return scores, tail_p
