"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/helpers.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import numpy as np
from numba import njit


@njit
def best_score_pwm(seq: np.ndarray, lom: np.ndarray) -> float:
    """
    Compute the best (max) log-odds score of `lom` over `seq`.
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


@njit
def slide_window(seq: np.ndarray, start: int, length: int, shift: int):
    """
    Slide in-place a contiguous window of size `length` at position `start`
    by `shift` (positive→right, negative→left). Reserved for future use.
    """
    if shift > 0:
        win = seq[start : start + length].copy()
        tail = seq[start + length : start + length + shift].copy()
        seq[start : start + shift] = tail
        seq[start + shift : start + shift + length] = win
    elif shift < 0:
        k = -shift
        s = start - k
        win = seq[start : start + length].copy()
        head = seq[s : s + k].copy()
        seq[s : s + length] = win
        seq[s + length : s + length + k] = head


@njit
def swap_block(seq: np.ndarray, a: int, b: int, length: int):
    """
    Swap two equal-length blocks in-place: seq[a:a+length] <-> seq[b:b+length].
    Reserved for future use.
    """
    tmp = seq[a : a + length].copy()
    seq[a : a + length] = seq[b : b + length]
    seq[b : b + length] = tmp


@njit
def _replace_block(seq: np.ndarray, start: int, length: int, new_block: np.ndarray):
    """
    Replace `seq[start:start+length]` in-place with `new_block`.
    Used by GibbsOptimizer for the “B” move.
    """
    seq[start : start + length] = new_block
