"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/sample/optimizer/helpers.py

Low-level, Numba-accelerated routines shared by both Gibbs and PT optimizers:
  - best_score_pwm: compute best log-odds of a PWM over a sequence
  - slide_window: reserved for sliding-window moves
  - swap_block: reserved for block-swap moves
  - _replace_block: replace a block in-place (used by Gibbs “B” move)

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
    win = seq[start : start + length].copy()
    if shift > 0:
        seq[start : start + length + shift] = np.concatenate([seq[start + shift : start + length + shift], win])
    else:
        s = start + shift
        seq[s : s + length] = win


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
