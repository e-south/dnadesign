"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/sequence.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """
    Return Hamming distance between two 1-D arrays, allowing unequal lengths.

    The distance is the mismatches over the shared prefix plus the length
    difference (i.e., extra positions count as mismatches).
    """
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    if arr_a.ndim != 1 or arr_b.ndim != 1:
        raise ValueError("hamming_distance requires 1-D arrays")
    if arr_a.size == arr_b.size:
        return int((arr_a != arr_b).sum())
    min_len = min(arr_a.size, arr_b.size)
    return int((arr_a[:min_len] != arr_b[:min_len]).sum()) + abs(int(arr_a.size) - int(arr_b.size))
