"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/pvalue.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from functools import lru_cache

import numpy as np
from numba import njit

_LOGODDS_SCALE = 1000.0 / np.log(2.0)
_LOGODDS_CACHE_MAXSIZE = 256


@njit
def _dp_convolve(lom_int: np.ndarray, bg: np.ndarray, offset: int, length: int) -> np.ndarray:
    """
    Build the null distribution by dynamic-programming convolution, fully in Numba.
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


def _bg_key(bg: np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(bg, dtype=float)
    if arr.shape != (4,):
        raise ValueError("background must be a length-4 probability vector")
    return tuple(float(x) for x in arr)


@lru_cache(maxsize=_LOGODDS_CACHE_MAXSIZE)
def _logodds_lookup_cached(
    lom_bytes: bytes,
    shape: tuple[int, int],
    bg_tuple: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    lom_int = np.frombuffer(lom_bytes, dtype=np.int32).reshape(shape)
    min_per_col = lom_int.min(axis=1)
    max_per_col = lom_int.max(axis=1)
    total_min = int(min_per_col.sum())
    total_max = int(max_per_col.sum())

    offset = -int(min_per_col.min())
    length = total_max - total_min + 1

    probs = _dp_convolve(lom_int, np.asarray(bg_tuple, dtype=float), offset, length)
    tail_p = probs[::-1].cumsum()[::-1]
    scores = np.arange(total_min, total_max + 1) / _LOGODDS_SCALE
    return scores, tail_p


def logodds_cache_info():  # pragma: no cover - thin wrapper
    return _logodds_lookup_cached.cache_info()


def clear_logodds_cache() -> None:  # pragma: no cover - thin wrapper
    _logodds_lookup_cached.cache_clear()


def logodds_to_p_lookup(lom: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute exact tail p-values for a log-odds matrix under a zero-order background,
    with the DP convolution accelerated by Numba.
    """
    # 1) Scale log-odds to integer resolution
    lom_int = np.round(lom * _LOGODDS_SCALE).astype(np.int32)
    return _logodds_lookup_cached(lom_int.tobytes(), lom_int.shape, _bg_key(bg))
