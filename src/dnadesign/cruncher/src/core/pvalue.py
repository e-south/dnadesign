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


@njit(cache=True)
def _dp_convolve_shifted(lom_shifted: np.ndarray, bg: np.ndarray, length: int) -> np.ndarray:
    """
    Build the null distribution by dynamic-programming convolution on per-column
    shifted integer scores. Each column in `lom_shifted` must have minimum 0.
    """
    w = lom_shifted.shape[0]
    probs = np.zeros(length, dtype=np.float64)
    probs[0] = 1.0
    max_idx = 0

    for i in range(w):
        col = lom_shifted[i]
        new = np.zeros(length, dtype=np.float64)
        next_max_idx = 0
        for b in range(col.shape[0]):
            shift = int(col[b])
            if shift >= length:
                continue
            candidate_max = max_idx + shift
            if candidate_max > next_max_idx:
                next_max_idx = candidate_max
            for j in range(max_idx + 1):
                new[j + shift] += probs[j] * bg[b]
        probs = new
        if next_max_idx >= length:
            max_idx = length - 1
        else:
            max_idx = next_max_idx
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

    length = total_max - total_min + 1
    lom_shifted = lom_int - min_per_col.reshape(-1, 1)

    probs = _dp_convolve_shifted(lom_shifted, np.asarray(bg_tuple, dtype=np.float64), length)
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
