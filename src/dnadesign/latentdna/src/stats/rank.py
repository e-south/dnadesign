"""Rank and ordinal association statistics used across LatentDNA."""

from __future__ import annotations

import numpy as np


def _finite_pair_arrays(
    left: list[float] | np.ndarray,
    right: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def rankdata_average(values: list[float] | np.ndarray) -> np.ndarray:
    """Return one-based average ranks with stable tie handling."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.asarray(array, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    sorted_values = array[order]
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    return ranks


def pearson_correlation(
    left: list[float] | np.ndarray,
    right: list[float] | np.ndarray,
    *,
    min_pairs: int = 2,
) -> float:
    """Return Pearson correlation, or NaN for degenerate inputs."""

    x, y = _finite_pair_arrays(left, right)
    if x.size < min_pairs:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(
    left: list[float] | np.ndarray,
    right: list[float] | np.ndarray,
    *,
    min_pairs: int = 2,
) -> float:
    """Return Spearman rank correlation, or NaN for degenerate inputs."""

    x, y = _finite_pair_arrays(left, right)
    if x.size < min_pairs:
        return float("nan")
    x_ranks = rankdata_average(x)
    y_ranks = rankdata_average(y)
    return pearson_correlation(x_ranks, y_ranks, min_pairs=min_pairs)


def kendall_tau_b(
    left: list[float] | np.ndarray,
    right: list[float] | np.ndarray,
    *,
    min_pairs: int = 2,
) -> float:
    """Return Kendall tau-b, including tie correction, or NaN when undefined."""

    x, y = _finite_pair_arrays(left, right)
    if x.size < min_pairs:
        return float("nan")
    concordant = 0
    discordant = 0
    tie_x = 0
    tie_y = 0
    for left_index in range(x.size):
        for right_index in range(left_index + 1, x.size):
            dx = np.sign(x[left_index] - x[right_index])
            dy = np.sign(y[left_index] - y[right_index])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
                continue
            if dy == 0:
                tie_y += 1
                continue
            if dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = np.sqrt(float((concordant + discordant + tie_x) * (concordant + discordant + tie_y)))
    if denominator < 1e-12:
        return float("nan")
    return float((concordant - discordant) / denominator)


def linear_r2(
    left: list[float] | np.ndarray,
    right: list[float] | np.ndarray,
    *,
    min_pairs: int = 2,
) -> float:
    """Return squared Pearson correlation, or NaN for degenerate inputs."""

    x, y = _finite_pair_arrays(left, right)
    pearson = pearson_correlation(x, y, min_pairs=min_pairs)
    if not np.isfinite(pearson):
        return float("nan")
    return pearson * pearson
