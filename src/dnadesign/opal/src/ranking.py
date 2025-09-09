"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/ranking.py

Implements competition ranking (e.g., 1,2,3,3,5) and helpers to compute
top-k tie thresholds. Assumes scores are sorted in descending order and
keeps tie behavior deterministic (sort by (-y_pred, id) upstream).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def competition_rank(scores_desc: Iterable[float]) -> np.ndarray:
    """
    Given scores in descending order, return competition ranks:
    Example: scores [9.0, 8.1, 7.2, 7.2, 6.0] -> ranks [1,2,3,3,5]
    """
    scores = np.asarray(list(scores_desc), dtype=float)
    ranks = np.empty_like(scores, dtype=np.int64)
    if scores.size == 0:
        return ranks
    rank = 1
    ranks[0] = 1
    for i in range(1, scores.size):
        if scores[i] == scores[i - 1]:
            ranks[i] = rank
        else:
            rank = i + 1
            ranks[i] = rank
    return ranks


def selection_threshold_for_top_k(scores_desc: np.ndarray, k: int) -> float:
    """
    Return the cutoff score such that all items >= cutoff are selected
    (ties included). Assumes scores_desc sorted DESC.
    """
    if scores_desc.size == 0 or k <= 0:
        return float("inf")
    idx = min(k - 1, scores_desc.size - 1)
    return float(scores_desc[idx])
