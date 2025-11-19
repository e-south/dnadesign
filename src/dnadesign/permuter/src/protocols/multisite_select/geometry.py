"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/multisite_select/geometry.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import List

import numpy as np


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=float)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("embedding contains zero-norm vectors")
    return m / norms


def angular_distance(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(math.acos(c))


def angular_dists_to_set(x: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    x: [D], S: [M,D] unit vectors → return min angle to S for each row of x
    """
    if S.size == 0:
        return np.full((1,), np.inf, dtype=float)
    dots = S @ x  # [M]
    dots = np.clip(dots, -1.0, 1.0)
    return np.arccos(np.max(dots))  # min angle → arccos(max dot)


def medoid_index(U: np.ndarray) -> int:
    """
    Return argmin of average pairwise angular distance inside U (unit vectors).
    Uses small-batch blocks to stay memory-friendly.
    """
    # dense Gram then acos, averaged per row
    G = np.clip(U @ U.T, -1.0, 1.0)  # [n,n]
    A = np.arccos(G).astype(float)  # [n,n]
    mean_per_row = A.mean(axis=1)
    i = int(np.argmin(mean_per_row))
    return i


def pairwise_angular(U: np.ndarray) -> np.ndarray:
    """A[i,j] = arccos(<u_i,u_j>) in radians; rows of U must be L2-normalized."""
    G = np.clip(U @ U.T, -1.0, 1.0)
    return np.arccos(G).astype(float)


def farthest_first_on_medoids(
    medoids: np.ndarray,  # [C,D] (unit vectors)
    *,
    M: int,
    seed_idx: int,
) -> List[int]:
    """
    k-center greedy on medoids. Returns list of chosen medoid indices.
    """
    C = medoids.shape[0]
    if C == 0:
        return []
    M = int(min(max(1, M), C))
    chosen: List[int] = [int(seed_idx)]
    # maintain best distance to chosen set
    dots = medoids @ medoids[seed_idx]  # [C]
    dots = np.clip(dots, -1.0, 1.0)
    best = np.arccos(dots)
    best[seed_idx] = 0.0
    while len(chosen) < M:
        # pick argmax of current min distances
        cand = int(np.argmax(best))
        if cand in chosen:
            # find next distinct
            order = np.argsort(best)[::-1]
            cand = int(next(i for i in order if i not in chosen))
        chosen.append(cand)
        # update min distances
        dots = medoids @ medoids[cand]
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        best = np.minimum(best, ang)
    return chosen
