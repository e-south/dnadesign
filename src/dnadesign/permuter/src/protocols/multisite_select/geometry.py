"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/multisite_select/geometry.py

Geometry utilities for Evo-2 logits embeddings:

  • row-wise L2 normalization
  • cosine similarity and angular distance
  • full pairwise angular distance matrix
  • cluster medoid index under angular distance

All routines assume finite numeric inputs; zero-norm vectors are rejected
assertively (no silent fallbacks).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

import numpy as np


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row of a 2D array.

    Args
    ----
    mat:
        Array-like [N, D]. Will be coerced to float64.

    Returns
    -------
    np.ndarray [N, D]
        Unit-norm vectors along axis 1.

    Raises
    ------
    ValueError
        If any row has zero L2 norm or contains non-finite values.
    """
    m = np.asarray(mat, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"l2_normalize_rows expects 2D input, got shape={m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError("l2_normalize_rows: matrix contains non-finite values")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("l2_normalize_rows: embedding contains zero-norm vectors")
    return m / norms


def _dot_clipped(u: np.ndarray, v: np.ndarray) -> float:
    """Dot product clipped to [-1, 1] for stable arccos."""
    return float(np.clip(float(np.dot(u, v)), -1.0, 1.0))


def angular_distance(u: np.ndarray, v: np.ndarray) -> float:
    """
    Angular distance in radians between two unit vectors.

    Assumes u and v are 1D and already L2-normalized; input is coerced to float64.
    """
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    if u.shape != v.shape:
        raise ValueError(f"angular_distance: mismatched shapes {u.shape} vs {v.shape}")
    c = _dot_clipped(u, v)
    return float(math.acos(c))


def pairwise_angular(U: np.ndarray) -> np.ndarray:
    """
    Full pairwise angular distance matrix in radians.

    Args
    ----
    U:
        [N, D] array of unit vectors (L2-normalized).

    Returns
    -------
    np.ndarray [N, N]
        A[i,j] = arccos(<U_i, U_j>) in radians.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError(f"pairwise_angular expects 2D input, got shape={U.shape}")
    G = np.clip(U @ U.T, -1.0, 1.0)
    return np.arccos(G).astype(float)


def medoid_index(U: np.ndarray) -> int:
    """
    Index of the medoid under angular distance.

    The medoid is the row whose average angular distance to all other rows is
    minimal. This is used for cluster-level diagnostics only.

    Args
    ----
    U:
        [N, D] array of unit vectors (L2-normalized).

    Returns
    -------
    int
        Row index (0-based) of the medoid.

    Raises
    ------
    ValueError
        If U is empty or not 2D.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[0] == 0:
        raise ValueError(
            f"medoid_index expects non-empty 2D input, got shape={U.shape}"
        )
    A = pairwise_angular(U)
    mean_per_row = A.mean(axis=1)
    return int(np.argmin(mean_per_row))


def min_angular_distance_to_set(
    x: np.ndarray,
    S: np.ndarray,
) -> float:
    """
    Minimum angular distance (in radians) between a candidate vector x and a set S.

    Args
    ----
    x:
        1D unit vector.
    S:
        [M, D] array of unit vectors; may be empty.

    Returns
    -------
    float
        Minimum angular distance in radians, or +inf if S is empty.
    """
    x = np.asarray(x, dtype=float).ravel()
    S = np.asarray(S, dtype=float)
    if S.size == 0:
        return float("inf")
    if S.ndim != 2:
        raise ValueError(f"min_angular_distance_to_set: S must be 2D, got {S.shape}")
    dots = np.clip(S @ x, -1.0, 1.0)  # [M]
    return float(np.arccos(float(np.max(dots))))
