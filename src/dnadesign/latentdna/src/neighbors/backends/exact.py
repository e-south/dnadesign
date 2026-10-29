"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/neighbors/backends/exact.py

Exact neighbor backend for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from ...contracts.errors import ContractViolationError


def _euclidean_distances(matrix: np.ndarray) -> np.ndarray:
    squared_norms = np.sum(matrix * matrix, axis=1, keepdims=True, dtype=np.float32)
    distances = squared_norms + squared_norms.T - 2.0 * (matrix @ matrix.T)
    distances = np.maximum(distances, 0.0, out=distances)
    return np.sqrt(distances, out=distances)


def _cosine_distances(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.clip(norms, a_min=1e-12, a_max=None)
    distances = 1.0 - (normalized @ normalized.T)
    return np.clip(distances, a_min=0.0, a_max=2.0, out=distances)


def fit_neighbors_exact(matrix: np.ndarray, *, k: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric == "euclidean":
        distances = _euclidean_distances(matrix)
    elif metric == "cosine":
        distances = _cosine_distances(matrix)
    else:
        raise ContractViolationError(f"unsupported neighbor metric: {metric!r}")

    np.fill_diagonal(distances, np.inf)
    partition = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    selected_distances = np.take_along_axis(distances, partition, axis=1)
    order = np.argsort(selected_distances, axis=1)
    neighbor_indices = np.take_along_axis(partition, order, axis=1)
    neighbor_distances = np.take_along_axis(selected_distances, order, axis=1)
    return (
        np.ascontiguousarray(np.asarray(neighbor_indices, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(neighbor_distances, dtype=np.float32)),
    )
