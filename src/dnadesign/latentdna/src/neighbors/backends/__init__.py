"""
Neighbor backend registry for latentdna.
"""

from __future__ import annotations

import numpy as np

from ...contracts.errors import ContractViolationError
from .approximate import approximate_backend_available, fit_neighbors_approximate
from .exact import fit_neighbors_exact


def fit_neighbors_with_backend(
    matrix: np.ndarray,
    *,
    k: int,
    metric: str,
    backend: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    if backend == "auto":
        if approximate_backend_available():
            indices, distances = fit_neighbors_approximate(matrix, k=k, metric=metric, seed=seed)
            return indices, distances, "approximate", True
        indices, distances = fit_neighbors_exact(matrix, k=k, metric=metric)
        return indices, distances, "exact", False
    if backend == "approximate":
        indices, distances = fit_neighbors_approximate(matrix, k=k, metric=metric, seed=seed)
        return indices, distances, "approximate", True
    if backend == "exact":
        indices, distances = fit_neighbors_exact(matrix, k=k, metric=metric)
        return indices, distances, "exact", False
    raise ContractViolationError(f"unsupported neighbor backend: {backend!r}")
