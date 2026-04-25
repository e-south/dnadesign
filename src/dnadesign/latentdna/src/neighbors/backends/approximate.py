"""
Approximate neighbor backend for latentdna.
"""

from __future__ import annotations

import numpy as np

from ...contracts.errors import BackendUnavailableError, ContractViolationError


def approximate_backend_available() -> bool:
    try:
        from pynndescent import NNDescent  # noqa: F401
    except Exception:
        return False
    return True


def _strip_self_neighbors(indices: np.ndarray, distances: np.ndarray, *, k: int) -> tuple[np.ndarray, np.ndarray]:
    neighbor_indices: list[list[int]] = []
    neighbor_distances: list[list[float]] = []
    for row_index, (row_indices, row_distances) in enumerate(zip(indices, distances, strict=True)):
        selected_indices: list[int] = []
        selected_distances: list[float] = []
        seen: set[int] = set()
        for candidate, distance in zip(row_indices, row_distances, strict=True):
            candidate_index = int(candidate)
            if candidate_index == row_index or candidate_index in seen:
                continue
            seen.add(candidate_index)
            selected_indices.append(candidate_index)
            selected_distances.append(float(distance))
            if len(selected_indices) == k:
                break
        if len(selected_indices) != k:
            raise ContractViolationError("approximate neighbor backend did not return enough non-self neighbors")
        neighbor_indices.append(selected_indices)
        neighbor_distances.append(selected_distances)
    return (
        np.ascontiguousarray(np.asarray(neighbor_indices, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(neighbor_distances, dtype=np.float32)),
    )


def fit_neighbors_approximate(matrix: np.ndarray, *, k: int, metric: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from pynndescent import NNDescent
    except Exception as exc:  # pragma: no cover - dependency controlled by env
        raise BackendUnavailableError(f"approximate neighbor backend is unavailable: {exc}") from exc

    if metric not in {"euclidean", "cosine"}:
        raise ContractViolationError(f"unsupported neighbor metric: {metric!r}")

    matrix32 = np.asarray(matrix, dtype=np.float32)
    if not matrix32.flags.c_contiguous or not matrix32.flags.writeable:
        matrix32 = np.array(matrix32, dtype=np.float32, copy=True, order="C")
    n_rows = matrix32.shape[0]
    n_neighbors = min(n_rows, max(k + 1, 5))
    index = NNDescent(matrix32, metric=metric, n_neighbors=n_neighbors, random_state=seed)
    indices, distances = index.neighbor_graph
    return _strip_self_neighbors(indices, distances, k=k)
