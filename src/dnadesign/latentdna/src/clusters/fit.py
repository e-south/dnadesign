"""
Cluster artifact builders for latentdna.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from ..contracts.errors import ContractViolationError
from ..io.json_io import write_json
from ..io.parquet_io import write_table
from ..views.scopes import resolve_view_scope
from ..workspaces.loader import WorkspaceContext


def _run_kmeans(
    matrix: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    if n_clusters < 1 or n_clusters > matrix.shape[0]:
        raise ContractViolationError(
            f"cluster count must be between 1 and {matrix.shape[0]} for the requested scope, got {n_clusters}"
        )
    if max_iter < 1:
        raise ContractViolationError(f"cluster max_iter must be at least 1, got {max_iter}")

    rng = np.random.default_rng(seed)
    initial_indices = rng.choice(matrix.shape[0], size=n_clusters, replace=False)
    centroids = np.asarray(matrix[initial_indices], dtype=np.float32).copy()
    labels = np.full(matrix.shape[0], fill_value=-1, dtype=np.int64)

    for iteration in range(1, max_iter + 1):
        squared_distances = np.sum((matrix[:, None, :] - centroids[None, :, :]) ** 2, axis=2, dtype=np.float32)
        next_labels = np.argmin(squared_distances, axis=1).astype(np.int64, copy=False)

        if np.array_equal(labels, next_labels):
            return labels, centroids, iteration - 1, True

        labels = next_labels
        next_centroids = centroids.copy()
        errors = squared_distances[np.arange(matrix.shape[0]), labels]
        for cluster_index in range(n_clusters):
            members = matrix[labels == cluster_index]
            if members.size == 0:
                replacement_index = int(np.argmax(errors))
                next_centroids[cluster_index] = matrix[replacement_index]
                continue
            next_centroids[cluster_index] = members.mean(axis=0, dtype=np.float32)

        if np.allclose(centroids, next_centroids):
            centroids = next_centroids
            return labels, centroids, iteration, True
        centroids = next_centroids

    return labels, centroids, max_iter, False


def fit_cluster_artifact(
    context: WorkspaceContext,
    *,
    cluster_id: str,
    view_id: str,
    n_clusters: int,
    seed: int,
    max_iter: int,
    sample_id: str | None,
    alignment_id: str | None,
) -> tuple[Path, int, str, str | None, int, bool, dict[int, int]]:
    matrix, rows, scope_kind, scope_id = resolve_view_scope(
        context,
        view_id=view_id,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    row_count = int(rows.num_rows)
    if row_count < 1:
        raise ContractViolationError("cluster fitting requires at least one row")

    labels, _, iterations, converged = _run_kmeans(
        np.ascontiguousarray(np.asarray(matrix, dtype=np.float32)),
        n_clusters=n_clusters,
        seed=seed,
        max_iter=max_iter,
    )
    counts = {int(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True), strict=True)}

    assignments = rows.append_column("cluster_label", pa.array(labels.tolist(), type=pa.int64()))
    artifact_dir = context.output_root / "clusters" / cluster_id
    write_table(assignments, artifact_dir / "assignments.parquet")
    write_json(
        artifact_dir / "summary.json",
        {
            "method": "kmeans",
            "view_id": view_id,
            "rows": row_count,
            "scope_kind": scope_kind,
            "scope_id": scope_id,
            "n_clusters": n_clusters,
            "seed": seed,
            "max_iter": max_iter,
            "iterations": iterations,
            "converged": converged,
            "cluster_sizes": counts,
        },
    )
    return artifact_dir, row_count, scope_kind, scope_id, iterations, converged, counts
