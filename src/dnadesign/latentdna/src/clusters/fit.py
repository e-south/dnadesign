"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/clusters/fit.py

Cluster artifact builders for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa

from ..contracts.errors import ContractViolationError
from ..contracts.workspace import SourceBackedViewConfig
from ..distances.score import _select_indices
from ..io.json_io import write_json
from ..io.parquet_io import write_table
from ..views.scopes import resolve_feature_scope
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


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    normalized = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    return np.asarray(normalized @ normalized.T, dtype=np.float32)


def _euclidean_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    deltas = matrix[:, None, :] - matrix[None, :, :]
    return np.asarray(np.linalg.norm(deltas, axis=2), dtype=np.float32)


def _build_neighbor_indices(matrix: np.ndarray, *, k: int, metric: str) -> np.ndarray:
    row_count = int(matrix.shape[0])
    if row_count < 2:
        raise ContractViolationError("Leiden clustering requires at least two rows")
    resolved_k = max(1, min(int(k), row_count - 1))
    if metric == "cosine":
        scores = _cosine_similarity_matrix(matrix)
        np.fill_diagonal(scores, -np.inf)
        order = np.argsort(scores, axis=1)[:, ::-1]
        return np.ascontiguousarray(order[:, :resolved_k], dtype=np.int64)
    if metric == "euclidean":
        distances = _euclidean_distance_matrix(matrix)
        np.fill_diagonal(distances, np.inf)
        order = np.argsort(distances, axis=1)
        return np.ascontiguousarray(order[:, :resolved_k], dtype=np.int64)
    raise ContractViolationError(f"unsupported cluster metric: {metric!r}")


def _edge_weight(matrix: np.ndarray, *, left: int, right: int, metric: str) -> float:
    if metric == "cosine":
        left_vector = matrix[left]
        right_vector = matrix[right]
        denominator = float(
            np.clip(
                np.linalg.norm(left_vector) * np.linalg.norm(right_vector),
                a_min=1e-12,
                a_max=None,
            )
        )
        return max(float(np.dot(left_vector, right_vector) / denominator), 0.0)
    if metric == "euclidean":
        distance = float(np.linalg.norm(matrix[left] - matrix[right]))
        return 1.0 / (1.0 + distance)
    raise ContractViolationError(f"unsupported cluster metric: {metric!r}")


def _run_leiden(
    matrix: np.ndarray,
    *,
    k: int,
    resolution: float,
    metric: str,
    seed: int,
    neighbor_indices: np.ndarray | None,
) -> np.ndarray:
    try:
        import igraph as ig
        import leidenalg
    except Exception as exc:  # pragma: no cover - dependency contract
        raise ContractViolationError("Leiden clustering requires igraph and leidenalg to be installed") from exc

    indices = neighbor_indices if neighbor_indices is not None else _build_neighbor_indices(matrix, k=k, metric=metric)
    if indices.ndim != 2 or indices.shape[0] != matrix.shape[0]:
        raise ContractViolationError("neighbor indices are misaligned with the requested cluster scope")

    edges: dict[tuple[int, int], float] = {}
    for left_index, neighbors in enumerate(indices):
        for neighbor_index in neighbors.tolist():
            if left_index == neighbor_index:
                continue
            edge = (left_index, neighbor_index) if left_index < neighbor_index else (neighbor_index, left_index)
            edges[edge] = _edge_weight(matrix, left=edge[0], right=edge[1], metric=metric)
    if not edges:
        raise ContractViolationError("Leiden clustering could not build any graph edges")

    edge_list = list(edges)
    graph = ig.Graph(n=int(matrix.shape[0]), edges=edge_list, directed=False)
    graph.es["weight"] = [float(edges[edge]) for edge in edge_list]
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=float(resolution),
        seed=int(seed),
    )
    return np.asarray(partition.membership, dtype=np.int64)


def _representative_row_id(rows: list[dict[str, object]], index: int) -> str:
    row = rows[index]
    for column in ("id", "subject_id", "context_id"):
        value = row.get(column)
        if value is not None:
            return str(value)
    first_column = next(iter(row))
    return str(row[first_column])


def _cluster_sizes_table(labels: np.ndarray) -> pa.Table:
    counts = Counter(int(label) for label in labels.tolist())
    return pa.Table.from_pylist(
        [{"cluster_label": label, "size": count} for label, count in sorted(counts.items(), key=lambda item: item[0])]
    )


def _cluster_enrichment_table(rows: list[dict[str, object]], labels: np.ndarray, *, column: str) -> pa.Table:
    background_counts = Counter(str(row[column]) for row in rows)
    ordered_values = sorted(background_counts, key=str)
    cluster_counts = Counter(int(label) for label in labels.tolist())
    output_rows: list[dict[str, object]] = []
    for cluster_label in sorted(cluster_counts):
        cluster_rows = [row for row, label in zip(rows, labels.tolist(), strict=True) if int(label) == cluster_label]
        cluster_value_counts = Counter(str(row[column]) for row in cluster_rows)
        cluster_total = max(len(cluster_rows), 1)
        background_total = max(len(rows), 1)
        for cohort_value in ordered_values:
            cluster_hits = int(cluster_value_counts.get(cohort_value, 0))
            background_hits = int(background_counts[cohort_value])
            cluster_fraction = cluster_hits / cluster_total
            background_fraction = background_hits / background_total
            enrichment_ratio = 0.0 if background_fraction == 0 else cluster_fraction / background_fraction
            output_rows.append(
                {
                    "cluster_label": int(cluster_label),
                    "cohort_column": column,
                    "cohort_value": cohort_value,
                    "cluster_size": int(cluster_total),
                    "cluster_hits": cluster_hits,
                    "cluster_fraction": float(cluster_fraction),
                    "background_hits": background_hits,
                    "background_fraction": float(background_fraction),
                    "enrichment_delta": float(cluster_fraction - background_fraction),
                    "enrichment_ratio": float(enrichment_ratio),
                }
            )
    return pa.Table.from_pylist(output_rows)


def _cluster_medoids_table(
    matrix: np.ndarray,
    rows: list[dict[str, object]],
    labels: np.ndarray,
    *,
    metric: str,
) -> pa.Table:
    output_rows: list[dict[str, object]] = []
    for cluster_label in sorted({int(label) for label in labels.tolist()}):
        indices = np.flatnonzero(labels == cluster_label)
        cluster_matrix = matrix[indices]
        centroid = np.asarray(cluster_matrix.mean(axis=0), dtype=np.float32)
        if metric == "cosine":
            norms = np.clip(np.linalg.norm(cluster_matrix, axis=1) * np.linalg.norm(centroid), a_min=1e-12, a_max=None)
            distances = 1.0 - ((cluster_matrix @ centroid) / norms)
        else:
            distances = np.linalg.norm(cluster_matrix - centroid, axis=1)
        local_index = int(np.argmin(distances))
        medoid_index = int(indices[local_index])
        output_rows.append(
            {
                "cluster_label": int(cluster_label),
                "medoid_index": medoid_index,
                "medoid_id": _representative_row_id(rows, medoid_index),
            }
        )
    return pa.Table.from_pylist(output_rows)


def _nearest_landmarks_table(
    context: WorkspaceContext,
    *,
    view_id: str | None,
    matrix: np.ndarray,
    rows: list[dict[str, object]],
    labels: np.ndarray,
    metric: str,
) -> pa.Table:
    if view_id is None:
        return pa.Table.from_pylist([])
    view = context.require_view(view_id)
    if not isinstance(view, SourceBackedViewConfig):
        return pa.Table.from_pylist([])

    candidate_landmarks = {
        landmark_id: landmark
        for landmark_id, landmark in context.config.landmarks.items()
        if landmark.source == view.source and rows and landmark.where.get("column") in rows[0]
    }
    if not candidate_landmarks:
        return pa.Table.from_pylist([])

    landmark_vectors: dict[str, np.ndarray] = {}
    for landmark_id, landmark in candidate_landmarks.items():
        indices = _select_indices(rows, landmark.where)
        if not indices:
            continue
        vectors = np.asarray(matrix[indices], dtype=np.float32)
        if landmark.representation.mode == "centroid":
            landmark_vectors[landmark_id] = np.asarray(vectors.mean(axis=0), dtype=np.float32)
        elif landmark.representation.mode == "medoid":
            distances = _euclidean_distance_matrix(vectors)
            medoid_index = int(np.argmin(distances.sum(axis=1)))
            landmark_vectors[landmark_id] = np.asarray(vectors[medoid_index], dtype=np.float32)
        else:
            landmark_vectors[landmark_id] = np.asarray(vectors[0], dtype=np.float32)
    if not landmark_vectors:
        return pa.Table.from_pylist([])

    output_rows: list[dict[str, object]] = []
    for cluster_label in sorted({int(label) for label in labels.tolist()}):
        indices = np.flatnonzero(labels == cluster_label)
        centroid = np.asarray(matrix[indices].mean(axis=0), dtype=np.float32)
        best_landmark = None
        best_distance = None
        for landmark_id, vector in landmark_vectors.items():
            if metric == "cosine":
                denominator = float(np.clip(np.linalg.norm(centroid) * np.linalg.norm(vector), a_min=1e-12, a_max=None))
                distance = 1.0 - float(np.dot(centroid, vector) / denominator)
            else:
                distance = float(np.linalg.norm(centroid - vector))
            if best_distance is None or distance < best_distance:
                best_landmark = landmark_id
                best_distance = distance
        output_rows.append(
            {
                "cluster_label": int(cluster_label),
                "nearest_landmark_id": best_landmark,
                "distance": float(best_distance or 0.0),
            }
        )
    return pa.Table.from_pylist(output_rows)


def fit_cluster_artifact(
    context: WorkspaceContext,
    *,
    cluster_id: str,
    view_id: str | None,
    reduced_view_id: str | None,
    method: str,
    n_clusters: int | None,
    seed: int,
    max_iter: int,
    sample_id: str | None,
    alignment_id: str | None,
    neighbor_set_id: str | None,
    metric: str,
    k: int,
    resolution: float,
) -> tuple[Path, dict[str, object]]:
    matrix, rows_table, scope_kind, scope_id = resolve_feature_scope(
        context,
        view_id=view_id,
        reduced_view_id=reduced_view_id,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    row_count = int(rows_table.num_rows)
    if row_count < 1:
        raise ContractViolationError("cluster fitting requires at least one row")

    matrix = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    if method == "kmeans":
        if n_clusters is None:
            raise ContractViolationError("kmeans clustering requires --n-clusters")
        labels, _, iterations, converged = _run_kmeans(matrix, n_clusters=n_clusters, seed=seed, max_iter=max_iter)
        summary: dict[str, object] = {
            "method": "kmeans",
            "view_id": view_id,
            "reduced_view_id": reduced_view_id,
            "rows": row_count,
            "scope_kind": scope_kind,
            "scope_id": scope_id,
            "n_clusters": int(n_clusters),
            "seed": seed,
            "max_iter": max_iter,
            "iterations": iterations,
            "converged": converged,
        }
    elif method == "leiden":
        neighbor_indices = None
        if neighbor_set_id is not None:
            from ..io.matrix_io import read_matrix
            from ..io.parquet_io import read_table

            neighbor_rows = read_table(context.output_root / "neighbors" / neighbor_set_id / "rows.parquet")
            if not neighbor_rows.equals(rows_table, check_metadata=False):
                raise ContractViolationError(
                    f"neighbor set {neighbor_set_id} does not match the requested cluster scope for {cluster_id}"
                )
            neighbor_indices = np.asarray(
                read_matrix(context.output_root / "neighbors" / neighbor_set_id / "indices.npy", mmap_mode=None),
                dtype=np.int64,
            )
        labels = _run_leiden(
            matrix,
            k=k,
            resolution=resolution,
            metric=metric,
            seed=seed,
            neighbor_indices=neighbor_indices,
        )
        summary = {
            "method": "leiden",
            "view_id": view_id,
            "reduced_view_id": reduced_view_id,
            "rows": row_count,
            "scope_kind": scope_kind,
            "scope_id": scope_id,
            "seed": seed,
            "metric": metric,
            "k": int(max(1, min(int(k), row_count - 1))),
            "resolution": float(resolution),
            "neighbor_set_id": neighbor_set_id,
        }
    else:
        raise ContractViolationError(f"unsupported clustering method: {method!r}")

    rows = rows_table.to_pylist()
    assignments = rows_table.append_column("cluster_label", pa.array(labels.tolist(), type=pa.int64()))
    artifact_dir = context.output_root / "clusters" / cluster_id
    write_table(assignments, artifact_dir / "assignments.parquet")

    cluster_sizes = _cluster_sizes_table(labels)
    write_table(cluster_sizes, artifact_dir / "cluster_sizes.parquet")
    medoids = _cluster_medoids_table(matrix, rows, labels, metric=metric)
    write_table(medoids, artifact_dir / "medoids.parquet")
    nearest_landmarks = _nearest_landmarks_table(
        context,
        view_id=view_id,
        matrix=matrix,
        rows=rows,
        labels=labels,
        metric=metric,
    )
    write_table(nearest_landmarks, artifact_dir / "nearest_landmarks.parquet")

    for column in ("design_family", "design_regulator_composition"):
        if rows and column in rows[0]:
            write_table(
                _cluster_enrichment_table(rows, labels, column=column),
                artifact_dir / f"cluster_enrichment__{column}.parquet",
            )

    summary["cluster_sizes"] = {int(label): int(count) for label, count in Counter(labels.tolist()).items()}
    write_json(artifact_dir / "summary.json", summary)
    return artifact_dir, summary
