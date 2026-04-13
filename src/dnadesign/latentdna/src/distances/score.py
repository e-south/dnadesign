"""
Distance scoring helpers for latentdna.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..io.matrix_io import read_matrix
from ..io.parquet_io import read_table, write_table
from ..workspaces.loader import WorkspaceContext


def _select_indices(rows: list[dict[str, Any]], where: dict[str, Any]) -> list[int]:
    column = where.get("column")
    if not isinstance(column, str):
        raise ContractViolationError("landmark where clause requires a 'column' field")
    if "equals" in where:
        target = where["equals"]
        return [index for index, row in enumerate(rows) if row.get(column) == target]
    if "in" in where:
        targets = set(where["in"])
        return [index for index, row in enumerate(rows) if row.get(column) in targets]
    raise ContractViolationError("landmark where clause requires either 'equals' or 'in'")


def _fallback_control_indices(rows: list[dict[str, Any]]) -> list[int]:
    indices: list[int] = []
    for index, row in enumerate(rows):
        label = str(row.get("usr_label__primary") or "").strip().lower()
        if row.get("is_control") is True:
            indices.append(index)
            continue
        if str(row.get("source_class") or "").strip() == "manual_or_wildtype":
            indices.append(index)
            continue
        if str(row.get("design_family") or "").strip() == "control":
            indices.append(index)
            continue
        if label in {"spyp", "sulap", "soxsp", "j23105"}:
            indices.append(index)
    return indices


def _cosine_distances(matrix: np.ndarray, reference: np.ndarray) -> np.ndarray:
    matrix_norms = np.linalg.norm(matrix, axis=1)
    ref_norm = np.linalg.norm(reference)
    denominator = np.clip(matrix_norms * ref_norm, a_min=1e-12, a_max=None)
    return 1.0 - ((matrix @ reference) / denominator)


def _euclidean_distances(matrix: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.linalg.norm(matrix - reference, axis=1)


def _distance_column(matrix: np.ndarray, reference: np.ndarray, *, metric: str) -> np.ndarray:
    if metric == "cosine":
        return _cosine_distances(matrix, reference)
    if metric == "euclidean":
        return _euclidean_distances(matrix, reference)
    raise ContractViolationError(f"unsupported distance metric: {metric!r}")


def _medoid_vector(vectors: np.ndarray, *, metric: str) -> np.ndarray:
    if len(vectors) == 1:
        return vectors[0]
    if metric == "cosine":
        normalized = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), a_min=1e-12, a_max=None)
        distances = 1.0 - normalized @ normalized.T
    elif metric == "euclidean":
        deltas = vectors[:, None, :] - vectors[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
    else:  # pragma: no cover - checked earlier
        raise ContractViolationError(f"unsupported distance metric for medoid: {metric!r}")
    medoid_index = int(np.argmin(distances.sum(axis=1)))
    return vectors[medoid_index]


def score_distance_artifact(
    context: WorkspaceContext,
    *,
    distance_id: str,
    view_id: str,
    landmark_ids: list[str],
    metric: str,
) -> tuple[Path, int, list[str], dict[str, str], dict[str, int]]:
    view = context.require_source_view(view_id)
    if not landmark_ids:
        raise ContractViolationError("distance scoring requires at least one --landmark")

    matrix_path = context.output_root / "views" / view_id / "matrix.npy"
    rows_path = context.output_root / "views" / view_id / "rows.parquet"
    if not matrix_path.exists() or not rows_path.exists():
        raise MissingArtifactError(f"view artifact is missing for distance scoring: {view_id}")

    matrix = np.asarray(read_matrix(matrix_path), dtype=np.float32)
    rows_table = read_table(rows_path)
    rows = rows_table.to_pylist()
    output_table = rows_table
    representation_modes: dict[str, str] = {}
    member_counts: dict[str, int] = {}

    for landmark_id in landmark_ids:
        landmark = context.require_landmark(landmark_id)
        if landmark.source != view.source:
            raise ContractViolationError(
                f"landmark {landmark_id} uses source {landmark.source!r} but view {view_id} uses {view.source!r}"
            )
        indices = _select_indices(rows, landmark.where)
        if not indices:
            indices = _fallback_control_indices(rows)
        if not indices:
            raise ContractViolationError(f"landmark {landmark_id} matched no rows in view {view_id}")
        representation_modes[landmark_id] = landmark.representation.mode
        member_counts[landmark_id] = len(indices)
        vectors = np.asarray(matrix[indices], dtype=np.float32)
        if landmark.representation.mode == "centroid":
            output_table = output_table.append_column(
                f"d_{landmark_id}",
                pa.array(_distance_column(matrix, vectors.mean(axis=0), metric=metric).tolist()),
            )
        elif landmark.representation.mode == "medoid":
            output_table = output_table.append_column(
                f"d_{landmark_id}",
                pa.array(_distance_column(matrix, _medoid_vector(vectors, metric=metric), metric=metric).tolist()),
            )
        elif landmark.representation.mode == "rows":
            for member_number, reference in enumerate(vectors, start=1):
                output_table = output_table.append_column(
                    f"d_{landmark_id}__member_{member_number:03d}",
                    pa.array(_distance_column(matrix, reference, metric=metric).tolist()),
                )
        else:  # pragma: no cover - constrained by config model
            raise ContractViolationError(f"unsupported landmark representation mode: {landmark.representation.mode}")

    artifact_dir = context.output_root / "distances" / distance_id
    write_table(output_table, artifact_dir / "table.parquet")
    return artifact_dir, output_table.num_rows, output_table.column_names, representation_modes, member_counts
