"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/distances/score.py

Distance scoring helpers for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
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


def _alignment_input_uses_source(context: WorkspaceContext, ref_id: str, source_id: str) -> bool:
    if ref_id == source_id:
        return True
    view = context.config.views.get(ref_id)
    return bool(view is not None and getattr(view, "source", None) == source_id)


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


def _alignment_projected_indices(
    context: WorkspaceContext,
    *,
    alignment_id: str,
    view_id: str,
    landmark_id: str,
    selector_column: str,
    where: dict[str, Any],
) -> list[int]:
    alignment = context.require_alignment(alignment_id)
    if view_id not in {alignment.left, alignment.right}:
        raise ContractViolationError(f"alignment {alignment_id} does not include view {view_id!r}")

    landmark = context.require_landmark(landmark_id)
    left_matches = _alignment_input_uses_source(context, alignment.left, landmark.source)
    right_matches = _alignment_input_uses_source(context, alignment.right, landmark.source)
    if left_matches == right_matches:
        raise ContractViolationError(
            f"alignment {alignment_id} cannot resolve landmark source {landmark.source!r} for view {view_id!r}"
        )

    matched_side = "left" if left_matches else "right"
    target_side = "left" if alignment.left == view_id else "right"

    if matched_side == "left":
        matched_rows_path = (
            context.output_root / "views" / alignment.left / "rows.parquet"
            if alignment.left in context.config.views
            else None
        )
    else:
        matched_rows_path = (
            context.output_root / "views" / alignment.right / "rows.parquet"
            if alignment.right in context.config.views
            else None
        )

    if matched_rows_path is not None and matched_rows_path.is_file():
        matched_rows_table = read_table(matched_rows_path)
    else:
        source = context.require_source(landmark.source)
        from ..sources.resolver import read_records_table, resolve_source

        resolved = resolve_source(landmark.source, source, workspace_dir=context.workspace_dir)
        matched_rows_table = read_records_table(resolved, columns=[selector_column])

    if selector_column not in matched_rows_table.column_names:
        raise ContractViolationError(
            "landmark "
            f"{landmark_id} selector column {selector_column!r} is not present "
            f"in alignment input {landmark.source!r}"
        )

    matched_rows = matched_rows_table.to_pylist()
    matched_indices = _select_indices(matched_rows, where)
    if not matched_indices:
        raise ContractViolationError(f"landmark {landmark_id} matched no rows in alignment input for {alignment_id!r}")

    mapping_path = context.output_root / "alignments" / alignment_id / "mapping.parquet"
    if not mapping_path.is_file():
        raise MissingArtifactError(f"alignment mapping artifact is missing: {alignment_id}")
    mapping_rows = read_table(mapping_path).to_pylist()

    matched_index_set = set(matched_indices)
    target_indices: set[int] = set()
    matched_column = f"{matched_side}_indices"
    target_column = f"{target_side}_indices"
    for row in mapping_rows:
        current_matched = {int(index) for index in row.get(matched_column, [])}
        if current_matched.intersection(matched_index_set):
            target_indices.update(int(index) for index in row.get(target_column, []))
    if not target_indices:
        raise ContractViolationError(
            f"landmark {landmark_id} matched no aligned rows in {alignment_id!r} for view {view_id!r}"
        )
    return sorted(target_indices)


def score_distance_artifact(
    context: WorkspaceContext,
    *,
    distance_id: str,
    view_id: str,
    landmark_ids: list[str],
    metric: str,
    alignment_id: str | None = None,
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
        if landmark.source == view.source:
            indices = _select_indices(rows, landmark.where)
        elif alignment_id is not None:
            indices = _alignment_projected_indices(
                context,
                alignment_id=alignment_id,
                view_id=view_id,
                landmark_id=landmark_id,
                selector_column=str(landmark.where["column"]),
                where=landmark.where,
            )
        else:
            raise ContractViolationError(
                f"landmark {landmark_id} uses source {landmark.source!r} but view {view_id} uses {view.source!r}; "
                "rerun with --alignment to project landmark rows onto the view support explicitly"
            )
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
