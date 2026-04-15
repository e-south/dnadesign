"""
Shared scoped-matrix helpers for latentdna view-backed artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from ..alignments.aggregators import aggregate_rows
from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..io.matrix_io import read_matrix
from ..io.parquet_io import read_table
from ..workspaces.loader import WorkspaceContext


def _ordered_indices(view_rows: list[dict], sample_rows: list[dict], *, record_key: str) -> list[int]:
    index_by_key = {row[record_key]: index for index, row in enumerate(view_rows)}
    indices: list[int] = []
    missing: list[str] = []
    for row in sample_rows:
        key = row[record_key]
        if key not in index_by_key:
            missing.append(str(key))
            continue
        indices.append(index_by_key[key])
    if missing:
        raise ContractViolationError(f"sample rows are not aligned to view on {record_key}: missing {missing[:5]}")
    return indices


def _matrix_source_paths(
    context: WorkspaceContext,
    *,
    view_id: str | None,
    reduced_view_id: str | None,
) -> tuple[Path, Path, dict[str, object], str, str]:
    if (view_id is None) == (reduced_view_id is None):
        raise ContractViolationError("matrix scope requires exactly one of view_id or reduced_view_id")
    artifact_kind = "view" if view_id is not None else "reduced_view"
    artifact_id = str(view_id if view_id is not None else reduced_view_id)
    artifact_dir = context.output_root / ("views" if artifact_kind == "view" else "reduced_views") / artifact_id
    matrix_path = artifact_dir / "matrix.npy"
    rows_path = artifact_dir / "rows.parquet"
    manifest_path = artifact_dir / "manifest.json"
    for required in [matrix_path, rows_path, manifest_path]:
        if not required.exists():
            raise MissingArtifactError(f"view artifact is missing for scoped view access: {required}")
    return matrix_path, rows_path, context.read_manifest(manifest_path), artifact_kind, artifact_id


def view_artifact_paths(context: WorkspaceContext, view_id: str) -> tuple[Path, Path, dict[str, object]]:
    matrix_path, rows_path, manifest, _, _ = _matrix_source_paths(context, view_id=view_id, reduced_view_id=None)
    return matrix_path, rows_path, manifest


def matrix_input_digest_path(
    context: WorkspaceContext,
    *,
    view_id: str | None,
    reduced_view_id: str | None,
) -> tuple[str, str, Path]:
    matrix_path, _, _, artifact_kind, artifact_id = _matrix_source_paths(
        context,
        view_id=view_id,
        reduced_view_id=reduced_view_id,
    )
    input_kind = "view_matrix" if artifact_kind == "view" else "reduced_view"
    return input_kind, artifact_id, matrix_path


def _full_scope(context: WorkspaceContext, view_id: str) -> tuple[np.ndarray, pa.Table, str, str | None]:
    matrix_path, rows_path, _ = view_artifact_paths(context, view_id)
    return np.asarray(read_matrix(matrix_path), dtype=np.float32), read_table(rows_path), "full_view", view_id


def _sample_scope(
    context: WorkspaceContext,
    view_id: str,
    *,
    sample_id: str,
) -> tuple[np.ndarray, pa.Table, str, str | None]:
    matrix_path, rows_path, manifest = view_artifact_paths(context, view_id)
    sample_rows_path = context.output_root / "samples" / sample_id / "rows.parquet"
    if not sample_rows_path.exists():
        raise MissingArtifactError(f"sample artifact is missing for scoped view access: {sample_id}")
    matrix = np.asarray(read_matrix(matrix_path), dtype=np.float32)
    view_rows = read_table(rows_path).to_pylist()
    sample_rows = read_table(sample_rows_path).to_pylist()
    record_key = str(manifest["params"]["record_key"])
    indices = _ordered_indices(view_rows, sample_rows, record_key=record_key)
    return np.asarray(matrix[indices], dtype=np.float32), pa.Table.from_pylist(sample_rows), "sample_set", sample_id


def _alignment_scope(
    context: WorkspaceContext,
    view_id: str,
    *,
    alignment_id: str,
) -> tuple[np.ndarray, pa.Table, str, str | None]:
    matrix_path, _, _ = view_artifact_paths(context, view_id)
    alignment_dir = context.output_root / "alignments" / alignment_id
    alignment_manifest_path = alignment_dir / "manifest.json"
    mapping_path = alignment_dir / "mapping.parquet"
    rows_path = alignment_dir / "rows.parquet"
    for required in [alignment_manifest_path, mapping_path, rows_path]:
        if not required.exists():
            raise MissingArtifactError(f"alignment artifact is missing for scoped view access: {required}")

    alignment_manifest = context.read_manifest(alignment_manifest_path)
    matrix = np.asarray(read_matrix(matrix_path), dtype=np.float32)
    mapping_rows = read_table(mapping_path).to_pylist()
    rows = read_table(rows_path)
    if alignment_manifest["params"]["left"] == view_id:
        index_field = "left_indices"
        mode = str(alignment_manifest["params"]["left_aggregation"])
    elif alignment_manifest["params"]["right"] == view_id:
        index_field = "right_indices"
        mode = str(alignment_manifest["params"]["right_aggregation"])
    else:
        raise ContractViolationError(f"alignment {alignment_id} does not include view {view_id}")

    aligned_matrix = np.vstack(
        [aggregate_rows(matrix, list(row[index_field]), mode=mode) for row in mapping_rows]
    ).astype(np.float32, copy=False)
    return np.ascontiguousarray(aligned_matrix), rows, "alignment_set", alignment_id


def resolve_view_scope(
    context: WorkspaceContext,
    *,
    view_id: str,
    sample_id: str | None,
    alignment_id: str | None,
) -> tuple[np.ndarray, pa.Table, str, str | None]:
    if sample_id and alignment_id:
        raise ContractViolationError("scope-aware view access accepts at most one scope of sample or alignment")
    if alignment_id:
        return _alignment_scope(context, view_id, alignment_id=alignment_id)
    if sample_id:
        return _sample_scope(context, view_id, sample_id=sample_id)
    return _full_scope(context, view_id)


def resolve_feature_scope(
    context: WorkspaceContext,
    *,
    view_id: str | None,
    reduced_view_id: str | None,
    sample_id: str | None,
    alignment_id: str | None,
) -> tuple[np.ndarray, pa.Table, str, str | None]:
    if reduced_view_id is not None:
        if sample_id is not None or alignment_id is not None:
            raise ContractViolationError(
                "reduced-view scoped access does not accept sample or alignment; "
                "reduce the scoped view first and pass --reduced-view"
            )
        matrix_path, rows_path, _, _, artifact_id = _matrix_source_paths(
            context,
            view_id=None,
            reduced_view_id=reduced_view_id,
        )
        return (
            np.asarray(read_matrix(matrix_path), dtype=np.float32),
            read_table(rows_path),
            "reduced_view",
            artifact_id,
        )
    if view_id is None:
        raise ContractViolationError("scope-aware matrix access requires exactly one of view_id or reduced_view_id")
    return resolve_view_scope(
        context,
        view_id=view_id,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )


def scope_input_digest_path(
    context: WorkspaceContext,
    *,
    view_id: str | None,
    reduced_view_id: str | None,
    sample_id: str | None,
    alignment_id: str | None,
) -> tuple[str, str, Path]:
    if reduced_view_id is not None:
        if sample_id is not None or alignment_id is not None:
            raise ContractViolationError(
                "reduced-view digest scope does not accept sample or alignment; "
                "reduce the scoped view first and pass --reduced-view"
            )
        _, rows_path, _, artifact_kind, artifact_id = _matrix_source_paths(
            context,
            view_id=None,
            reduced_view_id=reduced_view_id,
        )
        return artifact_kind, artifact_id, rows_path
    if alignment_id is not None:
        return "alignment_set", alignment_id, context.output_root / "alignments" / alignment_id / "rows.parquet"
    if sample_id is not None:
        return "sample_set", sample_id, context.output_root / "samples" / sample_id / "rows.parquet"
    return "view_rows", str(view_id), context.output_root / "views" / str(view_id) / "rows.parquet"
