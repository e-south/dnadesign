"""
Reducer fitting and reduced-view builders for latentdna.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from ..alignments.aggregators import aggregate_rows
from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..io.json_io import write_json
from ..io.matrix_io import read_matrix, write_matrix
from ..io.parquet_io import read_table, write_table
from ..workspaces.loader import WorkspaceContext


@dataclass(frozen=True, slots=True)
class ScopeMatrix:
    matrix: np.ndarray
    rows: pa.Table
    scope_kind: str
    scope_id: str | None


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


def _view_paths(context: WorkspaceContext, view_id: str) -> tuple[Path, Path, dict[str, object]]:
    view_dir = context.output_root / "views" / view_id
    matrix_path = view_dir / "matrix.npy"
    rows_path = view_dir / "rows.parquet"
    manifest_path = view_dir / "manifest.json"
    for required in [matrix_path, rows_path, manifest_path]:
        if not required.exists():
            raise MissingArtifactError(f"view artifact is missing for reduction: {required}")
    return matrix_path, rows_path, context.read_manifest(manifest_path)


def _full_scope(context: WorkspaceContext, view_id: str) -> ScopeMatrix:
    matrix_path, rows_path, _ = _view_paths(context, view_id)
    return ScopeMatrix(
        matrix=np.asarray(read_matrix(matrix_path), dtype=np.float32),
        rows=read_table(rows_path),
        scope_kind="full_view",
        scope_id=view_id,
    )


def _sample_scope(context: WorkspaceContext, view_id: str, *, sample_id: str) -> ScopeMatrix:
    matrix_path, rows_path, manifest = _view_paths(context, view_id)
    sample_rows_path = context.output_root / "samples" / sample_id / "rows.parquet"
    if not sample_rows_path.exists():
        raise MissingArtifactError(f"sample artifact is missing for reduction: {sample_id}")
    matrix = np.asarray(read_matrix(matrix_path), dtype=np.float32)
    view_rows = read_table(rows_path).to_pylist()
    sample_rows = read_table(sample_rows_path).to_pylist()
    record_key = str(manifest["params"]["record_key"])
    indices = _ordered_indices(view_rows, sample_rows, record_key=record_key)
    return ScopeMatrix(
        matrix=np.asarray(matrix[indices], dtype=np.float32),
        rows=pa.Table.from_pylist(sample_rows),
        scope_kind="sample_set",
        scope_id=sample_id,
    )


def _alignment_scope(context: WorkspaceContext, view_id: str, *, alignment_id: str) -> ScopeMatrix:
    matrix_path, _, _ = _view_paths(context, view_id)
    alignment_dir = context.output_root / "alignments" / alignment_id
    alignment_manifest_path = alignment_dir / "manifest.json"
    mapping_path = alignment_dir / "mapping.parquet"
    rows_path = alignment_dir / "rows.parquet"
    for required in [alignment_manifest_path, mapping_path, rows_path]:
        if not required.exists():
            raise MissingArtifactError(f"alignment artifact is missing for reduction: {required}")

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
    return ScopeMatrix(
        matrix=np.ascontiguousarray(aligned_matrix),
        rows=rows,
        scope_kind="alignment_set",
        scope_id=alignment_id,
    )


def _fit_pca(matrix: np.ndarray, *, dims: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ContractViolationError("PCA input matrix must be 2D")
    if matrix.shape[0] < 2:
        raise ContractViolationError("PCA reduction requires at least 2 rows")
    max_dims = min(matrix.shape[0], matrix.shape[1])
    if dims < 1 or dims > max_dims:
        raise ContractViolationError(f"PCA dims must be between 1 and {max_dims}, got {dims}")

    centered = np.asarray(matrix, dtype=np.float32) - np.asarray(matrix.mean(axis=0), dtype=np.float32)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = np.asarray(vt[:dims], dtype=np.float32)
    explained_variance = np.asarray((singular_values[:dims] ** 2) / (matrix.shape[0] - 1), dtype=np.float32)
    explained_variance_ratio = np.asarray(
        explained_variance / np.clip(explained_variance.sum(), a_min=1e-12, a_max=None),
        dtype=np.float32,
    )
    mean = np.asarray(matrix.mean(axis=0), dtype=np.float32)
    return mean, components, explained_variance, explained_variance_ratio


def _transform(matrix: np.ndarray, *, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    reduced = (np.asarray(matrix, dtype=np.float32) - mean) @ components.T
    return np.ascontiguousarray(np.asarray(reduced, dtype=np.float32))


def fit_pca_reducer_artifacts(
    context: WorkspaceContext,
    *,
    view_id: str,
    reducer_id: str,
    dims: int,
    sample_id: str | None,
    alignment_id: str | None,
    reduced_view_id: str | None,
    reducer_dir: Path | None = None,
    reduced_view_dir: Path | None = None,
) -> tuple[Path, Path | None, int, int, str, str | None]:
    if sample_id and alignment_id:
        raise ContractViolationError("view reduce accepts at most one fit scope of --sample or --alignment")

    if alignment_id:
        fit_scope = _alignment_scope(context, view_id, alignment_id=alignment_id)
        transform_scope = fit_scope
    elif sample_id:
        fit_scope = _sample_scope(context, view_id, sample_id=sample_id)
        transform_scope = _full_scope(context, view_id)
    else:
        fit_scope = _full_scope(context, view_id)
        transform_scope = fit_scope

    mean, components, explained_variance, explained_variance_ratio = _fit_pca(fit_scope.matrix, dims=dims)
    target_reducer_dir = reducer_dir or (context.output_root / "reducers" / reducer_id)
    target_reducer_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        target_reducer_dir / "state.npz",
        mean=mean,
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
    )
    write_json(
        target_reducer_dir / "summary.json",
        {
            "method": "pca",
            "fit_rows": int(fit_scope.matrix.shape[0]),
            "input_dims": int(fit_scope.matrix.shape[1]),
            "output_dims": int(components.shape[0]),
            "scope_kind": fit_scope.scope_kind,
            "scope_id": fit_scope.scope_id,
            "explained_variance_ratio": explained_variance_ratio.tolist(),
        },
    )

    target_reduced_view_dir = reduced_view_dir
    if reduced_view_id is not None:
        target_reduced_view_dir = target_reduced_view_dir or (context.output_root / "reduced_views" / reduced_view_id)
        reduced_matrix = _transform(transform_scope.matrix, mean=mean, components=components)
        write_matrix(target_reduced_view_dir / "matrix.npy", reduced_matrix)
        write_table(transform_scope.rows, target_reduced_view_dir / "rows.parquet")

    return (
        target_reducer_dir,
        target_reduced_view_dir,
        int(fit_scope.matrix.shape[0]),
        int(components.shape[0]),
        fit_scope.scope_kind,
        fit_scope.scope_id,
    )
