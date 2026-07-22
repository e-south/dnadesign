"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/views/reduce.py

Reducer fitting and reduced-view builders for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
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
from .pca_policy import select_pca_method, streaming_batch_rows


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
        matrix=read_matrix(matrix_path),
        rows=read_table(rows_path),
        scope_kind="full_view",
        scope_id=view_id,
    )


def _sample_scope(context: WorkspaceContext, view_id: str, *, sample_id: str) -> ScopeMatrix:
    matrix_path, rows_path, manifest = _view_paths(context, view_id)
    sample_rows_path = context.output_root / "samples" / sample_id / "rows.parquet"
    if not sample_rows_path.exists():
        raise MissingArtifactError(f"sample artifact is missing for reduction: {sample_id}")
    matrix = read_matrix(matrix_path)
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
    matrix = read_matrix(matrix_path)
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
    total_variance = np.float32(
        np.sum(np.square(singular_values, dtype=np.float64), dtype=np.float64) / max(matrix.shape[0] - 1, 1)
    )
    explained_variance_ratio = _explained_variance_ratio(explained_variance, total_variance=total_variance)
    mean = np.asarray(matrix.mean(axis=0), dtype=np.float32)
    return mean, components, explained_variance, explained_variance_ratio


def _batch_ranges(*, total_rows: int, batch_rows: int, min_rows: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_rows:
        stop = min(total_rows, start + batch_rows)
        if ranges and total_rows - stop < min_rows:
            previous_start, _ = ranges[-1]
            ranges[-1] = (previous_start, total_rows)
            break
        ranges.append((start, stop))
        start = stop
    return ranges or [(0, total_rows)]


def _centered_right_multiply(matrix: np.ndarray, mean: np.ndarray, columns: np.ndarray) -> np.ndarray:
    projected = np.asarray(matrix @ columns, dtype=np.float32)
    projected -= np.asarray(mean @ columns, dtype=np.float32)
    return projected


def _centered_left_multiply(matrix: np.ndarray, mean: np.ndarray, weights: np.ndarray) -> np.ndarray:
    centered = np.asarray(matrix.T @ weights, dtype=np.float32)
    centered -= np.outer(mean, np.asarray(weights.sum(axis=0), dtype=np.float32))
    return centered


def _explained_variance_ratio(explained_variance: np.ndarray, *, total_variance: float) -> np.ndarray:
    if total_variance <= 0.0:
        return np.zeros_like(explained_variance, dtype=np.float32)
    return np.asarray(
        explained_variance / float(total_variance),
        dtype=np.float32,
    )


def _centered_total_variance(matrix: np.ndarray, *, mean: np.ndarray) -> np.float32:
    sample_count = int(matrix.shape[0])
    batch_rows = streaming_batch_rows(
        total_rows=sample_count,
        dims=int(matrix.shape[1]),
        itemsize=matrix.dtype.itemsize,
        output_dims=1,
    )
    squared_norm = 0.0
    for start, stop in _batch_ranges(total_rows=sample_count, batch_rows=batch_rows, min_rows=1):
        batch = np.asarray(matrix[start:stop], dtype=np.float32)
        squared_norm += float(np.einsum("ij,ij->", batch, batch, dtype=np.float64))
    mean_sq_norm = float(np.dot(np.asarray(mean, dtype=np.float64), np.asarray(mean, dtype=np.float64)))
    centered_squared_norm = max(squared_norm - sample_count * mean_sq_norm, 0.0)
    return np.float32(centered_squared_norm / max(sample_count - 1, 1))


def _fit_randomized_pca(
    matrix: np.ndarray,
    *,
    dims: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ContractViolationError("PCA input matrix must be 2D")
    if matrix.shape[0] < 2:
        raise ContractViolationError("PCA reduction requires at least 2 rows")
    max_dims = min(matrix.shape[0], matrix.shape[1])
    if dims < 1 or dims > max_dims:
        raise ContractViolationError(f"PCA dims must be between 1 and {max_dims}, got {dims}")

    mean = np.asarray(matrix.mean(axis=0), dtype=np.float64).astype(np.float32, copy=False)
    sample_count = int(matrix.shape[0])
    feature_count = int(matrix.shape[1])
    sketch_rank = min(max(dims + 8, dims + 2), feature_count, sample_count)
    rng = np.random.default_rng(seed)
    omega = np.asarray(rng.normal(size=(feature_count, sketch_rank)), dtype=np.float32)
    sketch = _centered_right_multiply(matrix, mean, omega)
    for _ in range(1):
        basis, _ = np.linalg.qr(sketch, mode="reduced")
        sketch = _centered_right_multiply(matrix, mean, _centered_left_multiply(matrix, mean, basis))
    basis, _ = np.linalg.qr(sketch, mode="reduced")
    compressed = _centered_left_multiply(matrix, mean, basis).T
    _, singular_values, vt = np.linalg.svd(compressed, full_matrices=False)
    explained_variance = np.asarray((singular_values[:dims] ** 2) / max(sample_count - 1, 1), dtype=np.float32)
    components = np.asarray(vt[:dims], dtype=np.float32)
    total_variance = _centered_total_variance(matrix, mean=mean)
    explained_variance_ratio = _explained_variance_ratio(explained_variance, total_variance=total_variance)
    mean = np.asarray(mean, dtype=np.float32)
    return mean, components, explained_variance, explained_variance_ratio


def _fit_pca_dispatch(
    matrix: np.ndarray,
    *,
    dims: int,
    seed: int,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    method = select_pca_method(rows=int(matrix.shape[0]), dims=int(matrix.shape[1]), itemsize=matrix.dtype.itemsize)
    if method == "dense_svd":
        mean, components, explained_variance, explained_variance_ratio = _fit_pca(
            np.asarray(matrix, dtype=np.float32),
            dims=dims,
        )
    else:
        mean, components, explained_variance, explained_variance_ratio = _fit_randomized_pca(
            matrix,
            dims=dims,
            seed=seed,
        )
    return method, mean, components, explained_variance, explained_variance_ratio


def _transform(matrix: np.ndarray, *, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    reduced = np.asarray(matrix @ components.T, dtype=np.float32) - np.asarray(mean @ components.T, dtype=np.float32)
    return np.ascontiguousarray(np.asarray(reduced, dtype=np.float32))


def _write_transformed_matrix(
    matrix: np.ndarray,
    *,
    mean: np.ndarray,
    components: np.ndarray,
    target_path: Path,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    batch_rows = streaming_batch_rows(
        total_rows=int(matrix.shape[0]),
        dims=int(matrix.shape[1]),
        itemsize=np.dtype(np.float32).itemsize,
        output_dims=int(components.shape[0]),
    )
    transform_method = select_pca_method(
        rows=int(matrix.shape[0]),
        dims=int(matrix.shape[1]),
        itemsize=matrix.dtype.itemsize,
    )
    if transform_method == "dense_svd":
        write_matrix(target_path, _transform(np.asarray(matrix, dtype=np.float32), mean=mean, components=components))
        return

    output = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(matrix.shape[0]), int(components.shape[0])),
    )
    for start, stop in _batch_ranges(
        total_rows=int(matrix.shape[0]),
        batch_rows=batch_rows,
        min_rows=max(int(components.shape[0]), 1),
    ):
        output[start:stop] = _transform(matrix[start:stop], mean=mean, components=components)
    output.flush()
    del output


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
) -> tuple[Path, Path | None, int, int, str, str | None, str]:
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

    pca_method, mean, components, explained_variance, explained_variance_ratio = _fit_pca_dispatch(
        fit_scope.matrix,
        dims=dims,
        seed=context.config.defaults.random_seed,
    )
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
            "pca_method": pca_method,
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
        _write_transformed_matrix(
            transform_scope.matrix,
            mean=mean,
            components=components,
            target_path=target_reduced_view_dir / "matrix.npy",
        )
        write_table(transform_scope.rows, target_reduced_view_dir / "rows.parquet")

    return (
        target_reducer_dir,
        target_reduced_view_dir,
        int(fit_scope.matrix.shape[0]),
        int(components.shape[0]),
        fit_scope.scope_kind,
        fit_scope.scope_id,
        pca_method,
    )
