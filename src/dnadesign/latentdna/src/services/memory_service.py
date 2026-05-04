"""
Memory preflight helpers for heavy latentdna operations.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa

from ..contracts.errors import ContractViolationError, MemoryPreflightError, MissingArtifactError
from ..contracts.workspace import ReducedViewExportBlockConfig, TableColumnsExportBlockConfig
from ..io.matrix_io import read_matrix
from ..io.parquet_io import read_row_count
from ..neighbors.backends.approximate import approximate_backend_available
from ..sources.resolver import (
    inspect_source_schema,
    iter_records_batches,
    missing_overlay_merge_columns,
    require_matrix_bundle_paths,
    resolve_source,
)
from ..views.pca_policy import select_pca_method, streaming_batch_rows
from ..workspaces.loader import WorkspaceContext
from ._status import merge_statuses

ArtifactState = Literal["ok", "warning", "blocked"]
_VIEW_MATERIALIZE_BATCH_ROWS = 2048
_VIEW_MATERIALIZE_RESIDENT_OUTPUT_FACTOR = 2.25


@dataclass(frozen=True, slots=True)
class MemoryPreflight:
    operation: str
    algorithm: str
    estimated_peak_bytes: int
    system_ram_bytes: int
    warn_fraction_of_system_ram: float
    fail_fraction_of_system_ram: float
    require_override_above_fail: bool
    state: ArtifactState
    rows: int
    dims: int
    dtype: str
    notes: list[str]

    @property
    def fraction_of_system_ram(self) -> float:
        return self.estimated_peak_bytes / max(self.system_ram_bytes, 1)

    def message(self) -> str:
        used_gib = self.estimated_peak_bytes / float(1024**3)
        total_gib = self.system_ram_bytes / float(1024**3)
        fraction = self.fraction_of_system_ram
        if self.state == "blocked":
            return (
                f"{self.operation} estimated peak {used_gib:.2f} GiB "
                f"({fraction:.2f} of system RAM {total_gib:.2f} GiB) exceeds fail threshold "
                f"{self.fail_fraction_of_system_ram:.2f}; rerun with --allow-memory-overage to proceed"
            )
        if self.state == "warning":
            return (
                f"{self.operation} estimated peak {used_gib:.2f} GiB "
                f"({fraction:.2f} of system RAM {total_gib:.2f} GiB) exceeds warn threshold "
                f"{self.warn_fraction_of_system_ram:.2f}"
            )
        return ""

    def as_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["estimated_peak_gib"] = round(self.estimated_peak_bytes / float(1024**3), 6)
        payload["system_ram_gib"] = round(self.system_ram_bytes / float(1024**3), 6)
        payload["fraction_of_system_ram"] = round(self.fraction_of_system_ram, 6)
        return payload


def system_ram_bytes() -> int:
    override = os.environ.get("LATENTDNA_SYSTEM_RAM_BYTES")
    if override is not None:
        value = int(override)
        if value < 1:
            raise MemoryPreflightError("LATENTDNA_SYSTEM_RAM_BYTES must be a positive integer")
        return value
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
            return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        pass
    try:
        import psutil
    except Exception as exc:  # pragma: no cover - sysconf should cover supported hosts
        raise MemoryPreflightError(
            "system RAM could not be determined for memory preflight; "
            "set LATENTDNA_SYSTEM_RAM_BYTES to override explicitly"
        ) from exc
    total = int(psutil.virtual_memory().total)
    if total < 1:
        raise MemoryPreflightError(
            "system RAM could not be determined for memory preflight; "
            "set LATENTDNA_SYSTEM_RAM_BYTES to override explicitly"
        )
    return total


def evaluate_reduce_preflight(
    context: WorkspaceContext,
    *,
    view_id: str,
    dims: int,
    sample_id: str | None,
    alignment_id: str | None,
    reduced_view_id: str | None,
) -> MemoryPreflight:
    rows, input_dims, dtype, itemsize = _view_metadata(context, view_id=view_id)
    fit_rows, notes = _scope_rows(
        context,
        view_id=view_id,
        reduced_view_id=None,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    transform_rows = rows if sample_id is not None else fit_rows
    pca_method = select_pca_method(rows=fit_rows, dims=input_dims, itemsize=itemsize)
    fit_bytes = _matrix_bytes(fit_rows, input_dims, itemsize)
    if pca_method == "dense_svd":
        base_bytes = _matrix_bytes(rows, input_dims, itemsize)
        svd_workspace = (_matrix_bytes(fit_rows, input_dims, itemsize) * 2) + (
            min(fit_rows, input_dims) ** 2 * itemsize * 2
        )
        if sample_id is not None:
            estimated_peak = base_bytes + fit_bytes + base_bytes + svd_workspace
            notes.append("sample-scoped reducer fit loads the full source view when writing the transformed output")
        elif alignment_id is not None:
            estimated_peak = base_bytes + fit_bytes + svd_workspace
            notes.append("alignment-scoped reducer fit aggregates from the full source view before PCA")
        else:
            estimated_peak = base_bytes + svd_workspace
    else:
        sketch_rank = min(max(dims + 8, dims + 2), input_dims, fit_rows)
        sketch_bytes = _matrix_bytes(fit_rows, sketch_rank, np.dtype(np.float32).itemsize)
        basis_bytes = _matrix_bytes(fit_rows, sketch_rank, np.dtype(np.float32).itemsize)
        compressed_bytes = _matrix_bytes(sketch_rank, input_dims, np.dtype(np.float32).itemsize)
        transform_batch_rows = streaming_batch_rows(
            total_rows=transform_rows,
            dims=input_dims,
            itemsize=itemsize,
            output_dims=dims,
        )
        transform_batch_bytes = _matrix_bytes(min(transform_batch_rows, transform_rows), input_dims, itemsize)
        reduced_batch_bytes = _matrix_bytes(
            min(transform_batch_rows, transform_rows),
            dims,
            np.dtype(np.float32).itemsize,
        )
        reducer_state_bytes = _matrix_bytes(sketch_rank + 2, input_dims, np.dtype(np.float32).itemsize)
        fit_peak_bytes = sketch_bytes + basis_bytes + compressed_bytes + reducer_state_bytes
        transform_peak_bytes = (transform_batch_bytes * 2) + reduced_batch_bytes
        estimated_peak = max(fit_peak_bytes, transform_peak_bytes)
        notes.append("randomized-SVD PCA keeps a low-rank sketch in memory and transforms outputs in bounded batches")
        if sample_id is not None:
            estimated_peak += fit_bytes
            notes.append("sample-scoped reducer keeps the sampled fit scope resident while transforming the full view")
        if alignment_id is not None:
            notes.append("alignment-scoped reducer still aggregates alignment rows before each randomized-SVD pass")
    return _build_preflight(
        context,
        operation="view reduce",
        algorithm=f"pca_{pca_method}",
        estimated_peak_bytes=estimated_peak,
        rows=fit_rows,
        dims=input_dims,
        dtype=dtype,
        notes=notes,
    )


def evaluate_materialize_preflight(
    context: WorkspaceContext,
    *,
    view_id: str,
) -> MemoryPreflight:
    view = context.require_source_view(view_id)
    source = context.require_source(view.source)
    resolved = resolve_source(view.source, source, workspace_dir=context.workspace_dir)
    output_dtype = np.dtype(context.analysis_dtype)
    output_itemsize = int(output_dtype.itemsize)
    if view.vector.kind == "bundle_matrix":
        rows_path, matrix_path, _ = require_matrix_bundle_paths(resolved)
        rows, dims, source_dtype, source_itemsize = _artifact_matrix_metadata(matrix_path)
        source_bytes = _matrix_bytes(rows, dims, source_itemsize)
        output_bytes = _matrix_bytes(rows, dims, output_itemsize)
        schema_columns = inspect_source_schema(resolved)["columns"]
        rows_bytes = max(
            rows_path.stat().st_size,
            _row_count(rows_path) * max(len(schema_columns), 1) * 8,
        )
        estimated_peak = rows_bytes + source_bytes + (output_bytes if source_dtype != output_dtype.name else 0)
        notes = [
            "view materialize copies an existing matrix-bundle source into a workspace-owned artifact",
            "bundle-backed materialization eagerly loads rows.parquet before copying the matrix payload",
        ]
        if source_dtype != output_dtype.name:
            notes.append("the source matrix dtype differs from analysis_dtype, so materialization also casts values")
        return _build_preflight(
            context,
            operation="view materialize",
            algorithm="view_materialize_bundle_copy",
            estimated_peak_bytes=estimated_peak,
            rows=rows,
            dims=dims,
            dtype=output_dtype.name,
            notes=notes,
        )

    try:
        source_schema = inspect_source_schema(resolved)
        rows = int(source_schema["row_count"])
        dims = _source_vector_dims(resolved, vector_column=view.vector.name)
    except Exception as exc:
        missing_columns = missing_overlay_merge_columns(exc)
        if view.vector.name in missing_columns:
            raise ContractViolationError(
                f"view {view_id} vector column is missing from source {view.source}: {view.vector.name}"
            ) from exc
        raise
    batch_rows = min(rows, _VIEW_MATERIALIZE_BATCH_ROWS)
    batch_bytes = _matrix_bytes(batch_rows, dims, output_itemsize)
    output_bytes = _matrix_bytes(rows, dims, output_itemsize)
    resident_output_bytes = int(output_bytes * _VIEW_MATERIALIZE_RESIDENT_OUTPUT_FACTOR)
    estimated_peak = max(resident_output_bytes + (batch_bytes * 2), 64 * 1024**2)
    notes = [
        "view materialize streams source vectors in fixed-size batches into a disk-backed memmap",
        f"batch size for the current contract is {_VIEW_MATERIALIZE_BATCH_ROWS} rows",
        "disk-backed output pages can still become resident while the memmap is written",
    ]
    if rows > batch_rows:
        notes.append("estimated resident memory includes a conservative multiple of the output matrix size")
    return _build_preflight(
        context,
        operation="view materialize",
        algorithm="view_materialize_streaming_source",
        estimated_peak_bytes=estimated_peak,
        rows=rows,
        dims=dims,
        dtype=output_dtype.name,
        notes=notes,
    )


def evaluate_projection_preflight(
    context: WorkspaceContext,
    *,
    view_id: str,
    sample_id: str,
) -> MemoryPreflight:
    rows, dims, dtype, itemsize = _view_metadata(context, view_id=view_id)
    sample_manifest = context.read_manifest(context.output_root / "samples" / sample_id / "manifest.json")
    sample_params = sample_manifest.get("params", {}) if isinstance(sample_manifest.get("params"), dict) else {}
    sample_strategy = str(sample_params.get("strategy", "unknown"))
    sample_rows = _row_count(context.output_root / "samples" / sample_id / "rows.parquet")
    n_neighbors = max(2, min(15, sample_rows - 1))
    base_bytes = _matrix_bytes(rows, dims, itemsize)
    subset_bytes = _matrix_bytes(sample_rows, dims, itemsize)
    graph_bytes = sample_rows * n_neighbors * (8 + 4) * 2
    coords_bytes = sample_rows * 2 * np.dtype(np.float32).itemsize
    if sample_strategy == "all" and sample_rows == rows:
        working_set_bytes = max(int(base_bytes * 0.75), 1024**3)
        estimated_peak = base_bytes + working_set_bytes + graph_bytes + coords_bytes
        notes = ["projection fit reuses the full source view directly for strategy=all samples"]
    else:
        estimated_peak = base_bytes + (subset_bytes * 3) + graph_bytes + coords_bytes
        notes = ["projection fit samples from a fully materialized source view"]
    return _build_preflight(
        context,
        operation="projection fit",
        algorithm="umap_projection",
        estimated_peak_bytes=estimated_peak,
        rows=sample_rows,
        dims=dims,
        dtype=dtype,
        notes=notes,
    )


def evaluate_neighbors_preflight(
    context: WorkspaceContext,
    *,
    view_id: str | None,
    reduced_view_id: str | None = None,
    k: int,
    backend: str,
    sample_id: str | None,
    alignment_id: str | None,
) -> MemoryPreflight:
    full_rows, dims, dtype, itemsize, source_kind = _matrix_source_metadata(
        context,
        view_id=view_id,
        reduced_view_id=reduced_view_id,
    )
    scope_rows, notes = _scope_rows(
        context,
        view_id=view_id,
        reduced_view_id=reduced_view_id,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    matrix_bytes = _scoped_matrix_bytes(
        full_rows=full_rows,
        scope_rows=scope_rows,
        dims=dims,
        itemsize=itemsize,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    resolved_backend = _resolved_neighbor_backend(backend)
    result_bytes = scope_rows * k * (np.dtype(np.int64).itemsize + np.dtype(np.float32).itemsize)
    if resolved_backend == "approximate":
        search_bytes = _matrix_bytes(scope_rows, dims, itemsize) + (scope_rows * max(k + 1, 5) * 24)
        estimated_peak = matrix_bytes + search_bytes + result_bytes
        notes.append("approximate neighbors build a separate NN-descent index")
    else:
        pairwise_bytes = scope_rows * scope_rows * np.dtype(np.float32).itemsize
        normalized_bytes = _matrix_bytes(scope_rows, dims, itemsize)
        estimated_peak = matrix_bytes + pairwise_bytes + normalized_bytes + result_bytes
        notes.append("exact neighbors materialize a dense pairwise distance matrix")
    return _build_preflight(
        context,
        operation="neighbors fit",
        algorithm=f"neighbors_{resolved_backend}",
        estimated_peak_bytes=estimated_peak,
        rows=scope_rows,
        dims=dims,
        dtype=dtype,
        notes=[*notes, *([f"source artifact kind: {source_kind}"] if reduced_view_id is not None else [])],
    )


def evaluate_cluster_preflight(
    context: WorkspaceContext,
    *,
    view_id: str | None,
    reduced_view_id: str | None = None,
    method: str,
    n_clusters: int | None,
    k: int,
    sample_id: str | None,
    alignment_id: str | None,
    neighbor_set_id: str | None,
) -> MemoryPreflight:
    full_rows, dims, dtype, itemsize, source_kind = _matrix_source_metadata(
        context,
        view_id=view_id,
        reduced_view_id=reduced_view_id,
    )
    scope_rows, notes = _scope_rows(
        context,
        view_id=view_id,
        reduced_view_id=reduced_view_id,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    matrix_bytes = _scoped_matrix_bytes(
        full_rows=full_rows,
        scope_rows=scope_rows,
        dims=dims,
        itemsize=itemsize,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    label_bytes = scope_rows * np.dtype(np.int64).itemsize
    if method == "kmeans":
        cluster_count = max(int(n_clusters or 0), 1)
        estimated_peak = matrix_bytes + (scope_rows * cluster_count * np.dtype(np.float32).itemsize * 2)
        estimated_peak += cluster_count * dims * np.dtype(np.float32).itemsize * 2
        estimated_peak += label_bytes
        notes.append("kmeans allocates a dense rows-by-clusters distance workspace")
        algorithm = "cluster_kmeans"
    else:
        if neighbor_set_id is not None:
            neighbor_bytes = scope_rows * max(1, min(k, scope_rows - 1)) * np.dtype(np.int64).itemsize
            graph_bytes = scope_rows * max(1, min(k, scope_rows - 1)) * 16
            estimated_peak = matrix_bytes + neighbor_bytes + graph_bytes + label_bytes
            notes.append("leiden reuses a declared neighbor set for graph construction")
            algorithm = "cluster_leiden_neighbor_graph"
        else:
            pairwise_bytes = scope_rows * scope_rows * np.dtype(np.float32).itemsize
            order_bytes = scope_rows * max(1, min(k, scope_rows - 1)) * np.dtype(np.int64).itemsize
            estimated_peak = matrix_bytes + pairwise_bytes + order_bytes + label_bytes
            notes.append("leiden without a neighbor set computes a dense pairwise graph candidate matrix")
            algorithm = "cluster_leiden_exact"
    return _build_preflight(
        context,
        operation="cluster fit",
        algorithm=algorithm,
        estimated_peak_bytes=estimated_peak,
        rows=scope_rows,
        dims=dims,
        dtype=dtype,
        notes=[*notes, *([f"source artifact kind: {source_kind}"] if reduced_view_id is not None else [])],
    )


def evaluate_export_preflight(
    context: WorkspaceContext,
    *,
    export_id: str,
    export_kind: Literal["matrix", "table"],
) -> MemoryPreflight:
    export = context.require_export(export_id)
    basis_rows = _resolve_rows_artifact_count(context, export.row_basis)
    output_itemsize = np.dtype(np.float32).itemsize
    total_features = 0
    loaded_block_bytes = 0
    notes: list[str] = []
    for block in export.blocks:
        if isinstance(block, ReducedViewExportBlockConfig):
            source_rows, source_dims, _, _ = _artifact_matrix_metadata(
                context.output_root / "reduced_views" / block.source / "matrix.npy"
            )
            block_rows = _aligned_row_count(context, alignment_id=block.alignment) if block.alignment else source_rows
            block_bytes = _matrix_bytes(block_rows, source_dims, output_itemsize)
            if block.alignment is not None:
                block_bytes *= 2
                notes.append(f"export block {block.block_id} aggregates reduced-view rows onto an alignment basis")
            loaded_block_bytes += block_bytes
            total_features += source_dims
            continue
        if isinstance(block, TableColumnsExportBlockConfig):
            source_rows = _resolve_rows_artifact_count(context, block.source)
            block_rows = _aligned_row_count(context, alignment_id=block.alignment) if block.alignment else source_rows
            block_dims = len(block.columns)
            block_bytes = _matrix_bytes(block_rows, block_dims, output_itemsize)
            if block.alignment is not None:
                block_bytes *= 2
                notes.append(f"export block {block.block_id} aggregates table columns onto an alignment basis")
            loaded_block_bytes += block_bytes
            total_features += block_dims
    final_output_bytes = _matrix_bytes(basis_rows, total_features, output_itemsize)
    if export_kind == "table":
        estimated_peak = loaded_block_bytes + (final_output_bytes * 2)
        notes.append("table export materializes column lists before writing Arrow arrays")
        algorithm = "export_table_aligned_bundle"
    else:
        estimated_peak = loaded_block_bytes + final_output_bytes
        algorithm = "export_matrix_aligned_bundle"
    return _build_preflight(
        context,
        operation=f"export {export_kind}",
        algorithm=algorithm,
        estimated_peak_bytes=estimated_peak,
        rows=basis_rows,
        dims=total_features,
        dtype="float32",
        notes=notes,
    )


def apply_memory_preflight(
    preflight: MemoryPreflight,
    *,
    allow_memory_overage: bool,
) -> tuple[str, list[str]]:
    if preflight.state == "ok":
        return "ok", []
    message = preflight.message()
    if preflight.state == "blocked":
        if not allow_memory_overage:
            raise MemoryPreflightError(message)
        return "attention", [f"{message}; proceeding because --allow-memory-overage was set"]
    return "ok", [message]


def approximate_backend_warning(*, requested_backend: str, resolved_backend: str) -> tuple[str, list[str]]:
    if resolved_backend != "approximate":
        return "ok", []
    message = (
        "neighbors fit is running in explicit degraded mode with backend=approximate; "
        "the chosen method is recorded in the manifest and command result"
    )
    if requested_backend == "auto":
        message = (
            "neighbors fit resolved backend=auto to approximate; "
            "the chosen degraded mode is recorded in the manifest and command result"
        )
    return "attention", [message]


def merge_attention_status(statuses: list[str], warnings: list[str]) -> str:
    del warnings
    return merge_statuses(*statuses)


def _build_preflight(
    context: WorkspaceContext,
    *,
    operation: str,
    algorithm: str,
    estimated_peak_bytes: int,
    rows: int,
    dims: int,
    dtype: str,
    notes: list[str],
) -> MemoryPreflight:
    system_bytes = system_ram_bytes()
    policy = context.config.defaults.memory_policy
    fraction = estimated_peak_bytes / max(system_bytes, 1)
    if fraction > policy.fail_fraction_of_system_ram and policy.require_override_above_fail:
        state: ArtifactState = "blocked"
    elif fraction > policy.warn_fraction_of_system_ram:
        state = "warning"
    else:
        state = "ok"
    return MemoryPreflight(
        operation=operation,
        algorithm=algorithm,
        estimated_peak_bytes=int(max(estimated_peak_bytes, 0)),
        system_ram_bytes=system_bytes,
        warn_fraction_of_system_ram=policy.warn_fraction_of_system_ram,
        fail_fraction_of_system_ram=policy.fail_fraction_of_system_ram,
        require_override_above_fail=policy.require_override_above_fail,
        state=state,
        rows=rows,
        dims=dims,
        dtype=dtype,
        notes=notes,
    )


def _resolved_neighbor_backend(requested_backend: str) -> str:
    if requested_backend == "auto":
        return "approximate" if approximate_backend_available() else "exact"
    return requested_backend


def _view_metadata(context: WorkspaceContext, *, view_id: str) -> tuple[int, int, str, int]:
    return _artifact_matrix_metadata(context.output_root / "views" / view_id / "matrix.npy")


def _reduced_view_metadata(context: WorkspaceContext, *, reduced_view_id: str) -> tuple[int, int, str, int]:
    return _artifact_matrix_metadata(context.output_root / "reduced_views" / reduced_view_id / "matrix.npy")


def _matrix_source_metadata(
    context: WorkspaceContext,
    *,
    view_id: str | None,
    reduced_view_id: str | None,
) -> tuple[int, int, str, int, str]:
    if (view_id is None) == (reduced_view_id is None):
        raise MemoryPreflightError("memory preflight requires exactly one of view_id or reduced_view_id")
    if reduced_view_id is not None:
        rows, dims, dtype, itemsize = _reduced_view_metadata(context, reduced_view_id=reduced_view_id)
        return rows, dims, dtype, itemsize, "reduced_view"
    rows, dims, dtype, itemsize = _view_metadata(context, view_id=str(view_id))
    return rows, dims, dtype, itemsize, "view"


def _artifact_matrix_metadata(path: Path) -> tuple[int, int, str, int]:
    if not path.is_file():
        raise MissingArtifactError(f"matrix artifact not found for memory preflight: {path}")
    matrix = np.asarray(read_matrix(path))
    if matrix.ndim != 2:
        raise MemoryPreflightError(f"expected 2D matrix for memory preflight: {path}")
    dtype = str(matrix.dtype)
    return int(matrix.shape[0]), int(matrix.shape[1]), dtype, int(matrix.dtype.itemsize)


def _source_vector_dims(resolved, *, vector_column: str) -> int:
    for batch in iter_records_batches(resolved, columns=[vector_column], batch_size=1):
        column = batch.column(vector_column)
        if pa.types.is_fixed_size_list(column.type):
            return int(column.type.list_size)
        for scalar in column:
            value = scalar.as_py() if hasattr(scalar, "as_py") else scalar
            if value is None:
                raise MemoryPreflightError(
                    f"source vector column {vector_column!r} contains null rows and cannot be materialized"
                )
            if hasattr(value, "tolist"):
                value = value.tolist()
            if not isinstance(value, list | tuple):
                raise MemoryPreflightError(
                    f"source vector column {vector_column!r} must contain list-like rows for materialization"
                )
            return len(value)
    raise MemoryPreflightError(f"source vector column {vector_column!r} produced no rows for memory preflight")


def _row_count(path: Path) -> int:
    if not path.is_file():
        raise MissingArtifactError(f"row artifact not found for memory preflight: {path}")
    return int(read_row_count(path))


def _aligned_row_count(context: WorkspaceContext, *, alignment_id: str | None) -> int:
    if alignment_id is None:
        raise MemoryPreflightError("alignment id is required for aligned row count")
    return _row_count(context.output_root / "alignments" / alignment_id / "rows.parquet")


def _scope_rows(
    context: WorkspaceContext,
    *,
    view_id: str | None,
    reduced_view_id: str | None,
    sample_id: str | None,
    alignment_id: str | None,
) -> tuple[int, list[str]]:
    notes: list[str] = []
    if reduced_view_id is not None:
        if sample_id and alignment_id:
            raise MemoryPreflightError("memory preflight accepts at most one scope of sample or alignment")
        if sample_id is not None or alignment_id is not None:
            raise MemoryPreflightError(
                "reduced-view memory preflight does not accept sample or alignment; "
                "reduce the scoped view first and pass --reduced-view"
            )
        rows, _, _, _ = _reduced_view_metadata(context, reduced_view_id=reduced_view_id)
        notes.append("reduced-view source is already materialized at the requested scope")
        return rows, notes
    if sample_id and alignment_id:
        raise MemoryPreflightError("memory preflight accepts at most one scope of sample or alignment")
    if alignment_id is not None:
        return _row_count(context.output_root / "alignments" / alignment_id / "rows.parquet"), notes
    if sample_id is not None:
        return _row_count(context.output_root / "samples" / sample_id / "rows.parquet"), notes
    rows, _, _, _ = _view_metadata(context, view_id=str(view_id))
    return rows, notes


def _scoped_matrix_bytes(
    *,
    full_rows: int,
    scope_rows: int,
    dims: int,
    itemsize: int,
    sample_id: str | None,
    alignment_id: str | None,
) -> int:
    full_bytes = _matrix_bytes(full_rows, dims, itemsize)
    scope_bytes = _matrix_bytes(scope_rows, dims, itemsize)
    if sample_id is not None or alignment_id is not None:
        return full_bytes + scope_bytes
    return full_bytes


def _resolve_rows_artifact_count(context: WorkspaceContext, artifact_id: str) -> int:
    candidates = [
        context.output_root / "alignments" / artifact_id / "rows.parquet",
        context.output_root / "samples" / artifact_id / "rows.parquet",
        context.output_root / "reduced_views" / artifact_id / "rows.parquet",
        context.output_root / "views" / artifact_id / "rows.parquet",
        context.output_root / "scalars" / artifact_id / "table.parquet",
        context.output_root / "distances" / artifact_id / "table.parquet",
    ]
    for path in candidates:
        if path.is_file():
            return _row_count(path)
    raise MissingArtifactError(f"row-basis artifact not found for memory preflight: {artifact_id}")


def _matrix_bytes(rows: int, dims: int, itemsize: int) -> int:
    return int(max(rows, 0) * max(dims, 0) * max(itemsize, 0))
