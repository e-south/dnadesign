"""
View materialization helpers for latentdna.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts.errors import ContractViolationError
from ..contracts.workspace import MetadataLookupDerivationConfig
from ..io.matrix_io import read_matrix, write_matrix
from ..io.parquet_io import read_table, write_table
from ..metadata.derivations import derive_metadata_value
from ..sources.resolver import (
    ResolvedSource,
    inspect_source_schema,
    iter_records_batches,
    missing_overlay_merge_columns,
    read_records_table,
    require_matrix_bundle_paths,
    resolve_source,
)
from ..workspaces.loader import WorkspaceContext
from .row_contracts import source_backed_view_row_contract

_MATERIALIZE_BATCH_SIZE = 2048


def _reraise_missing_vector_column(
    exc: Exception,
    *,
    view_id: str,
    source_id: str,
    vector_column: str,
) -> None:
    missing_columns = missing_overlay_merge_columns(exc)
    if vector_column not in missing_columns:
        return
    raise ContractViolationError(
        f"view {view_id} vector column is missing from source {source_id}: {vector_column}"
    ) from exc


def _derived_metadata_value(
    context: WorkspaceContext,
    row: dict[str, object],
    *,
    column_name: str,
) -> object:
    derivation = (context.config.metadata.derivations or {}).get(column_name)
    if derivation is not None:
        if isinstance(derivation, MetadataLookupDerivationConfig):
            raise ContractViolationError(
                f"metadata column {column_name!r} uses a lookup derivation and must be materialized by batch"
            )
        return derive_metadata_value(row, derivation)
    raise ContractViolationError(f"metadata column {column_name!r} is requested but no derivation is configured")


def _metadata_value_arrow_type(value_type: str | None) -> pa.DataType | None:
    if value_type is None or value_type == "infer":
        return None
    if value_type == "string":
        return pa.string()
    if value_type == "int64":
        return pa.int64()
    if value_type == "float64":
        return pa.float64()
    if value_type == "bool":
        return pa.bool_()
    raise ContractViolationError(f"unsupported metadata derivation value_type: {value_type!r}")


def _derived_metadata_array(
    context: WorkspaceContext,
    rows: list[dict[str, object]],
    *,
    column_name: str,
) -> pa.Array:
    derivation = (context.config.metadata.derivations or {}).get(column_name)
    if derivation is None:
        raise ContractViolationError(f"metadata column {column_name!r} is requested but no derivation is configured")
    values = [_derived_metadata_value(context, row, column_name=column_name) for row in rows]
    field_type = _metadata_value_arrow_type(getattr(derivation, "value_type", None))
    return pa.array(values, type=field_type)


def _lookup_key(value: object, *, column_name: str, key_name: str) -> object:
    if value is None:
        raise ContractViolationError(
            f"metadata lookup derivation {column_name!r} found a null lookup key in row column {key_name!r}"
        )
    try:
        hash(value)
    except TypeError as exc:
        raise ContractViolationError(
            f"metadata lookup derivation {column_name!r} requires hashable lookup keys in column {key_name!r}"
        ) from exc
    return value


def _lookup_metadata_mapping(
    context: WorkspaceContext,
    *,
    column_name: str,
    derivation: MetadataLookupDerivationConfig,
) -> tuple[dict[object, object], pa.DataType]:
    source = context.require_source(derivation.source)
    resolved = resolve_source(derivation.source, source, workspace_dir=context.workspace_dir)
    if resolved.records_path is None:
        raise ContractViolationError(
            f"metadata lookup derivation {column_name!r} source {derivation.source!r} does not expose records"
        )
    try:
        table = read_records_table(resolved, columns=[derivation.right_key, derivation.value_column])
    except Exception as exc:
        missing_columns = missing_overlay_merge_columns(exc)
        if missing_columns:
            raise ContractViolationError(
                f"metadata lookup derivation {column_name!r} source {derivation.source!r} "
                f"is missing columns: {missing_columns}"
            ) from exc
        raise
    missing = [name for name in (derivation.right_key, derivation.value_column) if name not in set(table.column_names)]
    if missing:
        raise ContractViolationError(
            f"metadata lookup derivation {column_name!r} source {derivation.source!r} is missing columns: {missing}"
        )
    key_values = table[derivation.right_key].to_pylist()
    value_column = table[derivation.value_column]
    values = value_column.to_pylist()
    mapping: dict[object, object] = {}
    duplicate_keys: list[object] = []
    for key, value in zip(key_values, values, strict=True):
        normalized_key = _lookup_key(column_name=column_name, key_name=derivation.right_key, value=key)
        if normalized_key in mapping:
            duplicate_keys.append(normalized_key)
            continue
        mapping[normalized_key] = value
    if duplicate_keys:
        preview = sorted({str(key) for key in duplicate_keys})[:5]
        raise ContractViolationError(
            f"metadata lookup derivation {column_name!r} source {derivation.source!r} has duplicate "
            f"right_key values: {preview}"
        )
    return mapping, value_column.type


def _lookup_metadata_array(
    context: WorkspaceContext,
    row_dicts: list[dict[str, object]],
    *,
    column_name: str,
    derivation: MetadataLookupDerivationConfig,
    lookup_cache: dict[tuple[str, str, str, str], tuple[dict[object, object], pa.DataType]],
) -> pa.Array:
    cache_key = (derivation.source, derivation.right_key, derivation.value_column, column_name)
    if cache_key not in lookup_cache:
        lookup_cache[cache_key] = _lookup_metadata_mapping(
            context,
            column_name=column_name,
            derivation=derivation,
        )
    mapping, value_type = lookup_cache[cache_key]
    missing_keys: list[object] = []
    values: list[object] = []
    for row in row_dicts:
        left_key = _lookup_key(
            column_name=column_name, key_name=derivation.left_key, value=row.get(derivation.left_key)
        )
        if left_key not in mapping:
            missing_keys.append(left_key)
            values.append(None)
            continue
        values.append(mapping[left_key])
    if missing_keys and derivation.missing_policy == "error":
        preview = sorted({str(key) for key in missing_keys})[:5]
        raise ContractViolationError(
            f"metadata lookup derivation {column_name!r} has missing lookup matches for "
            f"{derivation.left_key!r}: {preview}"
        )
    return pa.array(values, type=value_type)


def _matrix_from_vector_column(column: pa.Array | pa.ChunkedArray, *, dtype: str, label: str) -> np.ndarray:
    if isinstance(column, pa.Array):
        column = pa.chunked_array([column])
    if column.null_count:
        raise ContractViolationError(f"view {label} vector column contains null rows")

    array = column.combine_chunks()
    value_type = getattr(array.type, "value_type", None)
    if value_type is None or not (
        pa.types.is_integer(value_type) or pa.types.is_floating(value_type) or pa.types.is_unsigned_integer(value_type)
    ):
        raise ContractViolationError(f"view {label} vector column must contain numeric list values")

    if pa.types.is_fixed_size_list(array.type):
        dims = array.type.list_size
        values = array.values.to_numpy(zero_copy_only=False)
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
        lengths = np.diff(offsets)
        if lengths.size == 0:
            return np.empty((0, 0), dtype=dtype)
        dims = int(lengths[0])
        if np.any(lengths != dims):
            raise ContractViolationError(f"view {label} vector column must be rectangular, not ragged")
        values = array.values.to_numpy(zero_copy_only=False)[int(offsets[0]) : int(offsets[-1])]
    else:
        raise ContractViolationError(f"view {label} vector column must materialize from an Arrow list array")

    matrix = np.asarray(values, dtype=dtype)
    expected = len(array) * dims
    if matrix.size != expected:
        raise ContractViolationError(
            f"view {label} vector column produced {matrix.size} values for {len(array)} rows x {dims} dims"
        )
    return np.ascontiguousarray(matrix.reshape(len(array), dims))


def _assert_unique_keys(
    seen: set[tuple[object, ...]],
    batch: pa.RecordBatch,
    *,
    key_names: list[str],
    label: str,
) -> None:
    key_columns = [batch.column(name).to_pylist() for name in key_names]
    for key in zip(*key_columns, strict=True):
        if key in seen:
            raise ContractViolationError(f"{label} must be unique for keys {key_names}: duplicate {key!r}")
        seen.add(key)


def _validate_rows_unique(rows: pa.Table, *, source, label: str) -> None:
    key_tables = [
        ([source.record_key], f"source {label} record_key"),
    ]
    if source.context_key:
        key_tables.append(([source.subject_key, source.context_key], f"source {label} (subject_key, context_key)"))
    else:
        key_tables.append(([source.subject_key], f"source {label} subject_key"))

    for key_names, error_label in key_tables:
        seen: set[tuple[object, ...]] = set()
        for row in rows.select(key_names).to_pylist():
            key = tuple(row[name] for name in key_names)
            if key in seen:
                raise ContractViolationError(f"{error_label} must be unique for keys {key_names}: duplicate {key!r}")
            seen.add(key)


def _materialize_matrix_bundle_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    resolved: ResolvedSource,
    source,
    artifact_dir: Path,
) -> tuple[Path, int, int, str, list[str], list[str]]:
    rows_path, matrix_path, _ = require_matrix_bundle_paths(resolved)
    inspect_source_schema(resolved)
    rows = read_table(rows_path)
    required_columns = [source.record_key, source.subject_key]
    if source.context_key is not None:
        required_columns.append(source.context_key)
    missing = [column for column in required_columns if column not in rows.column_names]
    if missing:
        raise ContractViolationError(f"view {view_id} matrix bundle rows are missing required columns: {missing}")
    _validate_rows_unique(rows, source=source, label=view_id)

    matrix = np.asarray(read_matrix(matrix_path), dtype=context.analysis_dtype)
    if matrix.ndim != 2:
        raise ContractViolationError(f"view {view_id} matrix bundle must be 2D")
    if matrix.shape[0] != rows.num_rows:
        raise ContractViolationError(
            f"view {view_id} matrix bundle row count mismatch: matrix has {matrix.shape[0]} rows but "
            f"rows.parquet has {rows.num_rows}"
        )

    write_matrix(artifact_dir / "matrix.npy", np.ascontiguousarray(matrix))
    write_table(rows, artifact_dir / "rows.parquet")
    row_columns = list(rows.column_names)
    return artifact_dir, rows.num_rows, int(matrix.shape[1]), source.record_key, row_columns, row_columns


def _materialize_tabular_vector_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    resolved: ResolvedSource,
    source,
    vector_column: str,
    artifact_dir: Path,
) -> tuple[Path, int, int, str, list[str], list[str]]:
    view = context.require_source_view(view_id)
    try:
        source_schema = inspect_source_schema(resolved)
    except Exception as exc:
        _reraise_missing_vector_column(exc, view_id=view_id, source_id=view.source, vector_column=vector_column)
        raise
    available_columns = set(source_schema["columns"])
    if vector_column not in available_columns:
        raise ContractViolationError(
            f"view {view_id} vector column is missing from source {view.source}: {vector_column}"
        )

    row_contract = source_backed_view_row_contract(
        context,
        source_id=view.source,
        source=source,
        available_columns=available_columns,
    )
    columns = [*row_contract.processing_row_columns, vector_column]
    processing_row_columns = row_contract.processing_row_columns
    output_row_columns = row_contract.output_row_columns
    derived_row_columns = row_contract.derived_row_columns
    artifact_dir.mkdir(parents=True, exist_ok=True)

    row_count = int(source_schema["row_count"])
    rows_path = artifact_dir / "rows.parquet"
    matrix_path = artifact_dir / "matrix.npy"
    seen_record_keys: set[tuple[object, ...]] = set()
    seen_subject_keys: set[tuple[object, ...]] = set()
    row_writer: pq.ParquetWriter | None = None
    matrix: np.memmap | None = None
    dims: int | None = None
    write_offset = 0
    lookup_cache: dict[tuple[str, str, str, str], tuple[dict[object, object], pa.DataType]] = {}

    try:
        for batch in iter_records_batches(resolved, columns=columns, batch_size=_MATERIALIZE_BATCH_SIZE):
            _assert_unique_keys(
                seen_record_keys,
                batch,
                key_names=[source.record_key],
                label=f"source {view.source} record_key",
            )
            if source.context_key:
                _assert_unique_keys(
                    seen_subject_keys,
                    batch,
                    key_names=[source.subject_key, source.context_key],
                    label=f"source {view.source} (subject_key, context_key)",
                )
            else:
                _assert_unique_keys(
                    seen_subject_keys,
                    batch,
                    key_names=[source.subject_key],
                    label=f"source {view.source} subject_key",
                )

            matrix_chunk = _matrix_from_vector_column(
                batch.column(vector_column),
                dtype=context.analysis_dtype,
                label=view_id,
            )
            if dims is None:
                dims = int(matrix_chunk.shape[1])
                matrix = np.lib.format.open_memmap(
                    matrix_path,
                    mode="w+",
                    dtype=np.dtype(context.analysis_dtype),
                    shape=(row_count, dims),
                )
            elif matrix_chunk.shape[1] != dims:
                raise ContractViolationError(
                    "view "
                    f"{view_id} vector column changed dimensionality within one source: "
                    f"{dims} vs {matrix_chunk.shape[1]}"
                )

            next_offset = write_offset + matrix_chunk.shape[0]
            if matrix is None or next_offset > row_count:
                raise ContractViolationError(
                    f"view {view_id} materialized more rows than expected: {next_offset} > {row_count}"
                )
            matrix[write_offset:next_offset] = matrix_chunk

            processing_rows_batch = batch.select(processing_row_columns)
            row_dicts = processing_rows_batch.to_pylist()
            rows_batch = batch.select(output_row_columns)
            for derived_column in [column for column in derived_row_columns if column not in rows_batch.column_names]:
                derivation = (context.config.metadata.derivations or {}).get(derived_column)
                if isinstance(derivation, MetadataLookupDerivationConfig):
                    rows_batch = rows_batch.append_column(
                        derived_column,
                        _lookup_metadata_array(
                            context,
                            row_dicts,
                            column_name=derived_column,
                            derivation=derivation,
                            lookup_cache=lookup_cache,
                        ),
                    )
                    continue
                rows_batch = rows_batch.append_column(
                    derived_column,
                    _derived_metadata_array(context, row_dicts, column_name=derived_column),
                )
            if row_writer is None:
                row_writer = pq.ParquetWriter(rows_path, rows_batch.schema)
            row_writer.write_batch(rows_batch)
            write_offset = next_offset
    except Exception as exc:
        _reraise_missing_vector_column(exc, view_id=view_id, source_id=view.source, vector_column=vector_column)
        raise
    finally:
        if row_writer is not None:
            row_writer.close()
        if matrix is not None:
            matrix.flush()
            del matrix

    if dims is None:
        raise ContractViolationError(f"view {view_id} source produced no rows")
    if write_offset != row_count:
        raise ContractViolationError(
            f"view {view_id} materialized {write_offset} rows but source schema reported {row_count}"
        )
    return (
        artifact_dir,
        row_count,
        dims,
        source.record_key,
        row_contract.materialized_row_columns,
        processing_row_columns,
    )


def materialize_view_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    artifact_dir: Path | None = None,
) -> tuple[Path, int, int, str, list[str], list[str]]:
    view = context.require_source_view(view_id)
    source = context.require_source(view.source)
    resolved = resolve_source(view.source, source, workspace_dir=context.workspace_dir)
    target_dir = artifact_dir or (context.output_root / "views" / view_id)
    if view.vector.kind == "bundle_matrix":
        return _materialize_matrix_bundle_artifact(
            context,
            view_id=view_id,
            resolved=resolved,
            source=source,
            artifact_dir=target_dir,
        )
    if resolved.records_path is None:
        raise ContractViolationError(f"view {view_id} source does not expose a records table")
    return _materialize_tabular_vector_artifact(
        context,
        view_id=view_id,
        resolved=resolved,
        source=source,
        vector_column=view.vector.name,
        artifact_dir=target_dir,
    )
