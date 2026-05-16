"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/views/materialize.py

View materialization helpers for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts.errors import ContractViolationError
from ..contracts.workspace import InferFeatureSidecarSourceConfig, MetadataLookupDerivationConfig
from ..io.matrix_io import read_matrix, write_matrix
from ..io.parquet_io import read_table, write_table
from ..metadata.derivations import derive_metadata_value
from ..sources import infer_feature_sidecar_source, infer_sidecar_join
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


@dataclass(frozen=True, slots=True)
class _TabularVectorMaterializationPlan:
    view_id: str
    source_id: str
    source: object
    vector_column: str
    artifact_dir: Path
    row_count: int
    row_contract: object
    processing_row_columns: list[str]
    output_row_columns: list[str]
    derived_row_columns: list[str]


@dataclass(slots=True)
class _TabularVectorMaterializationState:
    seen_record_keys: set[tuple[object, ...]]
    seen_subject_keys: set[tuple[object, ...]]
    row_writer: pq.ParquetWriter | None
    matrix: np.memmap | None
    dims: int | None
    write_offset: int
    lookup_cache: dict[tuple[str, str, str, str], tuple[dict[object, object], pa.DataType]]


def _new_tabular_vector_state() -> _TabularVectorMaterializationState:
    return _TabularVectorMaterializationState(
        seen_record_keys=set(),
        seen_subject_keys=set(),
        row_writer=None,
        matrix=None,
        dims=None,
        write_offset=0,
        lookup_cache={},
    )


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
    plan = _TabularVectorMaterializationPlan(
        view_id=view_id,
        source_id=view.source,
        source=source,
        vector_column=vector_column,
        artifact_dir=artifact_dir,
        row_count=row_count,
        row_contract=row_contract,
        processing_row_columns=processing_row_columns,
        output_row_columns=output_row_columns,
        derived_row_columns=derived_row_columns,
    )
    try:
        return _write_tabular_vector_artifact_from_batches(
            context,
            plan=plan,
            batches=iter_records_batches(resolved, columns=columns, batch_size=_MATERIALIZE_BATCH_SIZE),
        )
    except Exception as exc:
        _reraise_missing_vector_column(exc, view_id=view_id, source_id=view.source, vector_column=vector_column)
        raise


def _write_tabular_vector_artifact_from_batches(
    context: WorkspaceContext,
    *,
    plan: _TabularVectorMaterializationPlan,
    batches: Iterable[pa.RecordBatch],
) -> tuple[Path, int, int, str, list[str], list[str]]:
    plan.artifact_dir.mkdir(parents=True, exist_ok=True)
    state = _new_tabular_vector_state()
    try:
        for batch in batches:
            _write_tabular_vector_batch(context, plan=plan, state=state, batch=batch)
    finally:
        _close_tabular_vector_state(state)

    return _finalize_tabular_vector_result(plan, state)


def _write_tabular_vector_batch(
    context: WorkspaceContext,
    *,
    plan: _TabularVectorMaterializationPlan,
    state: _TabularVectorMaterializationState,
    batch: pa.RecordBatch,
) -> None:
    source = plan.source
    _assert_unique_keys(
        state.seen_record_keys,
        batch,
        key_names=[source.record_key],
        label=f"source {plan.source_id} record_key",
    )
    if source.context_key:
        _assert_unique_keys(
            state.seen_subject_keys,
            batch,
            key_names=[source.subject_key, source.context_key],
            label=f"source {plan.source_id} (subject_key, context_key)",
        )
    else:
        _assert_unique_keys(
            state.seen_subject_keys,
            batch,
            key_names=[source.subject_key],
            label=f"source {plan.source_id} subject_key",
        )

    matrix_chunk = _matrix_from_vector_column(
        batch.column(plan.vector_column),
        dtype=context.analysis_dtype,
        label=plan.view_id,
    )
    if state.dims is None:
        state.dims = int(matrix_chunk.shape[1])
        state.matrix = np.lib.format.open_memmap(
            plan.artifact_dir / "matrix.npy",
            mode="w+",
            dtype=np.dtype(context.analysis_dtype),
            shape=(plan.row_count, state.dims),
        )
    elif matrix_chunk.shape[1] != state.dims:
        raise ContractViolationError(
            "view "
            f"{plan.view_id} vector column changed dimensionality within one source: "
            f"{state.dims} vs {matrix_chunk.shape[1]}"
        )

    next_offset = state.write_offset + matrix_chunk.shape[0]
    if state.matrix is None or next_offset > plan.row_count:
        raise ContractViolationError(
            f"view {plan.view_id} materialized more rows than expected: {next_offset} > {plan.row_count}"
        )
    state.matrix[state.write_offset : next_offset] = matrix_chunk

    processing_rows_batch = batch.select(plan.processing_row_columns)
    row_dicts = processing_rows_batch.to_pylist()
    rows_batch = batch.select(plan.output_row_columns)
    for derived_column in [column for column in plan.derived_row_columns if column not in rows_batch.column_names]:
        derivation = (context.config.metadata.derivations or {}).get(derived_column)
        if isinstance(derivation, MetadataLookupDerivationConfig):
            rows_batch = rows_batch.append_column(
                derived_column,
                _lookup_metadata_array(
                    context,
                    row_dicts,
                    column_name=derived_column,
                    derivation=derivation,
                    lookup_cache=state.lookup_cache,
                ),
            )
            continue
        rows_batch = rows_batch.append_column(
            derived_column,
            _derived_metadata_array(context, row_dicts, column_name=derived_column),
        )
    if state.row_writer is None:
        state.row_writer = pq.ParquetWriter(plan.artifact_dir / "rows.parquet", rows_batch.schema)
    state.row_writer.write_batch(rows_batch)
    state.write_offset = next_offset


def _close_tabular_vector_state(state: _TabularVectorMaterializationState) -> None:
    if state.row_writer is not None:
        state.row_writer.close()
        state.row_writer = None
    if state.matrix is not None:
        state.matrix.flush()
        del state.matrix
        state.matrix = None


def _finalize_tabular_vector_result(
    plan: _TabularVectorMaterializationPlan,
    state: _TabularVectorMaterializationState,
) -> tuple[Path, int, int, str, list[str], list[str]]:
    if state.dims is None:
        raise ContractViolationError(f"view {plan.view_id} source produced no rows")
    if state.write_offset != plan.row_count:
        raise ContractViolationError(
            f"view {plan.view_id} materialized {state.write_offset} rows but source schema reported {plan.row_count}"
        )
    return (
        plan.artifact_dir,
        plan.row_count,
        state.dims,
        plan.source.record_key,
        plan.row_contract.materialized_row_columns,
        plan.processing_row_columns,
    )


def materialize_grouped_infer_sidecar_view_artifacts(
    context: WorkspaceContext,
    *,
    view_ids: list[str],
    artifact_dirs: dict[str, Path],
) -> dict[str, tuple[Path, int, int, str, list[str], list[str]]]:
    if not view_ids:
        return {}
    plans: dict[str, _TabularVectorMaterializationPlan] = {}
    requests: list[infer_sidecar_join.SidecarBatchRequest] = []
    root: str | None = None
    dataset: str | None = None
    for view_id in view_ids:
        view = context.require_source_view(view_id)
        source = context.require_source(view.source)
        if not isinstance(source, InferFeatureSidecarSourceConfig):
            raise ContractViolationError(f"view {view_id} source is not an Infer feature sidecar")
        if view.vector.kind != "column":
            raise ContractViolationError(f"view {view_id} vector must be a sidecar value column")
        if root is None:
            root = source.root
            dataset = source.dataset
        elif root != source.root or dataset != source.dataset:
            raise ContractViolationError("grouped Infer sidecar materialization requires one root/dataset")
        resolved = resolve_source(view.source, source, workspace_dir=context.workspace_dir)
        try:
            source_schema = inspect_source_schema(resolved)
        except Exception as exc:
            _reraise_missing_vector_column(exc, view_id=view_id, source_id=view.source, vector_column=view.vector.name)
            raise
        available_columns = set(source_schema["columns"])
        if view.vector.name not in available_columns:
            raise ContractViolationError(
                f"view {view_id} vector column is missing from source {view.source}: {view.vector.name}"
            )
        row_contract = source_backed_view_row_contract(
            context,
            source_id=view.source,
            source=source,
            available_columns=available_columns,
        )
        processing_row_columns = row_contract.processing_row_columns
        output_row_columns = row_contract.output_row_columns
        derived_row_columns = row_contract.derived_row_columns
        columns = [*processing_row_columns, view.vector.name]
        plans[view_id] = _TabularVectorMaterializationPlan(
            view_id=view_id,
            source_id=view.source,
            source=source,
            vector_column=view.vector.name,
            artifact_dir=artifact_dirs[view_id],
            row_count=int(source_schema["row_count"]),
            row_contract=row_contract,
            processing_row_columns=processing_row_columns,
            output_row_columns=output_row_columns,
            derived_row_columns=derived_row_columns,
        )
        requests.append(
            infer_sidecar_join.SidecarBatchRequest(
                request_id=view_id,
                where=source.where,
                columns=columns,
            )
        )
    assert root is not None
    assert dataset is not None

    for plan in plans.values():
        plan.artifact_dir.mkdir(parents=True, exist_ok=True)
    states = {view_id: _new_tabular_vector_state() for view_id in view_ids}
    try:
        for grouped in infer_feature_sidecar_source.iter_grouped_batches(
            root,
            dataset,
            workspace_dir=context.workspace_dir,
            requests=requests,
            batch_size=_MATERIALIZE_BATCH_SIZE,
        ):
            for view_id, batch in grouped.items():
                _write_tabular_vector_batch(context, plan=plans[view_id], state=states[view_id], batch=batch)
    finally:
        for state in states.values():
            _close_tabular_vector_state(state)
    return {view_id: _finalize_tabular_vector_result(plans[view_id], states[view_id]) for view_id in view_ids}


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
