"""
View materialization helpers for latentdna.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts.errors import ContractViolationError
from ..contracts.workspace import PromoterMetadataCohortConfig
from ..io.matrix_io import read_matrix, write_matrix
from ..io.parquet_io import read_table, write_table
from ..sources.resolver import (
    ResolvedSource,
    inspect_source_schema,
    iter_records_batches,
    require_matrix_bundle_paths,
    resolve_source,
)
from ..workspaces.loader import WorkspaceContext

_MATERIALIZE_BATCH_SIZE = 2048
_SIGMA70_PATTERN = re.compile(r"__sigma70_([A-Za-z0-9]+)$")
_CONTROL_LABELS = {"spyp", "sulap", "soxsp", "j23105", "spy_p", "sul_ap", "sox_sp"}


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_regulators(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        values = value
    else:
        values = [value]
    normalized = sorted({_normalize_text(item) for item in values if _normalize_text(item) is not None}, key=str)
    return [str(item) for item in normalized]


def _is_control_row(row: dict[str, object]) -> bool:
    label = (_normalize_text(row.get("usr_label__primary")) or "").lower()
    template_id = (
        _normalize_text(row.get("template_id")) or _normalize_text(row.get("construct__template_id")) or ""
    ).lower()
    plan = _normalize_text(row.get("densegen__plan"))
    if template_id in {"wt", "wildtype", "manual"}:
        return True
    if label in _CONTROL_LABELS:
        return True
    return plan is None


def _design_family(row: dict[str, object]) -> str:
    plan = _normalize_text(row.get("densegen__plan"))
    if plan is not None:
        if plan.startswith("background_only"):
            return "background_only"
        if plan.startswith("ethanol_ciprofloxacin"):
            return "ethanol_ciprofloxacin"
        if plan.startswith("ethanol"):
            return "ethanol"
        if plan.startswith("ciprofloxacin"):
            return "ciprofloxacin"
    if _is_control_row(row):
        return "control"
    return "control"


def _design_regulator_composition(row: dict[str, object]) -> str:
    if _is_control_row(row):
        return "control"
    regulators = _normalized_regulators(row.get("densegen__required_regulators"))
    if regulators:
        if regulators == ["baeR", "lexA"]:
            return "baeR+lexA"
        if regulators == ["cpxR", "lexA"]:
            return "cpxR+lexA"
        if len(regulators) == 1:
            return regulators[0]
        return "+".join(regulators)

    plan = _normalize_text(row.get("densegen__plan")) or ""
    tokens = [token for token in plan.split("__") if token]
    if len(tokens) >= 2 and not tokens[1].startswith("sigma70_"):
        composition = tokens[1].replace("_", "+")
        return composition or "unknown"
    family = _design_family(row)
    if family == "background_only":
        return "background"
    return "unknown"


def _sigma70_variant(row: dict[str, object]) -> str:
    if _is_control_row(row):
        return "control"
    plan = _normalize_text(row.get("densegen__plan")) or ""
    match = _SIGMA70_PATTERN.search(plan)
    if match is None:
        return "control" if _is_control_row(row) else "unknown"
    return match.group(1).lower()


def _campaign_prior(row: dict[str, object]) -> str:
    family = _design_family(row)
    return {
        "background_only": "background",
        "ethanol": "ethanol",
        "ciprofloxacin": "cipro",
        "ethanol_ciprofloxacin": "and",
        "control": "control",
    }.get(family, "control")


def _source_class(row: dict[str, object]) -> str:
    return "manual_or_wildtype" if _is_control_row(row) else "densegen"


def _promoter_metadata_value(row: dict[str, object], *, derive: str) -> object:
    if derive == "design_family":
        return _design_family(row)
    if derive == "design_regulator_composition":
        return _design_regulator_composition(row)
    if derive == "sigma70_variant":
        return _sigma70_variant(row)
    if derive == "campaign_prior":
        return _campaign_prior(row)
    if derive == "is_control":
        return _is_control_row(row)
    if derive == "source_class":
        return _source_class(row)
    raise ContractViolationError(f"unsupported promoter metadata derivation: {derive}")


def _promoter_metadata_columns(
    rows: list[dict[str, object]],
    *,
    configs: list[tuple[str, PromoterMetadataCohortConfig]],
) -> dict[str, pa.Array]:
    arrays: dict[str, pa.Array] = {}
    for cohort_id, config in configs:
        values = [_promoter_metadata_value(row, derive=config.derive) for row in rows]
        arrays[cohort_id] = pa.array(values)
    return arrays


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
) -> tuple[Path, int, int, str, list[str]]:
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
    return artifact_dir, rows.num_rows, int(matrix.shape[1]), source.record_key, list(rows.column_names)


def _materialize_tabular_vector_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    resolved: ResolvedSource,
    source,
    vector_column: str,
    artifact_dir: Path,
) -> tuple[Path, int, int, str, list[str]]:
    view = context.require_source_view(view_id)
    available_columns = set(inspect_source_schema(resolved)["columns"])
    promoter_cohorts = [
        (cohort_id, cohort)
        for cohort_id, cohort in context.config.cohorts.items()
        if isinstance(cohort, PromoterMetadataCohortConfig)
    ]
    if vector_column not in available_columns:
        raise ContractViolationError(
            f"view {view_id} vector column is missing from source {view.source}: {vector_column}"
        )

    promoter_metadata_columns = [
        "densegen__plan",
        "densegen__required_regulators",
        "usr_label__primary",
        "template_id",
        "construct__template_id",
    ]
    metadata_columns = list(
        dict.fromkeys(
            [
                source.record_key,
                source.subject_key,
                *(context.config.metadata.include or []),
                *promoter_metadata_columns,
            ]
        )
    )
    if source.context_key:
        metadata_columns.append(source.context_key)
    columns = [name for name in dict.fromkeys([*metadata_columns, vector_column]) if name in available_columns]
    row_columns = [name for name in columns if name != vector_column]
    derived_row_columns = [cohort_id for cohort_id, _ in promoter_cohorts]
    artifact_dir.mkdir(parents=True, exist_ok=True)

    row_count = int(inspect_source_schema(resolved)["row_count"])
    rows_path = artifact_dir / "rows.parquet"
    matrix_path = artifact_dir / "matrix.npy"
    seen_record_keys: set[tuple[object, ...]] = set()
    seen_subject_keys: set[tuple[object, ...]] = set()
    row_writer: pq.ParquetWriter | None = None
    matrix: np.memmap | None = None
    dims: int | None = None
    write_offset = 0

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

            rows_batch = batch.select(row_columns)
            if promoter_cohorts:
                row_dicts = rows_batch.to_pylist()
                for cohort_id, array in _promoter_metadata_columns(row_dicts, configs=promoter_cohorts).items():
                    rows_batch = rows_batch.append_column(cohort_id, array)
            if row_writer is None:
                row_writer = pq.ParquetWriter(rows_path, rows_batch.schema)
            row_writer.write_batch(rows_batch)
            write_offset = next_offset
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
    return artifact_dir, row_count, dims, source.record_key, [*row_columns, *derived_row_columns]


def materialize_view_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    artifact_dir: Path | None = None,
) -> tuple[Path, int, int, str, list[str]]:
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
