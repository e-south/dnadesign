"""
LatentDNA source adapter for canonical Infer feature sidecars.

This adapter exposes `_derived/infer/feature_aliases.parquet` joined to
`feature_vectors.parquet`, USR sequence views, mutable view semantics, and the
owning dataset rows. It lets LatentDNA consume the modern sequence-view feature
contract without reintroducing legacy row-overlay embedding columns.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dnadesign.usr import Dataset, sequence_views_path, view_semantics_path

from ..contracts.errors import SourceResolutionError
from ..io.parquet_io import read_schema
from . import usr_source

_FEATURE_ALIAS_RELATIVE_PATH = "_derived/infer/feature_aliases.parquet"
_FEATURE_VECTOR_RELATIVE_PATH = "_derived/infer/feature_vectors.parquet"
_VECTOR_COLUMN = "value"
_VECTOR_CREATED_AT_COLUMN = "feature_vector_created_at"
_ALIAS_CREATED_AT_COLUMN = "feature_alias_created_at"
_BATCH_SIZE = 2048
_ALIAS_SCHEMA = pa.schema(
    [
        pa.field("alias_id", pa.string()),
        pa.field("view_id", pa.string()),
        pa.field("view_name", pa.string()),
        pa.field("sequence_id", pa.string()),
        pa.field("feature_vector_key", pa.string()),
        pa.field("forward_pass_key", pa.string()),
        pa.field("provider", pa.string()),
        pa.field("model_name", pa.string()),
        pa.field("model_revision", pa.string()),
        pa.field("layer_name", pa.string()),
        pa.field("representation_kind", pa.string()),
        pa.field("pooling_operation", pa.string()),
        pa.field("pooling_start_0", pa.int64()),
        pa.field("pooling_end_0", pa.int64()),
        pa.field("orientation", pa.string()),
        pa.field("source_dataset_id", pa.string()),
        pa.field("feature_request_digest", pa.string()),
        pa.field("created_at", pa.string()),
    ]
)
_VECTOR_SCHEMA = pa.schema(
    [
        pa.field("feature_vector_key", pa.string()),
        pa.field(_VECTOR_COLUMN, pa.list_(pa.float64())),
        pa.field("created_at", pa.string()),
    ]
)


def dataset_dir(root: str, dataset: str, *, workspace_dir: Path) -> Path:
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = workspace_dir / root_path
    return (root_path / dataset).resolve()


def feature_aliases_path(root: str, dataset: str, *, workspace_dir: Path) -> Path:
    return dataset_dir(root, dataset, workspace_dir=workspace_dir) / _FEATURE_ALIAS_RELATIVE_PATH


def feature_vectors_path(root: str, dataset: str, *, workspace_dir: Path) -> Path:
    return dataset_dir(root, dataset, workspace_dir=workspace_dir) / _FEATURE_VECTOR_RELATIVE_PATH


def _dataset(root: str, dataset: str, *, workspace_dir: Path) -> Dataset:
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = workspace_dir / root_path
    return Dataset(root_path.resolve(), dataset)


def _require_table(path: Path, *, label: str) -> pa.Table:
    if not path.is_file():
        raise SourceResolutionError(f"{label} not found: {path}")
    return pq.read_table(path)


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)


def _normalize_where(where: Mapping[str, object] | None) -> dict[str, set[str]]:
    if not where:
        return {}
    if set(where) == {"column", "equals"}:
        return {str(where["column"]): {str(where["equals"])}}
    normalized: dict[str, set[str]] = {}
    for column, value in where.items():
        if isinstance(value, Mapping) and set(value) == {"equals"}:
            normalized[str(column)] = {str(value["equals"])}
            continue
        if isinstance(value, list | tuple | set):
            normalized[str(column)] = {str(item) for item in value}
            continue
        normalized[str(column)] = {str(value)}
    return normalized


def _apply_where(table: pa.Table, where: Mapping[str, object] | None) -> pa.Table:
    clauses = _normalize_where(where)
    if not clauses:
        return table
    mask = None
    for column, values in clauses.items():
        if column not in table.column_names:
            raise SourceResolutionError(f"infer feature sidecar where column is missing: {column}")
        column_mask = pc.is_in(table[column], value_set=pa.array(sorted(values), type=pa.string()))
        mask = column_mask if mask is None else pc.and_(mask, column_mask)
    assert mask is not None
    return table.filter(mask)


def _read_alias_table(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
) -> pa.Table:
    path = feature_aliases_path(root, dataset, workspace_dir=workspace_dir)
    if not path.is_file():
        return _apply_where(_empty_table(_ALIAS_SCHEMA), where)
    return _apply_where(pq.read_table(path), where)


@lru_cache(maxsize=16)
def _vector_key_set_for_path(path_text: str) -> set[str]:
    path = Path(path_text)
    if not path.is_file():
        return set()
    table = pq.read_table(path, columns=["feature_vector_key"])
    return {str(value) for value in table.column("feature_vector_key").to_pylist() if value is not None}


def _vector_key_set(root: str, dataset: str, *, workspace_dir: Path) -> set[str]:
    path = feature_vectors_path(root, dataset, workspace_dir=workspace_dir)
    return _vector_key_set_for_path(path.as_posix())


def _assert_vectors_exist(
    aliases: pa.Table,
    *,
    root: str,
    dataset: str,
    workspace_dir: Path,
) -> None:
    wanted = {str(value) for value in aliases.column("feature_vector_key").to_pylist() if value is not None}
    if not wanted:
        return
    present = _vector_key_set(root, dataset, workspace_dir=workspace_dir)
    missing = sorted(wanted - present)
    if missing:
        preview = ", ".join(missing[:5])
        raise SourceResolutionError(
            f"infer feature sidecar aliases reference missing feature vectors in {dataset}: {preview}"
        )


def _schema_field_types(root: str, dataset: str, *, workspace_dir: Path) -> dict[str, pa.DataType]:
    ds = _dataset(root, dataset, workspace_dir=workspace_dir)
    field_types = {field.name: field.type for field in ds.schema()}
    for field in _ALIAS_SCHEMA:
        name = _ALIAS_CREATED_AT_COLUMN if field.name == "created_at" else field.name
        field_types.setdefault(name, field.type)
    for sidecar_path in [feature_aliases_path(root, dataset, workspace_dir=workspace_dir), sequence_views_path(ds)]:
        if not sidecar_path.is_file():
            continue
        for field in read_schema(sidecar_path):
            name = _ALIAS_CREATED_AT_COLUMN if field.name == "created_at" else field.name
            field_types.setdefault(name, field.type)
    semantics_path = view_semantics_path(ds)
    if semantics_path.is_file():
        for field in read_schema(semantics_path):
            field_types.setdefault(field.name, field.type)
    field_types[_VECTOR_COLUMN] = _VECTOR_SCHEMA.field(_VECTOR_COLUMN).type
    field_types[_VECTOR_CREATED_AT_COLUMN] = pa.string()
    return field_types


def _schema_columns(root: str, dataset: str, *, workspace_dir: Path, where: Mapping[str, object] | None) -> list[str]:
    aliases = _read_alias_table(root, dataset, workspace_dir=workspace_dir, where=where)
    _assert_vectors_exist(aliases, root=root, dataset=dataset, workspace_dir=workspace_dir)
    return list(_schema_field_types(root, dataset, workspace_dir=workspace_dir))


def inspect_schema(
    root: str, dataset: str, *, workspace_dir: Path, where: Mapping[str, object] | None
) -> dict[str, Any]:
    aliases = _read_alias_table(root, dataset, workspace_dir=workspace_dir, where=where)
    _assert_vectors_exist(aliases, root=root, dataset=dataset, workspace_dir=workspace_dir)
    return {
        "path": (dataset_dir(root, dataset, workspace_dir=workspace_dir) / "_derived/infer").as_posix(),
        "row_count": aliases.num_rows,
        "columns": _schema_columns(root, dataset, workspace_dir=workspace_dir, where=where),
        "vector_columns": [_VECTOR_COLUMN],
    }


def _renamed_alias_rows(aliases: pa.Table) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in aliases.to_pylist():
        payload = dict(row)
        if "created_at" in payload:
            payload[_ALIAS_CREATED_AT_COLUMN] = payload.pop("created_at")
        rows.append(payload)
    return rows


def _rows_by_key(path: Path, *, key: str, wanted: set[str] | None = None) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, object]] = {}
    for row in pq.read_table(path).to_pylist():
        raw_key = row.get(key)
        if raw_key is None:
            continue
        row_key = str(raw_key)
        if wanted is not None and row_key not in wanted:
            continue
        rows[row_key] = dict(row)
    return rows


def _field_type_from_values(values: list[object]) -> pa.DataType:
    samples: list[object] = []
    for value in values:
        if value is None:
            continue
        samples.append(value)
        if len(samples) >= 256:
            break
    if samples:
        return pa.array(samples).type
    return pa.string()


def _stable_batch_schema(
    columns: list[str],
    metadata: Mapping[str, dict[str, object]],
    *,
    field_types: Mapping[str, pa.DataType] | None = None,
) -> pa.Schema:
    fields: list[pa.Field] = []
    metadata_rows = list(metadata.values())
    for column in columns:
        field_type = (field_types or {}).get(column)
        if field_type is None and column == _VECTOR_COLUMN:
            field_type = pa.list_(pa.float64())
        if field_type is None and column == _VECTOR_CREATED_AT_COLUMN:
            field_type = pa.string()
        if field_type is not None:
            fields.append(pa.field(column, field_type))
            continue
        fields.append(pa.field(column, _field_type_from_values([row.get(column) for row in metadata_rows])))
    return pa.schema(fields)


def _record_rows_by_id(dataset: Dataset, *, columns: list[str]) -> dict[str, dict[str, object]]:
    if "id" not in columns:
        columns = ["id", *columns]
    rows: dict[str, dict[str, object]] = {}
    for batch in dataset.scan(columns=columns, include_overlays=True, include_deleted=False, batch_size=65536):
        for row in batch.to_pylist():
            rows[str(row["id"])] = dict(row)
    return rows


def _metadata_rows(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
    columns: list[str] | None,
) -> dict[str, dict[str, object]]:
    ds = _dataset(root, dataset, workspace_dir=workspace_dir)
    aliases = _read_alias_table(root, dataset, workspace_dir=workspace_dir, where=where)
    _assert_vectors_exist(aliases, root=root, dataset=dataset, workspace_dir=workspace_dir)
    alias_rows = _renamed_alias_rows(aliases)
    wanted_view_ids = {str(row.get("view_id") or "") for row in alias_rows if row.get("view_id")}
    sequence_views = _rows_by_key(sequence_views_path(ds), key="view_id", wanted=wanted_view_ids)
    semantics = _rows_by_key(view_semantics_path(ds), key="view_id", wanted=wanted_view_ids)
    schema_columns = set(_schema_columns(root, dataset, workspace_dir=workspace_dir, where=where))
    requested = set(columns or schema_columns)
    requested.discard(_VECTOR_COLUMN)
    requested.discard(_VECTOR_CREATED_AT_COLUMN)
    sidecar_columns = set(alias_rows[0]) if alias_rows else set()
    if sequence_views:
        sidecar_columns.update(next(iter(sequence_views.values())))
    if semantics:
        sidecar_columns.update(next(iter(semantics.values())))
    record_columns = sorted((requested - sidecar_columns) | {"id"})
    record_columns = [column for column in record_columns if column in set(field.name for field in ds.schema())]
    records = _record_rows_by_id(ds, columns=record_columns)

    rows: dict[str, dict[str, object]] = {}
    for alias_row in alias_rows:
        sequence_id = str(alias_row["sequence_id"])
        record = records.get(sequence_id)
        if record is None:
            raise SourceResolutionError(
                f"infer feature sidecar alias references missing source record {sequence_id!r} in {dataset}"
            )
        view_id = str(alias_row.get("view_id") or "")
        payload = {**record, **(sequence_views.get(view_id) or {}), **(semantics.get(view_id) or {}), **alias_row}
        rows[str(alias_row["feature_vector_key"])] = payload
    return rows


def iter_batches(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
    columns: list[str] | None,
    batch_size: int = _BATCH_SIZE,
):
    selected_columns = list(columns or _schema_columns(root, dataset, workspace_dir=workspace_dir, where=where))
    metadata = _metadata_rows(root, dataset, workspace_dir=workspace_dir, where=where, columns=selected_columns)
    if not metadata:
        return
    output_schema = _stable_batch_schema(
        selected_columns,
        metadata,
        field_types=_schema_field_types(root, dataset, workspace_dir=workspace_dir),
    )
    vector_path = feature_vectors_path(root, dataset, workspace_dir=workspace_dir)
    wanted_keys = set(metadata)
    vector_columns = ["feature_vector_key"]
    if _VECTOR_COLUMN in selected_columns:
        vector_columns.append(_VECTOR_COLUMN)
    if _VECTOR_CREATED_AT_COLUMN in selected_columns:
        vector_columns.append("created_at")
    for batch in pq.ParquetFile(vector_path).iter_batches(columns=vector_columns, batch_size=batch_size):
        batch_rows = []
        for vector_row in batch.to_pylist():
            key = str(vector_row["feature_vector_key"])
            if key not in wanted_keys:
                continue
            row = dict(metadata[key])
            if _VECTOR_COLUMN in selected_columns:
                row[_VECTOR_COLUMN] = vector_row[_VECTOR_COLUMN]
            if _VECTOR_CREATED_AT_COLUMN in selected_columns:
                row[_VECTOR_CREATED_AT_COLUMN] = vector_row.get("created_at")
            batch_rows.append({column: row.get(column) for column in selected_columns})
        if batch_rows:
            yield pa.Table.from_pylist(batch_rows, schema=output_schema).to_batches()[0]


def read_table(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
    columns: list[str] | None,
) -> pa.Table:
    batches = list(iter_batches(root, dataset, workspace_dir=workspace_dir, where=where, columns=columns))
    if batches:
        return pa.Table.from_batches(batches)
    selected_columns = list(columns or _schema_columns(root, dataset, workspace_dir=workspace_dir, where=where))
    schema = _stable_batch_schema(
        selected_columns,
        {},
        field_types=_schema_field_types(root, dataset, workspace_dir=workspace_dir),
    )
    return pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)


def source_provenance(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    columns: list[str] | None,
) -> list[dict[str, object]]:
    entries = usr_source.source_provenance(root, dataset, workspace_dir=workspace_dir, columns=columns)
    ds = _dataset(root, dataset, workspace_dir=workspace_dir)
    for path, role in [
        (feature_aliases_path(root, dataset, workspace_dir=workspace_dir), "infer_feature_aliases"),
        (feature_vectors_path(root, dataset, workspace_dir=workspace_dir), "infer_feature_vectors"),
        (sequence_views_path(ds), "sequence_views"),
        (view_semantics_path(ds), "view_semantics"),
    ]:
        if path.is_file():
            entries.append({"kind": "file", "id": role, "path": path.as_posix(), "role": role})
    return entries
