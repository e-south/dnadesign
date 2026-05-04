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
import pyarrow.parquet as pq

from dnadesign.usr import sequence_views_path, view_semantics_path

from ..contracts.errors import SourceResolutionError
from ..io.parquet_io import read_schema
from . import infer_sidecar_common, usr_source

_FEATURE_ALIAS_RELATIVE_PATH = "_derived/infer/feature_aliases.parquet"
_FEATURE_VECTOR_RELATIVE_PATH = "_derived/infer/feature_vectors.parquet"
_VECTOR_COLUMN = "value"
_VECTOR_CREATED_AT_COLUMN = "feature_vector_created_at"
_ALIAS_CREATED_AT_COLUMN = "feature_alias_created_at"
_SOURCE_LABEL = "infer feature sidecar"
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
_VECTOR_VALUE_FIELD_TYPES = {
    _VECTOR_COLUMN: pa.list_(pa.float64()),
    _VECTOR_CREATED_AT_COLUMN: pa.string(),
}


def dataset_dir(root: str, dataset: str, *, workspace_dir: Path) -> Path:
    return infer_sidecar_common.dataset_dir(root, dataset, workspace_dir=workspace_dir)


def feature_aliases_path(root: str, dataset: str, *, workspace_dir: Path) -> Path:
    return dataset_dir(root, dataset, workspace_dir=workspace_dir) / _FEATURE_ALIAS_RELATIVE_PATH


def feature_vectors_path(root: str, dataset: str, *, workspace_dir: Path) -> Path:
    return dataset_dir(root, dataset, workspace_dir=workspace_dir) / _FEATURE_VECTOR_RELATIVE_PATH


def _require_table(path: Path, *, label: str) -> pa.Table:
    if not path.is_file():
        raise SourceResolutionError(f"{label} not found: {path}")
    return pq.read_table(path)


def _read_alias_table(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
) -> pa.Table:
    path = feature_aliases_path(root, dataset, workspace_dir=workspace_dir)
    if not path.is_file():
        table = infer_sidecar_common.empty_table(_ALIAS_SCHEMA)
    else:
        table = pq.read_table(path)
    return infer_sidecar_common.apply_where(table, where, source_label=_SOURCE_LABEL)


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


def _alias_feature_vector_keys(aliases: pa.Table, *, dataset: str) -> list[str]:
    keys: list[str] = []
    for index, raw_key in enumerate(aliases.column("feature_vector_key").to_pylist()):
        if raw_key is None or not str(raw_key).strip():
            raise SourceResolutionError(
                f"infer feature sidecar alias row {index} in {dataset} has no feature_vector_key"
            )
        keys.append(str(raw_key))
    return keys


def _assert_vectors_exist(
    aliases: pa.Table,
    *,
    root: str,
    dataset: str,
    workspace_dir: Path,
) -> None:
    wanted = set(_alias_feature_vector_keys(aliases, dataset=dataset))
    if not wanted:
        return
    present = _vector_key_set(root, dataset, workspace_dir=workspace_dir)
    missing = sorted(wanted - present)
    if missing:
        preview = ", ".join(missing[:5])
        raise SourceResolutionError(
            f"infer feature sidecar aliases reference missing feature vectors in {dataset}: {preview}"
        )


def _assert_feature_vector_keys_exist(
    keys: list[str],
    *,
    root: str,
    dataset: str,
    workspace_dir: Path,
) -> None:
    if not keys:
        return
    present = _vector_key_set(root, dataset, workspace_dir=workspace_dir)
    missing = sorted(set(keys) - present)
    if missing:
        preview = ", ".join(missing[:5])
        raise SourceResolutionError(
            f"infer feature sidecar aliases reference missing feature vectors in {dataset}: {preview}"
        )


def _schema_field_types(root: str, dataset: str, *, workspace_dir: Path) -> dict[str, pa.DataType]:
    ds = infer_sidecar_common.load_dataset(root, dataset, workspace_dir=workspace_dir)
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


def _schema_columns(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
    aliases: pa.Table | None = None,
) -> list[str]:
    if aliases is None:
        aliases = _read_alias_table(root, dataset, workspace_dir=workspace_dir, where=where)
        _assert_vectors_exist(aliases, root=root, dataset=dataset, workspace_dir=workspace_dir)
    return list(_schema_field_types(root, dataset, workspace_dir=workspace_dir))


def inspect_schema(
    root: str, dataset: str, *, workspace_dir: Path, where: Mapping[str, object] | None
) -> dict[str, Any]:
    aliases = _read_alias_table(root, dataset, workspace_dir=workspace_dir, where=where)
    feature_vector_keys = _alias_feature_vector_keys(aliases, dataset=dataset)
    _assert_feature_vector_keys_exist(feature_vector_keys, root=root, dataset=dataset, workspace_dir=workspace_dir)
    return {
        "path": (dataset_dir(root, dataset, workspace_dir=workspace_dir) / "_derived/infer").as_posix(),
        "row_count": len(feature_vector_keys),
        "columns": _schema_columns(root, dataset, workspace_dir=workspace_dir, where=where, aliases=aliases),
        "vector_columns": [_VECTOR_COLUMN],
    }


def _metadata_rows(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
    columns: list[str] | None,
) -> dict[str, list[dict[str, object]]]:
    ds = infer_sidecar_common.load_dataset(root, dataset, workspace_dir=workspace_dir)
    aliases = _read_alias_table(root, dataset, workspace_dir=workspace_dir, where=where)
    _assert_vectors_exist(aliases, root=root, dataset=dataset, workspace_dir=workspace_dir)
    alias_rows = infer_sidecar_common.renamed_created_at_rows(
        aliases,
        created_at_column=_ALIAS_CREATED_AT_COLUMN,
    )
    wanted_view_ids = {str(row.get("view_id") or "") for row in alias_rows if row.get("view_id")}
    sequence_views = infer_sidecar_common.rows_by_key(sequence_views_path(ds), key="view_id", wanted=wanted_view_ids)
    semantics = infer_sidecar_common.rows_by_key(view_semantics_path(ds), key="view_id", wanted=wanted_view_ids)
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
    records = infer_sidecar_common.record_rows_by_id(ds, columns=record_columns)

    rows: dict[str, list[dict[str, object]]] = {}
    for alias_row in alias_rows:
        sequence_id = str(alias_row["sequence_id"])
        record = records.get(sequence_id)
        if record is None:
            raise SourceResolutionError(
                f"infer feature sidecar alias references missing source record {sequence_id!r} in {dataset}"
            )
        view_id = str(alias_row.get("view_id") or "")
        payload = {**record, **(sequence_views.get(view_id) or {}), **(semantics.get(view_id) or {}), **alias_row}
        feature_vector_key = str(alias_row["feature_vector_key"])
        rows.setdefault(feature_vector_key, []).append(payload)
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
    metadata_schema_rows = {
        f"{feature_key}#{index}": row for feature_key, rows in metadata.items() for index, row in enumerate(rows)
    }
    output_schema = infer_sidecar_common.stable_batch_schema(
        selected_columns,
        metadata_schema_rows,
        field_types=_schema_field_types(root, dataset, workspace_dir=workspace_dir),
        value_field_types=_VECTOR_VALUE_FIELD_TYPES,
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
            for metadata_row in metadata[key]:
                row = dict(metadata_row)
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
    schema = infer_sidecar_common.stable_batch_schema(
        selected_columns,
        {},
        field_types=_schema_field_types(root, dataset, workspace_dir=workspace_dir),
        value_field_types=_VECTOR_VALUE_FIELD_TYPES,
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
    ds = infer_sidecar_common.load_dataset(root, dataset, workspace_dir=workspace_dir)
    for path, role in [
        (feature_aliases_path(root, dataset, workspace_dir=workspace_dir), "infer_feature_aliases"),
        (feature_vectors_path(root, dataset, workspace_dir=workspace_dir), "infer_feature_vectors"),
        (sequence_views_path(ds), "sequence_views"),
        (view_semantics_path(ds), "view_semantics"),
    ]:
        if path.is_file():
            entries.append({"kind": "file", "id": role, "path": path.as_posix(), "role": role})
    return entries
