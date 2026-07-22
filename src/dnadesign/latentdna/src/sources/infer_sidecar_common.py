"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/sources/infer_sidecar_common.py

Shared contracts for Infer vector and scalar sidecar source adapters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dnadesign.usr import Dataset

from ..contracts.errors import SourceResolutionError


def dataset_dir(root: str, dataset: str, *, workspace_dir: Path) -> Path:
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = workspace_dir / root_path
    return (root_path / dataset).resolve()


def load_dataset(root: str, dataset: str, *, workspace_dir: Path) -> Dataset:
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = workspace_dir / root_path
    return Dataset(root_path.resolve(), dataset)


def empty_table(schema: pa.Schema) -> pa.Table:
    return pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)


WhereClauses = dict[str, set[object]]


def normalize_where(where: Mapping[str, object] | None) -> WhereClauses:
    if not where:
        return {}
    if set(where) == {"column", "equals"}:
        return {str(where["column"]): {where["equals"]}}
    normalized: WhereClauses = {}
    for column, value in where.items():
        if isinstance(value, Mapping) and set(value) == {"equals"}:
            normalized[str(column)] = {value["equals"]}
            continue
        if isinstance(value, list | tuple | set):
            normalized[str(column)] = set(value)
            continue
        normalized[str(column)] = {value}
    return normalized


def _where_value_set(
    values: set[object],
    *,
    column_type: pa.DataType,
    column: str,
    source_label: str,
) -> pa.Array:
    ordered_values = sorted(values, key=lambda item: str(item))
    if pa.types.is_string(column_type) or pa.types.is_large_string(column_type):
        ordered_values = [str(value) for value in ordered_values]
    try:
        return pa.array(ordered_values, type=column_type)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as exc:
        raise SourceResolutionError(
            f"{source_label} where values for column {column!r} do not match {column_type}: {ordered_values!r}"
        ) from exc


def apply_where(table: pa.Table, where: Mapping[str, object] | None, *, source_label: str) -> pa.Table:
    clauses = normalize_where(where)
    if not clauses:
        return table
    mask = None
    for column, values in clauses.items():
        if column not in table.column_names:
            raise SourceResolutionError(f"{source_label} where column is missing: {column}")
        column_mask = pc.is_in(
            table[column],
            value_set=_where_value_set(
                values,
                column_type=table.schema.field(column).type,
                column=column,
                source_label=source_label,
            ),
        )
        mask = column_mask if mask is None else pc.and_(mask, column_mask)
    assert mask is not None
    return table.filter(mask)


def renamed_created_at_rows(aliases: pa.Table, *, created_at_column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in aliases.to_pylist():
        payload = dict(row)
        if "created_at" in payload:
            payload[created_at_column] = payload.pop("created_at")
        rows.append(payload)
    return rows


def rows_by_key(path: Path, *, key: str, wanted: set[str] | None = None) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, object]] = {}
    for row_index, row in enumerate(pq.read_table(path).to_pylist()):
        raw_key = row.get(key)
        if raw_key is None:
            continue
        row_key = str(raw_key)
        if wanted is not None and row_key not in wanted:
            continue
        if row_key in rows:
            raise SourceResolutionError(f"{path} contains duplicate {key!r} value {row_key!r} at row {row_index}")
        rows[row_key] = dict(row)
    return rows


def field_type_from_values(values: list[object]) -> pa.DataType:
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


def stable_batch_schema(
    columns: list[str],
    metadata: Mapping[str, dict[str, object]],
    *,
    field_types: Mapping[str, pa.DataType] | None = None,
    value_field_types: Mapping[str, pa.DataType] | None = None,
) -> pa.Schema:
    fields: list[pa.Field] = []
    metadata_rows = list(metadata.values())
    for column in columns:
        field_type = (field_types or {}).get(column)
        if field_type is None:
            field_type = (value_field_types or {}).get(column)
        if field_type is not None:
            fields.append(pa.field(column, field_type))
            continue
        fields.append(pa.field(column, field_type_from_values([row.get(column) for row in metadata_rows])))
    return pa.schema(fields)


def record_rows_by_id(dataset: Dataset, *, columns: list[str]) -> dict[str, dict[str, object]]:
    if "id" not in columns:
        columns = ["id", *columns]
    rows: dict[str, dict[str, object]] = {}
    for batch in dataset.scan(columns=columns, include_overlays=True, include_deleted=False, batch_size=65536):
        for row in batch.to_pylist():
            row_id = str(row["id"])
            if row_id in rows:
                raise SourceResolutionError(f"dataset {dataset.name} contains duplicate record id {row_id!r}")
            rows[row_id] = dict(row)
    return rows
