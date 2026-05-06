"""
Shared Infer sidecar join implementation for vector and scalar adapters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.usr import sequence_views_path, view_semantics_path

from ..contracts.errors import SourceResolutionError
from ..io.parquet_io import read_schema
from . import infer_sidecar_common, usr_source


@dataclass(frozen=True, slots=True)
class InferSidecarJoinContract:
    source_label: str
    alias_role: str
    payload_role: str
    missing_payload_label: str
    alias_relative_path: str
    payload_relative_path: str
    payload_key_column: str
    payload_value_column: str
    payload_created_at_column: str
    alias_created_at_column: str
    alias_schema: pa.Schema
    payload_schema: pa.Schema
    payload_value_field_types: Mapping[str, pa.DataType]
    vector_columns: list[str]


def alias_path(contract: InferSidecarJoinContract, root: str, dataset: str, *, workspace_dir: Path) -> Path:
    return infer_sidecar_common.dataset_dir(root, dataset, workspace_dir=workspace_dir) / contract.alias_relative_path


def payload_path(contract: InferSidecarJoinContract, root: str, dataset: str, *, workspace_dir: Path) -> Path:
    return infer_sidecar_common.dataset_dir(root, dataset, workspace_dir=workspace_dir) / contract.payload_relative_path


def read_alias_table(
    contract: InferSidecarJoinContract,
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
) -> pa.Table:
    path = alias_path(contract, root, dataset, workspace_dir=workspace_dir)
    if not path.is_file():
        table = infer_sidecar_common.empty_table(contract.alias_schema)
    else:
        try:
            table = pq.read_table(path).cast(contract.alias_schema)
        except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
            raise SourceResolutionError(
                f"{contract.source_label} alias table in {dataset} does not match the current Infer sidecar "
                "schema; regenerate canonical sidecars with runtime_fingerprint_key and sequence_case_policy."
            ) from exc
    return infer_sidecar_common.apply_where(table, where, source_label=contract.source_label)


@lru_cache(maxsize=32)
def _payload_key_set_for_path(path_text: str, key_column: str, mtime_ns: int, size: int) -> set[str]:
    del mtime_ns, size
    path = Path(path_text)
    if not path.is_file():
        return set()
    table = pq.read_table(path, columns=[key_column])
    return {str(value) for value in table.column(key_column).to_pylist() if value is not None}


def _payload_key_set(
    contract: InferSidecarJoinContract,
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
) -> set[str]:
    path = payload_path(contract, root, dataset, workspace_dir=workspace_dir)
    if not path.is_file():
        return set()
    stat = path.stat()
    return _payload_key_set_for_path(path.as_posix(), contract.payload_key_column, stat.st_mtime_ns, stat.st_size)


def alias_payload_keys(contract: InferSidecarJoinContract, aliases: pa.Table, *, dataset: str) -> list[str]:
    keys: list[str] = []
    for index, raw_key in enumerate(aliases.column(contract.payload_key_column).to_pylist()):
        if raw_key is None or not str(raw_key).strip():
            raise SourceResolutionError(
                f"{contract.source_label} alias row {index} in {dataset} has no {contract.payload_key_column}"
            )
        keys.append(str(raw_key))
    return keys


def assert_payload_keys_exist(
    contract: InferSidecarJoinContract,
    keys: list[str],
    *,
    root: str,
    dataset: str,
    workspace_dir: Path,
) -> None:
    if not keys:
        return
    present = _payload_key_set(contract, root, dataset, workspace_dir=workspace_dir)
    missing = sorted(set(keys) - present)
    if missing:
        preview = ", ".join(missing[:5])
        raise SourceResolutionError(
            f"{contract.source_label} aliases reference missing "
            f"{contract.missing_payload_label} in {dataset}: {preview}"
        )


def assert_payloads_exist(
    contract: InferSidecarJoinContract,
    aliases: pa.Table,
    *,
    root: str,
    dataset: str,
    workspace_dir: Path,
) -> None:
    assert_payload_keys_exist(
        contract,
        alias_payload_keys(contract, aliases, dataset=dataset),
        root=root,
        dataset=dataset,
        workspace_dir=workspace_dir,
    )


def schema_field_types(
    contract: InferSidecarJoinContract, root: str, dataset: str, *, workspace_dir: Path
) -> dict[str, pa.DataType]:
    ds = infer_sidecar_common.load_dataset(root, dataset, workspace_dir=workspace_dir)
    field_types = {field.name: field.type for field in ds.schema()}
    for field in contract.alias_schema:
        name = contract.alias_created_at_column if field.name == "created_at" else field.name
        field_types.setdefault(name, field.type)
    for sidecar_path in [alias_path(contract, root, dataset, workspace_dir=workspace_dir), sequence_views_path(ds)]:
        if not sidecar_path.is_file():
            continue
        for field in read_schema(sidecar_path):
            name = contract.alias_created_at_column if field.name == "created_at" else field.name
            field_types.setdefault(name, field.type)
    semantics_path = view_semantics_path(ds)
    if semantics_path.is_file():
        for field in read_schema(semantics_path):
            field_types.setdefault(field.name, field.type)
    for column, field_type in contract.payload_value_field_types.items():
        field_types[column] = field_type
    return field_types


def schema_columns(
    contract: InferSidecarJoinContract,
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
    aliases: pa.Table | None = None,
) -> list[str]:
    if aliases is None:
        aliases = read_alias_table(contract, root, dataset, workspace_dir=workspace_dir, where=where)
        assert_payloads_exist(contract, aliases, root=root, dataset=dataset, workspace_dir=workspace_dir)
    return list(schema_field_types(contract, root, dataset, workspace_dir=workspace_dir))


def selected_columns(
    contract: InferSidecarJoinContract,
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
    columns: list[str] | None,
    aliases: pa.Table | None = None,
) -> list[str]:
    available_columns = schema_columns(
        contract,
        root,
        dataset,
        workspace_dir=workspace_dir,
        where=where,
        aliases=aliases,
    )
    if columns is None:
        return available_columns
    available = set(available_columns)
    missing = [column for column in columns if column not in available]
    if missing:
        raise SourceResolutionError(
            f"{contract.source_label} requested columns are unavailable in {dataset}: {', '.join(missing)}"
        )
    return list(columns)


def inspect_schema(
    contract: InferSidecarJoinContract,
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
    aliases: pa.Table | None = None,
) -> dict[str, Any]:
    aliases = (
        aliases
        if aliases is not None
        else read_alias_table(contract, root, dataset, workspace_dir=workspace_dir, where=where)
    )
    payload_keys = alias_payload_keys(contract, aliases, dataset=dataset)
    assert_payload_keys_exist(contract, payload_keys, root=root, dataset=dataset, workspace_dir=workspace_dir)
    return {
        "path": (
            infer_sidecar_common.dataset_dir(root, dataset, workspace_dir=workspace_dir) / "_derived/infer"
        ).as_posix(),
        "row_count": len(payload_keys),
        "columns": schema_columns(contract, root, dataset, workspace_dir=workspace_dir, where=where, aliases=aliases),
        "vector_columns": list(contract.vector_columns),
    }


def metadata_rows(
    contract: InferSidecarJoinContract,
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
    columns: list[str] | None,
    aliases: pa.Table | None = None,
) -> dict[str, list[dict[str, object]]]:
    ds = infer_sidecar_common.load_dataset(root, dataset, workspace_dir=workspace_dir)
    aliases = (
        aliases
        if aliases is not None
        else read_alias_table(contract, root, dataset, workspace_dir=workspace_dir, where=where)
    )
    assert_payloads_exist(contract, aliases, root=root, dataset=dataset, workspace_dir=workspace_dir)
    alias_rows = infer_sidecar_common.renamed_created_at_rows(
        aliases,
        created_at_column=contract.alias_created_at_column,
    )
    if not alias_rows:
        return {}
    wanted_view_ids = {str(row.get("view_id") or "") for row in alias_rows if row.get("view_id")}
    sequence_views = infer_sidecar_common.rows_by_key(sequence_views_path(ds), key="view_id", wanted=wanted_view_ids)
    semantics = infer_sidecar_common.rows_by_key(view_semantics_path(ds), key="view_id", wanted=wanted_view_ids)
    all_schema_columns = set(
        schema_columns(contract, root, dataset, workspace_dir=workspace_dir, where=where, aliases=aliases)
    )
    requested = set(columns or all_schema_columns)
    requested.discard(contract.payload_value_column)
    requested.discard(contract.payload_created_at_column)
    sidecar_columns = set(alias_rows[0])
    if sequence_views:
        sidecar_columns.update(next(iter(sequence_views.values())))
    if semantics:
        sidecar_columns.update(next(iter(semantics.values())))
    record_columns = sorted((requested - sidecar_columns) | {"id"})
    dataset_columns = {field.name for field in ds.schema()}
    record_columns = [column for column in record_columns if column in dataset_columns]
    records = infer_sidecar_common.record_rows_by_id(ds, columns=record_columns)

    rows: dict[str, list[dict[str, object]]] = {}
    for alias_row in alias_rows:
        sequence_id = str(alias_row["sequence_id"])
        record = records.get(sequence_id)
        if record is None:
            raise SourceResolutionError(
                f"{contract.source_label} alias references missing source record {sequence_id!r} in {dataset}"
            )
        view_id = str(alias_row.get("view_id") or "")
        payload = {**record, **(sequence_views.get(view_id) or {}), **(semantics.get(view_id) or {}), **alias_row}
        payload_key = str(alias_row[contract.payload_key_column])
        rows.setdefault(payload_key, []).append(payload)
    return rows


def iter_batches(
    contract: InferSidecarJoinContract,
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
    columns: list[str] | None,
    batch_size: int,
):
    aliases = read_alias_table(contract, root, dataset, workspace_dir=workspace_dir, where=where)
    assert_payloads_exist(contract, aliases, root=root, dataset=dataset, workspace_dir=workspace_dir)
    selected = selected_columns(
        contract,
        root,
        dataset,
        workspace_dir=workspace_dir,
        where=where,
        columns=columns,
        aliases=aliases,
    )
    metadata = metadata_rows(
        contract,
        root,
        dataset,
        workspace_dir=workspace_dir,
        where=where,
        columns=selected,
        aliases=aliases,
    )
    if not metadata:
        return
    metadata_schema_rows = {
        f"{payload_key}#{index}": row for payload_key, rows in metadata.items() for index, row in enumerate(rows)
    }
    output_schema = infer_sidecar_common.stable_batch_schema(
        selected,
        metadata_schema_rows,
        field_types=schema_field_types(contract, root, dataset, workspace_dir=workspace_dir),
        value_field_types=contract.payload_value_field_types,
    )
    payload_table_path = payload_path(contract, root, dataset, workspace_dir=workspace_dir)
    wanted_keys = set(metadata)
    payload_columns = [contract.payload_key_column]
    if contract.payload_value_column in selected:
        payload_columns.append(contract.payload_value_column)
    if contract.payload_created_at_column in selected:
        payload_columns.append("created_at")
    if not payload_table_path.is_file():
        return
    for batch in pq.ParquetFile(payload_table_path).iter_batches(columns=payload_columns, batch_size=batch_size):
        batch_rows = []
        for payload_row in batch.to_pylist():
            key = str(payload_row[contract.payload_key_column])
            if key not in wanted_keys:
                continue
            for metadata_row in metadata[key]:
                row = dict(metadata_row)
                if contract.payload_value_column in selected:
                    row[contract.payload_value_column] = payload_row[contract.payload_value_column]
                if contract.payload_created_at_column in selected:
                    row[contract.payload_created_at_column] = payload_row.get("created_at")
                batch_rows.append({column: row.get(column) for column in selected})
        if batch_rows:
            yield pa.Table.from_pylist(batch_rows, schema=output_schema).to_batches()[0]


def read_table(
    contract: InferSidecarJoinContract,
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
    columns: list[str] | None,
    batch_size: int,
) -> pa.Table:
    batches = list(
        iter_batches(
            contract,
            root,
            dataset,
            workspace_dir=workspace_dir,
            where=where,
            columns=columns,
            batch_size=batch_size,
        )
    )
    if batches:
        return pa.Table.from_batches(batches)
    selected = selected_columns(contract, root, dataset, workspace_dir=workspace_dir, where=where, columns=columns)
    schema = infer_sidecar_common.stable_batch_schema(
        selected,
        {},
        field_types=schema_field_types(contract, root, dataset, workspace_dir=workspace_dir),
        value_field_types=contract.payload_value_field_types,
    )
    return infer_sidecar_common.empty_table(schema)


def source_provenance(
    contract: InferSidecarJoinContract,
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    columns: list[str] | None,
) -> list[dict[str, object]]:
    entries = usr_source.source_provenance(root, dataset, workspace_dir=workspace_dir, columns=columns)
    ds = infer_sidecar_common.load_dataset(root, dataset, workspace_dir=workspace_dir)
    for path, role in [
        (alias_path(contract, root, dataset, workspace_dir=workspace_dir), contract.alias_role),
        (payload_path(contract, root, dataset, workspace_dir=workspace_dir), contract.payload_role),
        (sequence_views_path(ds), "sequence_views"),
        (view_semantics_path(ds), "view_semantics"),
    ]:
        if path.is_file():
            entries.append({"kind": "file", "id": role, "path": path.as_posix(), "role": role})
    return entries
