"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/sources/infer_sidecar_join.py

Shared Infer sidecar join implementation for vector and scalar adapters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
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


@dataclass(frozen=True, slots=True)
class SidecarBatchRequest:
    request_id: str
    where: Mapping[str, object] | None
    columns: list[str] | None


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


def _first_true_index(mask: pa.Array | pa.ChunkedArray) -> int:
    chunks = mask.chunks if isinstance(mask, pa.ChunkedArray) else [mask]
    offset = 0
    for chunk in chunks:
        for index, value in enumerate(chunk.to_pylist()):
            if value:
                return offset + index
        offset += len(chunk)
    return -1


def alias_payload_keys(contract: InferSidecarJoinContract, aliases: pa.Table, *, dataset: str) -> list[str]:
    keys = aliases.column(contract.payload_key_column)
    null_mask = pc.is_null(keys)
    if bool(pc.any(null_mask).as_py()):
        index = _first_true_index(null_mask)
        raise SourceResolutionError(
            f"{contract.source_label} alias row {index} in {dataset} has no {contract.payload_key_column}"
        )
    blank_mask = pc.fill_null(pc.equal(pc.utf8_trim_whitespace(keys), ""), False)
    if bool(pc.any(blank_mask).as_py()):
        index = _first_true_index(blank_mask)
        raise SourceResolutionError(
            f"{contract.source_label} alias row {index} in {dataset} has no {contract.payload_key_column}"
        )
    return keys.to_pylist()


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


def _iter_metadata_only_batches(
    *,
    selected: list[str],
    metadata: dict[str, list[dict[str, object]]],
    output_schema: pa.Schema,
    batch_size: int,
):
    batch_rows: list[dict[str, object]] = []
    for rows in metadata.values():
        for metadata_row in rows:
            batch_rows.append({column: metadata_row.get(column) for column in selected})
            if len(batch_rows) >= batch_size:
                yield pa.Table.from_pylist(batch_rows, schema=output_schema).to_batches()[0]
                batch_rows = []
    if batch_rows:
        yield pa.Table.from_pylist(batch_rows, schema=output_schema).to_batches()[0]


def _record_batch_column(batch: pa.RecordBatch, column_name: str) -> pa.Array:
    index = batch.schema.get_field_index(column_name)
    if index < 0:
        raise SourceResolutionError(f"payload batch is missing required column: {column_name}")
    return batch.column(index)


def _payload_batch_from_matches(
    contract: InferSidecarJoinContract,
    batch: pa.RecordBatch,
    *,
    selected: list[str],
    output_schema: pa.Schema,
    metadata_rows: list[dict[str, object]],
    payload_indices: list[int],
) -> pa.RecordBatch | None:
    if not metadata_rows:
        return None
    index_array = pa.array(payload_indices, type=pa.int64())
    arrays: list[pa.Array] = []
    for column in selected:
        field = output_schema.field(column)
        if column == contract.payload_value_column:
            arrays.append(pc.take(_record_batch_column(batch, contract.payload_value_column), index_array))
            continue
        if column == contract.payload_created_at_column:
            arrays.append(pc.take(_record_batch_column(batch, "created_at"), index_array))
            continue
        arrays.append(pa.array([row.get(column) for row in metadata_rows], type=field.type))
    return pa.record_batch(arrays, schema=output_schema)


def _match_payload_batch_rows(
    contract: InferSidecarJoinContract,
    batch: pa.RecordBatch,
    *,
    metadata: dict[str, list[dict[str, object]]],
) -> tuple[list[dict[str, object]], list[int]]:
    matched_metadata_rows: list[dict[str, object]] = []
    payload_indices: list[int] = []
    for row_index, key_value in enumerate(_record_batch_column(batch, contract.payload_key_column).to_pylist()):
        request_metadata_rows = metadata.get(str(key_value))
        if not request_metadata_rows:
            continue
        for metadata_row in request_metadata_rows:
            matched_metadata_rows.append(metadata_row)
            payload_indices.append(row_index)
    return matched_metadata_rows, payload_indices


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
    needs_payload_scan = contract.payload_value_column in selected or contract.payload_created_at_column in selected
    if not needs_payload_scan:
        yield from _iter_metadata_only_batches(
            selected=selected,
            metadata=metadata,
            output_schema=output_schema,
            batch_size=batch_size,
        )
        return

    payload_table_path = payload_path(contract, root, dataset, workspace_dir=workspace_dir)
    payload_columns = [contract.payload_key_column]
    if contract.payload_value_column in selected:
        payload_columns.append(contract.payload_value_column)
    if contract.payload_created_at_column in selected:
        payload_columns.append("created_at")
    if not payload_table_path.is_file():
        return
    for batch in pq.ParquetFile(payload_table_path).iter_batches(columns=payload_columns, batch_size=batch_size):
        matched_metadata_rows, payload_indices = _match_payload_batch_rows(contract, batch, metadata=metadata)
        if not matched_metadata_rows:
            continue
        batch_payload = _payload_batch_from_matches(
            contract,
            batch,
            selected=selected,
            output_schema=output_schema,
            metadata_rows=matched_metadata_rows,
            payload_indices=payload_indices,
        )
        if batch_payload is not None:
            yield batch_payload


def iter_grouped_batches(
    contract: InferSidecarJoinContract,
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    requests: list[SidecarBatchRequest],
    batch_size: int,
):
    if not requests:
        return

    request_state: dict[str, dict[str, object]] = {}
    union_wanted_keys: set[str] = set()
    for request in requests:
        aliases = read_alias_table(contract, root, dataset, workspace_dir=workspace_dir, where=request.where)
        assert_payloads_exist(contract, aliases, root=root, dataset=dataset, workspace_dir=workspace_dir)
        selected = selected_columns(
            contract,
            root,
            dataset,
            workspace_dir=workspace_dir,
            where=request.where,
            columns=request.columns,
            aliases=aliases,
        )
        metadata = metadata_rows(
            contract,
            root,
            dataset,
            workspace_dir=workspace_dir,
            where=request.where,
            columns=selected,
            aliases=aliases,
        )
        if not metadata:
            continue
        metadata_schema_rows = {
            f"{payload_key}#{index}": row for payload_key, rows in metadata.items() for index, row in enumerate(rows)
        }
        output_schema = infer_sidecar_common.stable_batch_schema(
            selected,
            metadata_schema_rows,
            field_types=schema_field_types(contract, root, dataset, workspace_dir=workspace_dir),
            value_field_types=contract.payload_value_field_types,
        )
        needs_payload_scan = contract.payload_value_column in selected or contract.payload_created_at_column in selected
        if not needs_payload_scan:
            request_state[request.request_id] = {
                "selected": selected,
                "metadata": metadata,
                "schema": output_schema,
            }
            continue
        request_state[request.request_id] = {
            "selected": selected,
            "metadata": metadata,
            "schema": output_schema,
        }
        union_wanted_keys.update(metadata)

    if not request_state:
        return

    metadata_only_batches: dict[str, list[pa.RecordBatch]] = {}
    for request_id, state in request_state.items():
        selected = state["selected"]
        assert isinstance(selected, list)
        if contract.payload_value_column in selected or contract.payload_created_at_column in selected:
            continue
        metadata = state["metadata"]
        output_schema = state["schema"]
        assert isinstance(metadata, dict)
        assert isinstance(output_schema, pa.Schema)
        metadata_only_batches[request_id] = list(
            _iter_metadata_only_batches(
                selected=selected,
                metadata=metadata,
                output_schema=output_schema,
                batch_size=batch_size,
            )
        )
    if metadata_only_batches:
        for request_id, batches in metadata_only_batches.items():
            for batch in batches:
                yield {request_id: batch}

    if not union_wanted_keys:
        return

    payload_table_path = payload_path(contract, root, dataset, workspace_dir=workspace_dir)
    if not payload_table_path.is_file():
        return
    grouped_selected = [state["selected"] for state in request_state.values()]
    needs_value_column = any(
        isinstance(selected, list) and contract.payload_value_column in selected for selected in grouped_selected
    )
    needs_created_at_column = any(
        isinstance(selected, list) and contract.payload_created_at_column in selected for selected in grouped_selected
    )
    payload_columns = [contract.payload_key_column]
    if needs_value_column:
        payload_columns.append(contract.payload_value_column)
    if needs_created_at_column:
        payload_columns.append("created_at")
    for batch in pq.ParquetFile(payload_table_path).iter_batches(columns=payload_columns, batch_size=batch_size):
        grouped_metadata_rows: dict[str, list[dict[str, object]]] = {request_id: [] for request_id in request_state}
        grouped_payload_indices: dict[str, list[int]] = {request_id: [] for request_id in request_state}
        for row_index, key_value in enumerate(_record_batch_column(batch, contract.payload_key_column).to_pylist()):
            key = str(key_value)
            if key not in union_wanted_keys:
                continue
            for request_id, state in request_state.items():
                metadata = state["metadata"]
                assert isinstance(metadata, dict)
                request_metadata_rows = metadata.get(key)
                if not request_metadata_rows:
                    continue
                for metadata_row in request_metadata_rows:
                    grouped_metadata_rows[request_id].append(metadata_row)
                    grouped_payload_indices[request_id].append(row_index)
        batch_payload: dict[str, pa.RecordBatch] = {}
        for request_id, metadata_rows_for_request in grouped_metadata_rows.items():
            if not metadata_rows_for_request:
                continue
            selected = request_state[request_id]["selected"]
            output_schema = request_state[request_id]["schema"]
            assert isinstance(selected, list)
            assert isinstance(output_schema, pa.Schema)
            request_batch = _payload_batch_from_matches(
                contract,
                batch,
                selected=selected,
                output_schema=output_schema,
                metadata_rows=metadata_rows_for_request,
                payload_indices=grouped_payload_indices[request_id],
            )
            if request_batch is not None:
                batch_payload[request_id] = request_batch
        if batch_payload:
            yield batch_payload


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
