"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/sequence_views/store.py

Parquet-backed storage helpers for USR sequence-view sidecars.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts import SchemaError
from ..storage.locking import dataset_write_lock
from ..storage.parquet import PARQUET_COMPRESSION
from .models import (
    SEQUENCE_VIEW_SIDECAR_RELATIVE_PATH,
    SequenceViewConflictPolicy,
    SequenceViewRecord,
    SequenceViewSelector,
)

if TYPE_CHECKING:
    from ..dataset import Dataset


def sequence_views_path(dataset: Dataset) -> Path:
    return dataset.dir / SEQUENCE_VIEW_SIDECAR_RELATIVE_PATH


_SEQUENCE_VIEW_SCHEMA = pa.schema(
    [
        pa.field("view_id", pa.string()),
        pa.field("sequence_id", pa.string()),
        pa.field("view_name", pa.string()),
        pa.field("aliases", pa.list_(pa.string())),
        pa.field("product_kind", pa.string()),
        pa.field("context_kind", pa.string()),
        pa.field("orientation", pa.string()),
        pa.field("analysis_only", pa.bool_()),
        pa.field("source_dataset_id", pa.string()),
        pa.field("source_label", pa.string()),
        pa.field("parent_sequence_id", pa.string()),
        pa.field("parent_dataset_id", pa.string()),
        pa.field("derivation_id", pa.string()),
        pa.field("derivation_spec_id", pa.string()),
        pa.field("template_sequence_id", pa.string()),
        pa.field("template_dataset_id", pa.string()),
        pa.field("source_interval_start_0", pa.int64()),
        pa.field("source_interval_end_0", pa.int64()),
        pa.field("anchor_start_0", pa.int64()),
        pa.field("anchor_end_0", pa.int64()),
        pa.field("forward_anchor_start_0", pa.int64()),
        pa.field("forward_anchor_end_0", pa.int64()),
        pa.field("recommended_pooling", pa.string()),
        pa.field("created_at", pa.string()),
        pa.field("created_by", pa.string()),
    ]
)


def _write_sequence_views_atomic(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".parquet.tmp")
    try:
        pq.write_table(table, tmp_path, compression=PARQUET_COMPRESSION)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _sequence_lengths_by_id(dataset: Dataset) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for batch in dataset.scan(columns=["id", "length"], include_overlays=False):
        ids = batch.column(batch.schema.get_field_index("id")).to_pylist()
        seq_lengths = batch.column(batch.schema.get_field_index("length")).to_pylist()
        for row_id, length in zip(ids, seq_lengths, strict=True):
            if row_id is None:
                continue
            lengths[str(row_id)] = int(length)
    return lengths


def _casefolded_aliases(values: list[str] | None) -> set[str]:
    return {alias.casefold() for alias in values or []}


def _lookup_sequence_lengths(
    dataset: Dataset,
    *,
    dataset_id: str | None,
    length_cache: dict[str, dict[str, int]],
) -> dict[str, int]:
    target_dataset_id = dataset.name if dataset_id is None else str(dataset_id).strip()
    if not target_dataset_id:
        target_dataset_id = dataset.name
    cached = length_cache.get(target_dataset_id)
    if cached is not None:
        return cached
    target_dataset = dataset if target_dataset_id == dataset.name else dataset.open(dataset.root, target_dataset_id)
    lengths = _sequence_lengths_by_id(target_dataset)
    length_cache[target_dataset_id] = lengths
    return lengths


def _validate_view_bounds(
    dataset: Dataset,
    record: SequenceViewRecord,
    *,
    sequence_lengths: dict[str, int],
    length_cache: dict[str, dict[str, int]],
) -> None:
    if record.sequence_id not in sequence_lengths:
        raise SchemaError(f"Sequence view references missing sequence_id '{record.sequence_id}'.")
    sequence_length = sequence_lengths[record.sequence_id]
    parent_length: int | None = None
    if record.parent_sequence_id is not None:
        parent_lengths = _lookup_sequence_lengths(
            dataset,
            dataset_id=record.parent_dataset_id,
            length_cache=length_cache,
        )
        parent_length = parent_lengths.get(record.parent_sequence_id)
        if parent_length is None:
            dataset_label = record.parent_dataset_id or dataset.name
            raise SchemaError(
                "Sequence view references missing parent_sequence_id "
                f"'{record.parent_sequence_id}' in dataset '{dataset_label}'."
            )
    for label, start, end, max_length in (
        (
            "source_interval",
            record.source_interval_start_0,
            record.source_interval_end_0,
            parent_length if record.parent_sequence_id is not None else sequence_length,
        ),
        ("anchor", record.anchor_start_0, record.anchor_end_0, sequence_length),
        ("forward_anchor", record.forward_anchor_start_0, record.forward_anchor_end_0, sequence_length),
    ):
        if start is None and end is None:
            continue
        if start is None or end is None:
            raise SchemaError(f"{label} bounds must provide both start and end when present.")
        if end < start:
            raise SchemaError(f"{label} end must be >= start for sequence view '{record.view_id}'.")
        if max_length is not None and end > max_length:
            raise SchemaError(
                f"{label} bounds exceed the sequence length for sequence view '{record.view_id}': "
                f"{start}:{end} > {max_length}."
            )


def _rows_to_table(rows: list[SequenceViewRecord]) -> pa.Table:
    if not rows:
        arrays = [pa.array([], type=field.type) for field in _SEQUENCE_VIEW_SCHEMA]
        return pa.Table.from_arrays(arrays, schema=_SEQUENCE_VIEW_SCHEMA)
    return pa.table(
        {
            field.name: pa.array([row.model_dump().get(field.name) for row in rows], type=field.type)
            for field in _SEQUENCE_VIEW_SCHEMA
        },
        schema=_SEQUENCE_VIEW_SCHEMA,
    )


def _append_alias_payload(record: SequenceViewRecord) -> dict[str, object]:
    payload = record.model_dump()
    for key in ("aliases", "created_at", "created_by"):
        payload.pop(key, None)
    return payload


def load_sequence_views(dataset: Dataset) -> list[SequenceViewRecord]:
    path = sequence_views_path(dataset)
    if not path.exists():
        return []
    table = pq.read_table(path)
    if not table.schema.equals(_SEQUENCE_VIEW_SCHEMA, check_metadata=False):
        table = table.cast(_SEQUENCE_VIEW_SCHEMA)
    rows: list[SequenceViewRecord] = []
    payload = table.to_pylist()
    for raw in payload:
        rows.append(SequenceViewRecord.model_validate(dict(raw)))
    return rows


def select_sequence_views(
    dataset: Dataset,
    *,
    selector: SequenceViewSelector | None = None,
) -> list[SequenceViewRecord]:
    rows = load_sequence_views(dataset)
    if selector is None:
        return rows
    selected: list[SequenceViewRecord] = []
    for row in rows:
        if selector.view_id is not None and row.view_id != selector.view_id:
            continue
        if selector.sequence_id is not None and row.sequence_id != selector.sequence_id:
            continue
        if selector.product_kind is not None and row.product_kind != selector.product_kind:
            continue
        if selector.view_name is not None and row.view_name != selector.view_name:
            continue
        if selector.alias is not None and selector.alias.casefold() not in _casefolded_aliases(row.aliases):
            continue
        selected.append(row)
    return selected


def write_sequence_views(
    dataset: Dataset,
    rows: Iterable[SequenceViewRecord | dict[str, object]],
    *,
    conflict_policy: SequenceViewConflictPolicy = "error",
    actor: dict[str, object] | None = None,
) -> int:
    if conflict_policy not in {"error", "idempotent", "replace", "append_alias"}:
        raise SchemaError(f"Unsupported sequence-view conflict policy '{conflict_policy}'.")
    incoming = [
        row if isinstance(row, SequenceViewRecord) else SequenceViewRecord.model_validate(dict(row)) for row in rows
    ]
    if not incoming:
        return 0

    dataset._require_exists()  # noqa: SLF001
    with dataset_write_lock(dataset.dir):
        sequence_lengths = _sequence_lengths_by_id(dataset)
        length_cache = {dataset.name: sequence_lengths}
        for row in incoming:
            _validate_view_bounds(dataset, row, sequence_lengths=sequence_lengths, length_cache=length_cache)

        existing = {row.view_id: row for row in load_sequence_views(dataset)}
        for row in incoming:
            incoming_aliases = _casefolded_aliases(row.aliases)
            same_alias_conflict = [
                existing_row.view_id
                for existing_row in existing.values()
                if existing_row.view_id != row.view_id
                and incoming_aliases.intersection(_casefolded_aliases(existing_row.aliases))
            ]
            if same_alias_conflict:
                preview = ", ".join(sorted(same_alias_conflict)[:3])
                raise SchemaError(
                    f"Sequence view aliases must remain unique across view_ids. Conflicts with: {preview}."
                )
            current = existing.get(row.view_id)
            if current is None:
                existing[row.view_id] = row
                continue
            if current.semantic_payload() != row.semantic_payload():
                raise SchemaError(f"Sequence view id collision with different semantic content for '{row.view_id}'.")
            if conflict_policy == "error":
                raise SchemaError(f"Sequence view '{row.view_id}' already exists.")
            if conflict_policy == "idempotent":
                current_dump = current.model_dump()
                row_dump = row.model_dump()
                comparable_current = {k: v for k, v in current_dump.items() if k not in {"created_at", "created_by"}}
                comparable_row = {k: v for k, v in row_dump.items() if k not in {"created_at", "created_by"}}
                if comparable_current != comparable_row:
                    raise SchemaError(f"Sequence view '{row.view_id}' already exists with different mutable metadata.")
                continue
            if conflict_policy == "replace":
                existing[row.view_id] = row
                continue
            if _append_alias_payload(current) != _append_alias_payload(row):
                raise SchemaError(
                    f"Sequence view '{row.view_id}' already exists; append_alias can only add human aliases."
                )
            merged = current.model_copy(
                update={
                    "aliases": list(dict.fromkeys([*(current.aliases or []), *(row.aliases or [])])) or None,
                }
            )
            existing[row.view_id] = merged

        sorted_rows = [existing[key] for key in sorted(existing)]
        target_path = sequence_views_path(dataset)
        _write_sequence_views_atomic(target_path, _rows_to_table(sorted_rows))
        if actor is not None:
            dataset._record_event(  # noqa: SLF001
                "write_sequence_views",
                args={
                    "count": len(incoming),
                    "conflict_policy": conflict_policy,
                },
                metrics={"views_written": len(incoming)},
                target_path=target_path,
                actor=actor,
            )
    return len(incoming)
