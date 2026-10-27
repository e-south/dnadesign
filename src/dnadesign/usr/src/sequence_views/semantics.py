"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/sequence_views/semantics.py

Mutable semantic addendum sidecar for sequence views.

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
    VIEW_SEMANTICS_SIDECAR_RELATIVE_PATH,
    ViewSemanticsConflictPolicy,
    ViewSemanticsRecord,
)
from .store import load_sequence_views

if TYPE_CHECKING:
    from ..dataset import Dataset


_VIEW_SEMANTICS_SCHEMA = pa.schema(
    [
        pa.field("view_id", pa.string()),
        pa.field("sequence_id", pa.string()),
        pa.field("source_family", pa.string()),
        pa.field("selection_basis", pa.string()),
        pa.field("view_collections", pa.list_(pa.string())),
        pa.field("role_tags", pa.list_(pa.string())),
        pa.field("study_id", pa.string()),
        pa.field("created_at", pa.string()),
        pa.field("created_by", pa.string()),
    ]
)


def view_semantics_path(dataset: Dataset) -> Path:
    return dataset.dir / VIEW_SEMANTICS_SIDECAR_RELATIVE_PATH


def _write_view_semantics_atomic(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".parquet.tmp")
    try:
        pq.write_table(table, tmp_path, compression=PARQUET_COMPRESSION)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _rows_to_table(rows: list[ViewSemanticsRecord]) -> pa.Table:
    if not rows:
        arrays = [pa.array([], type=field.type) for field in _VIEW_SEMANTICS_SCHEMA]
        return pa.Table.from_arrays(arrays, schema=_VIEW_SEMANTICS_SCHEMA)
    return pa.table(
        {
            field.name: pa.array([row.model_dump().get(field.name) for row in rows], type=field.type)
            for field in _VIEW_SEMANTICS_SCHEMA
        },
        schema=_VIEW_SEMANTICS_SCHEMA,
    )


def load_view_semantics(dataset: Dataset) -> list[ViewSemanticsRecord]:
    path = view_semantics_path(dataset)
    if not path.exists():
        return []
    table = pq.read_table(path)
    if not table.schema.equals(_VIEW_SEMANTICS_SCHEMA, check_metadata=False):
        table = table.cast(_VIEW_SEMANTICS_SCHEMA)
    return [ViewSemanticsRecord.model_validate(dict(row)) for row in table.to_pylist()]


def load_view_semantics_index(dataset: Dataset) -> dict[str, dict[str, object]]:
    return {str(row.view_id): row.model_dump() for row in load_view_semantics(dataset)}


def _validate_semantics_against_views(dataset: Dataset, rows: Iterable[ViewSemanticsRecord]) -> None:
    views_by_id = {str(view.view_id): view for view in load_sequence_views(dataset)}
    if not views_by_id:
        raise SchemaError(
            "View-semantics sidecars require _views/sequence_views.parquet to exist before addendum writes."
        )
    for row in rows:
        view = views_by_id.get(row.view_id)
        if view is None:
            raise SchemaError(f"View-semantics row references missing view_id '{row.view_id}'.")
        if view.sequence_id != row.sequence_id:
            raise SchemaError(
                "View-semantics row sequence_id does not match the sequence view: "
                f"view_id={row.view_id} expected={view.sequence_id} observed={row.sequence_id}."
            )


def write_view_semantics(
    dataset: Dataset,
    rows: Iterable[ViewSemanticsRecord | dict[str, object]],
    *,
    conflict_policy: ViewSemanticsConflictPolicy = "error",
    actor: dict[str, object] | None = None,
) -> int:
    if conflict_policy not in {"error", "idempotent", "replace"}:
        raise SchemaError(f"Unsupported view-semantics conflict policy '{conflict_policy}'.")
    incoming = [
        row if isinstance(row, ViewSemanticsRecord) else ViewSemanticsRecord.model_validate(dict(row)) for row in rows
    ]
    if not incoming:
        return 0

    dataset._require_exists()  # noqa: SLF001
    with dataset_write_lock(dataset.dir):
        _validate_semantics_against_views(dataset, incoming)
        existing = {str(row.view_id): row for row in load_view_semantics(dataset)}
        for row in incoming:
            current = existing.get(row.view_id)
            if current is None:
                existing[row.view_id] = row
                continue
            if conflict_policy == "error":
                raise SchemaError(f"View-semantics row '{row.view_id}' already exists.")
            if conflict_policy == "idempotent":
                current_payload = {
                    k: v for k, v in current.model_dump().items() if k not in {"created_at", "created_by"}
                }
                incoming_payload = {k: v for k, v in row.model_dump().items() if k not in {"created_at", "created_by"}}
                if current_payload != incoming_payload:
                    raise SchemaError(
                        f"View-semantics row '{row.view_id}' already exists with different mutable metadata."
                    )
                continue
            existing[row.view_id] = row

        sorted_rows = [existing[key] for key in sorted(existing)]
        target_path = view_semantics_path(dataset)
        _write_view_semantics_atomic(target_path, _rows_to_table(sorted_rows))
        if actor is not None:
            dataset._record_event(  # noqa: SLF001
                "write_view_semantics",
                args={
                    "count": len(incoming),
                    "conflict_policy": conflict_policy,
                },
                metrics={"view_semantics_written": len(incoming)},
                target_path=target_path,
                actor=actor,
            )
    return len(incoming)
