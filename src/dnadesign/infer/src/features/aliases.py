"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/aliases.py

Sequence-view feature alias persistence for deduplicated Infer execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import socket
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dnadesign.usr import Dataset

from .cache_keys import DNA_SEQUENCE_CASE_POLICY, stable_sha256
from .sidecar_io import (
    atomic_parquet_temp_path,
    read_table_with_schema,
    sidecar_dataset_lock,
    write_table_atomic,
)

FEATURE_ALIAS_RELATIVE_PATH = "_derived/infer/feature_aliases.parquet"
FEATURE_VECTOR_RELATIVE_PATH = "_derived/infer/feature_vectors.parquet"
FEATURE_SCALAR_ALIAS_RELATIVE_PATH = "_derived/infer/feature_scalar_aliases.parquet"
FEATURE_SCALAR_RELATIVE_PATH = "_derived/infer/feature_scalars.parquet"
FEATURE_ALIAS_INVENTORY_COLUMNS = (
    "view_id",
    "feature_vector_key",
    "model_name",
    "layer_name",
    "representation_kind",
    "pooling_operation",
    "orientation",
    "runtime_fingerprint_key",
    "sequence_case_policy",
)
FEATURE_SCALAR_ALIAS_INVENTORY_COLUMNS = (
    "view_id",
    "feature_scalar_key",
    "model_name",
    "scalar_kind",
    "orientation",
    "runtime_fingerprint_key",
    "sequence_case_policy",
)

_FEATURE_ALIAS_SCHEMA = pa.schema(
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
        pa.field("runtime_fingerprint_key", pa.string()),
        pa.field("sequence_case_policy", pa.string()),
        pa.field("created_at", pa.string()),
    ]
)

_FEATURE_VECTOR_SCHEMA = pa.schema(
    [
        pa.field("feature_vector_key", pa.string()),
        pa.field("value", pa.list_(pa.float64())),
        pa.field("created_at", pa.string()),
    ]
)

_FEATURE_SCALAR_ALIAS_SCHEMA = pa.schema(
    [
        pa.field("alias_id", pa.string()),
        pa.field("view_id", pa.string()),
        pa.field("view_name", pa.string()),
        pa.field("sequence_id", pa.string()),
        pa.field("feature_scalar_key", pa.string()),
        pa.field("forward_pass_key", pa.string()),
        pa.field("provider", pa.string()),
        pa.field("model_name", pa.string()),
        pa.field("model_revision", pa.string()),
        pa.field("scalar_kind", pa.string()),
        pa.field("orientation", pa.string()),
        pa.field("source_dataset_id", pa.string()),
        pa.field("feature_request_digest", pa.string()),
        pa.field("runtime_fingerprint_key", pa.string()),
        pa.field("sequence_case_policy", pa.string()),
        pa.field("created_at", pa.string()),
    ]
)

_FEATURE_SCALAR_SCHEMA = pa.schema(
    [
        pa.field("feature_scalar_key", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("created_at", pa.string()),
    ]
)
_VECTOR_BATCH_SIZE = 256
_SCALAR_BATCH_SIZE = 4096
_COMPACT_BATCH_SIZE = 4096
_FEATURE_BUNDLE_PROGRESS_ACTION = "infer_feature_bundle_progress"
_FEATURE_BUNDLE_COMPLETE_ACTION = "infer_feature_bundle_complete"
_FEATURE_BUNDLE_SHARD_COMMIT_ACTION = "infer_feature_bundle_shard_commit"
_RUNTIME_CONTRACT_FIELDS = ("runtime_fingerprint_key", "sequence_case_policy")


def _infer_sidecar_actor(*, default_run_id: str) -> dict[str, object]:
    run_id = str(os.getenv("USR_ACTOR_RUN_ID") or "").strip() or default_run_id
    return {
        "tool": "infer",
        "run_id": run_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }


def feature_alias_path(*, dataset_root: str | Path, dataset_id: str) -> Path:
    return Path(dataset_root) / dataset_id / FEATURE_ALIAS_RELATIVE_PATH


def feature_vector_path(*, dataset_root: str | Path, dataset_id: str) -> Path:
    return Path(dataset_root) / dataset_id / FEATURE_VECTOR_RELATIVE_PATH


def feature_scalar_alias_path(*, dataset_root: str | Path, dataset_id: str) -> Path:
    return Path(dataset_root) / dataset_id / FEATURE_SCALAR_ALIAS_RELATIVE_PATH


def feature_scalar_path(*, dataset_root: str | Path, dataset_id: str) -> Path:
    return Path(dataset_root) / dataset_id / FEATURE_SCALAR_RELATIVE_PATH


def _has_current_alias_contract(row: dict[str, object]) -> bool:
    return (
        bool(str(row.get("runtime_fingerprint_key") or "").strip())
        and str(row.get("sequence_case_policy") or "").strip() == DNA_SEQUENCE_CASE_POLICY
    )


def _has_expected_alias_contract(row: dict[str, object], *, runtime_fingerprint_key: str) -> bool:
    return (
        str(row.get("runtime_fingerprint_key") or "").strip() == str(runtime_fingerprint_key).strip()
        and str(row.get("sequence_case_policy") or "").strip() == DNA_SEQUENCE_CASE_POLICY
    )


def _assert_current_alias_contract(row: dict[str, object], *, table_name: str) -> None:
    missing = [field for field in _RUNTIME_CONTRACT_FIELDS if not str(row.get(field) or "").strip()]
    if missing:
        raise ValueError(
            f"{table_name} row '{row.get('alias_id')}' is missing current runtime contract field(s): "
            + ", ".join(missing)
        )
    case_policy = str(row.get("sequence_case_policy") or "").strip()
    if case_policy != DNA_SEQUENCE_CASE_POLICY:
        raise ValueError(
            f"{table_name} row '{row.get('alias_id')}' has sequence_case_policy={case_policy!r}; "
            f"expected {DNA_SEQUENCE_CASE_POLICY!r}."
        )


def compute_feature_alias_id(
    *,
    view_id: str | None,
    sequence_id: str,
    feature_vector_key: str,
    representation_kind: str,
) -> str:
    return f"alias_{
        stable_sha256(
            {
                'view_id': view_id,
                'sequence_id': sequence_id,
                'feature_vector_key': feature_vector_key,
                'representation_kind': representation_kind,
            }
        )[:24]
    }"


def compute_feature_scalar_key(
    *,
    forward_pass_key: str,
    scalar_kind: str,
    dtype_or_storage_format: str = "float64",
) -> str:
    return f"scalar_{
        stable_sha256(
            {
                'forward_pass_key': forward_pass_key,
                'scalar_kind': scalar_kind,
                'dtype_or_storage_format': dtype_or_storage_format,
            }
        )[:24]
    }"


def compute_feature_scalar_alias_id(
    *,
    view_id: str | None,
    sequence_id: str,
    feature_scalar_key: str,
    scalar_kind: str,
) -> str:
    return f"scalar_alias_{
        stable_sha256(
            {
                'view_id': view_id,
                'sequence_id': sequence_id,
                'feature_scalar_key': feature_scalar_key,
                'scalar_kind': scalar_kind,
            }
        )[:24]
    }"


def persist_feature_alias_rows(rows: Iterable[dict[str, object]]) -> int:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        payload = dict(row)
        dataset_root = str(payload.pop("_dataset_root")).strip()
        dataset_id = str(payload.pop("_dataset_id")).strip()
        grouped[(dataset_root, dataset_id)].append(payload)

    total_written = 0
    for (dataset_root, dataset_id), payload_rows in grouped.items():
        group_written = 0
        path = feature_alias_path(dataset_root=dataset_root, dataset_id=dataset_id)
        with sidecar_dataset_lock(dataset_root=dataset_root, dataset_id=dataset_id):
            existing_rows: list[dict[str, object]] = []
            if path.exists():
                existing_rows = read_table_with_schema(path, schema=_FEATURE_ALIAS_SCHEMA).to_pylist()
            existing_by_id = {str(row["alias_id"]): row for row in existing_rows if _has_current_alias_contract(row)}
            for row in payload_rows:
                _assert_current_alias_contract(row, table_name="feature_aliases")
                alias_id = str(row["alias_id"])
                current = existing_by_id.get(alias_id)
                if current is not None:
                    mutable_fields = {"created_at", "view_name", "feature_request_digest"}
                    comparable_current = {key: value for key, value in current.items() if key not in mutable_fields}
                    comparable_row = {key: value for key, value in row.items() if key not in mutable_fields}
                    if comparable_current != comparable_row:
                        raise ValueError(f"Feature alias collision with different payload for '{alias_id}'.")
                    existing_by_id[alias_id] = {**current, **row, "created_at": current.get("created_at")}
                    continue
                existing_by_id[alias_id] = row
                total_written += 1
                group_written += 1
            table = pa.table(
                {
                    field.name: pa.array([row.get(field.name) for row in existing_by_id.values()], type=field.type)
                    for field in _FEATURE_ALIAS_SCHEMA
                },
                schema=_FEATURE_ALIAS_SCHEMA,
            )
            write_table_atomic(table, path)
        if group_written:
            Dataset(Path(dataset_root), dataset_id).log_event(
                "infer_feature_aliases_write",
                args={"rows_written": group_written},
                artifacts={"path": path.relative_to(Path(dataset_root) / dataset_id).as_posix()},
                target_path=path,
                actor=_infer_sidecar_actor(default_run_id="feature-alias-persistence"),
            )
    return total_written


def persist_feature_scalar_alias_rows(rows: Iterable[dict[str, object]]) -> int:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        payload = dict(row)
        dataset_root = str(payload.pop("_dataset_root")).strip()
        dataset_id = str(payload.pop("_dataset_id")).strip()
        grouped[(dataset_root, dataset_id)].append(payload)

    total_written = 0
    for (dataset_root, dataset_id), payload_rows in grouped.items():
        group_written = 0
        path = feature_scalar_alias_path(dataset_root=dataset_root, dataset_id=dataset_id)
        with sidecar_dataset_lock(dataset_root=dataset_root, dataset_id=dataset_id):
            existing_rows: list[dict[str, object]] = []
            if path.exists():
                existing_rows = read_table_with_schema(path, schema=_FEATURE_SCALAR_ALIAS_SCHEMA).to_pylist()
            existing_by_id = {str(row["alias_id"]): row for row in existing_rows if _has_current_alias_contract(row)}
            for row in payload_rows:
                _assert_current_alias_contract(row, table_name="feature_scalar_aliases")
                alias_id = str(row["alias_id"])
                current = existing_by_id.get(alias_id)
                if current is not None:
                    mutable_fields = {"created_at", "view_name", "feature_request_digest"}
                    comparable_current = {key: value for key, value in current.items() if key not in mutable_fields}
                    comparable_row = {key: value for key, value in row.items() if key not in mutable_fields}
                    if comparable_current != comparable_row:
                        raise ValueError(f"Feature scalar alias collision with different payload for '{alias_id}'.")
                    existing_by_id[alias_id] = {**current, **row, "created_at": current.get("created_at")}
                    continue
                existing_by_id[alias_id] = row
                total_written += 1
                group_written += 1
            table = pa.table(
                {
                    field.name: pa.array([row.get(field.name) for row in existing_by_id.values()], type=field.type)
                    for field in _FEATURE_SCALAR_ALIAS_SCHEMA
                },
                schema=_FEATURE_SCALAR_ALIAS_SCHEMA,
            )
            write_table_atomic(table, path)
        if group_written:
            Dataset(Path(dataset_root), dataset_id).log_event(
                "infer_feature_scalar_aliases_write",
                args={"rows_written": group_written},
                artifacts={"path": path.relative_to(Path(dataset_root) / dataset_id).as_posix()},
                target_path=path,
                actor=_infer_sidecar_actor(default_run_id="feature-scalar-alias-persistence"),
            )
    return total_written


def load_feature_alias_rows(
    *,
    dataset_root: str | Path,
    dataset_id: str,
) -> list[dict[str, object]]:
    path = feature_alias_path(dataset_root=dataset_root, dataset_id=dataset_id)
    if not path.exists():
        return []
    return read_table_with_schema(path, schema=_FEATURE_ALIAS_SCHEMA).to_pylist()


def load_feature_alias_inventory_rows(
    *,
    dataset_root: str | Path,
    dataset_id: str,
) -> list[dict[str, object]]:
    path = feature_alias_path(dataset_root=dataset_root, dataset_id=dataset_id)
    if not path.exists():
        return []
    return _read_table_columns_with_schema(
        path,
        schema=_FEATURE_ALIAS_SCHEMA,
        columns=FEATURE_ALIAS_INVENTORY_COLUMNS,
    ).to_pylist()


def load_feature_alias_ids(
    *,
    dataset_root: str | Path,
    dataset_id: str,
) -> set[str]:
    """Load only alias ids for inventory paths that must not touch vector payloads."""

    path = feature_alias_path(dataset_root=dataset_root, dataset_id=dataset_id)
    if not path.exists():
        return set()
    table = pq.read_table(path, columns=["alias_id"])
    return {str(value) for value in table.column("alias_id").to_pylist() if str(value).strip()}


def load_feature_scalar_alias_ids(
    *,
    dataset_root: str | Path,
    dataset_id: str,
) -> set[str]:
    path = feature_scalar_alias_path(dataset_root=dataset_root, dataset_id=dataset_id)
    if not path.exists():
        return set()
    table = pq.read_table(path, columns=["alias_id"])
    return {str(value) for value in table.column("alias_id").to_pylist() if str(value).strip()}


def load_feature_scalar_alias_rows(
    *,
    dataset_root: str | Path,
    dataset_id: str,
) -> list[dict[str, object]]:
    path = feature_scalar_alias_path(dataset_root=dataset_root, dataset_id=dataset_id)
    if not path.exists():
        return []
    return read_table_with_schema(path, schema=_FEATURE_SCALAR_ALIAS_SCHEMA).to_pylist()


def load_feature_scalar_alias_inventory_rows(
    *,
    dataset_root: str | Path,
    dataset_id: str,
) -> list[dict[str, object]]:
    path = feature_scalar_alias_path(dataset_root=dataset_root, dataset_id=dataset_id)
    if not path.exists():
        return []
    return _read_table_columns_with_schema(
        path,
        schema=_FEATURE_SCALAR_ALIAS_SCHEMA,
        columns=FEATURE_SCALAR_ALIAS_INVENTORY_COLUMNS,
    ).to_pylist()


def load_feature_vector_keys(
    *,
    dataset_root: str | Path,
    dataset_id: str,
    keys: Iterable[str],
) -> set[str]:
    """Return persisted feature-vector keys without reading embedding payloads.

    Feature vectors can be multi-gigabyte parquet files because the `value`
    column stores embedding arrays. Completion/status checks only need key
    membership, so they must avoid materializing that payload column.
    """

    wanted = {str(key) for key in keys if str(key).strip()}
    if not wanted:
        return set()
    path = feature_vector_path(dataset_root=dataset_root, dataset_id=dataset_id)
    if not path.exists():
        return set()
    table = pq.read_table(path, columns=["feature_vector_key"])
    return {str(key) for key in table.column("feature_vector_key").to_pylist() if str(key) in wanted}


def load_feature_scalar_keys(
    *,
    dataset_root: str | Path,
    dataset_id: str,
    keys: Iterable[str],
) -> set[str]:
    wanted = {str(key) for key in keys if str(key).strip()}
    if not wanted:
        return set()
    path = feature_scalar_path(dataset_root=dataset_root, dataset_id=dataset_id)
    if not path.exists():
        return set()
    table = pq.read_table(path, columns=["feature_scalar_key"])
    return {str(key) for key in table.column("feature_scalar_key").to_pylist() if str(key) in wanted}


def _read_table_columns_with_schema(path: Path, *, schema: pa.Schema, columns: tuple[str, ...]) -> pa.Table:
    selected_fields = [schema.field(name) for name in columns]
    file_schema = pq.read_schema(path)
    available_columns = [name for name in columns if name in file_schema.names]
    table = pq.read_table(path, columns=available_columns) if available_columns else None
    num_rows = table.num_rows if table is not None else pq.ParquetFile(path).metadata.num_rows
    projected_columns = {}
    for field in selected_fields:
        if table is not None and field.name in table.column_names:
            projected_columns[field.name] = table.column(field.name).cast(field.type)
        else:
            projected_columns[field.name] = pa.nulls(num_rows, type=field.type)
    return pa.table(projected_columns, schema=pa.schema(selected_fields))


def load_feature_vector_rows(
    *,
    dataset_root: str | Path,
    dataset_id: str,
    keys: Iterable[str],
) -> dict[str, list[float]]:
    wanted = {str(key) for key in keys if str(key).strip()}
    if not wanted:
        return {}
    path = feature_vector_path(dataset_root=dataset_root, dataset_id=dataset_id)
    if not path.exists():
        return {}
    out: dict[str, list[float]] = {}
    for batch in pq.ParquetFile(path).iter_batches(
        columns=["feature_vector_key", "value"],
        batch_size=_VECTOR_BATCH_SIZE,
    ):
        for row in batch.to_pylist():
            key = str(row["feature_vector_key"])
            if key in wanted:
                out[key] = [float(value) for value in row["value"]]
        if len(out) == len(wanted):
            break
    return out


def load_feature_scalar_rows(
    *,
    dataset_root: str | Path,
    dataset_id: str,
    keys: Iterable[str],
) -> dict[str, float]:
    wanted = {str(key) for key in keys if str(key).strip()}
    if not wanted:
        return {}
    path = feature_scalar_path(dataset_root=dataset_root, dataset_id=dataset_id)
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    for batch in pq.ParquetFile(path).iter_batches(
        columns=["feature_scalar_key", "value"],
        batch_size=_SCALAR_BATCH_SIZE,
    ):
        for row in batch.to_pylist():
            key = str(row["feature_scalar_key"])
            if key in wanted:
                out[key] = float(row["value"])
        if len(out) == len(wanted):
            break
    return out


def _alias_orientation(value: object) -> str:
    orientation = str(value or "").strip()
    return orientation or "forward"


def _vector_alias_slot(row: dict[str, object]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("model_name") or ""),
        str(row.get("view_id") or ""),
        str(row.get("representation_kind") or ""),
        str(row.get("layer_name") or ""),
        str(row.get("pooling_operation") or ""),
        _alias_orientation(row.get("orientation")),
    )


def _scalar_alias_slot(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model_name") or ""),
        str(row.get("view_id") or ""),
        str(row.get("scalar_kind") or ""),
        _alias_orientation(row.get("orientation")),
    )


def _expected_runtime_by_slot(
    rows: Iterable[dict[str, object]],
    *,
    table_name: str,
    slot_func,
) -> dict[tuple[object, ...], str]:
    expected: dict[tuple[object, ...], str] = {}
    for raw_row in rows:
        row = dict(raw_row)
        _assert_current_alias_contract(row, table_name=table_name)
        slot = slot_func(row)
        runtime_fingerprint_key = str(row.get("runtime_fingerprint_key") or "").strip()
        current = expected.get(slot)
        if current is not None and current != runtime_fingerprint_key:
            raise ValueError(
                f"{table_name} slot {slot!r} resolves to multiple runtime fingerprints: "
                f"{current!r} and {runtime_fingerprint_key!r}."
            )
        expected[slot] = runtime_fingerprint_key
    return expected


def _group_expected_alias_slots(
    rows: Iterable[dict[str, object]],
    *,
    table_name: str,
    slot_func,
) -> dict[tuple[str, str], dict[tuple[object, ...], str]]:
    grouped_rows: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for raw_row in rows:
        row = dict(raw_row)
        dataset_root = str(row.get("_dataset_root") or "").strip()
        dataset_id = str(row.get("_dataset_id") or "").strip()
        if not dataset_root or not dataset_id:
            raise ValueError(f"{table_name} row '{row.get('alias_id')}' is missing dataset routing fields.")
        grouped_rows[(dataset_root, dataset_id)].append(row)
    return {
        dataset_key: _expected_runtime_by_slot(payload_rows, table_name=table_name, slot_func=slot_func)
        for dataset_key, payload_rows in grouped_rows.items()
    }


def prune_stale_feature_alias_entries(
    *,
    current_vector_alias_rows: Iterable[dict[str, object]],
    current_scalar_alias_rows: Iterable[dict[str, object]],
) -> dict[str, int]:
    """Remove alias entries superseded by current runtime rows for the same feature slots."""

    vector_slots = _group_expected_alias_slots(
        current_vector_alias_rows,
        table_name="feature_aliases",
        slot_func=_vector_alias_slot,
    )
    scalar_slots = _group_expected_alias_slots(
        current_scalar_alias_rows,
        table_name="feature_scalar_aliases",
        slot_func=_scalar_alias_slot,
    )
    dataset_keys = set(vector_slots) | set(scalar_slots)
    removed_vector_rows = 0
    removed_scalar_rows = 0
    for dataset_root, dataset_id in sorted(dataset_keys):
        with sidecar_dataset_lock(dataset_root=dataset_root, dataset_id=dataset_id):
            vector_removed = _prune_alias_table_for_expected_slots(
                feature_alias_path(dataset_root=dataset_root, dataset_id=dataset_id),
                schema=_FEATURE_ALIAS_SCHEMA,
                expected_runtime_by_slot=vector_slots.get((dataset_root, dataset_id), {}),
                slot_func=_vector_alias_slot,
            )
            scalar_removed = _prune_alias_table_for_expected_slots(
                feature_scalar_alias_path(dataset_root=dataset_root, dataset_id=dataset_id),
                schema=_FEATURE_SCALAR_ALIAS_SCHEMA,
                expected_runtime_by_slot=scalar_slots.get((dataset_root, dataset_id), {}),
                slot_func=_scalar_alias_slot,
            )
        removed_vector_rows += vector_removed
        removed_scalar_rows += scalar_removed
        if vector_removed or scalar_removed:
            Dataset(Path(dataset_root), dataset_id).log_event(
                "infer_feature_aliases_prune_stale",
                args={
                    "removed_vector_alias_rows": vector_removed,
                    "removed_scalar_alias_rows": scalar_removed,
                    "sidecar_contract": "sequence_view_feature_bundle",
                    "prune_policy": "same_slot_noncurrent_runtime",
                },
                metrics={
                    "removed_vector_alias_rows": vector_removed,
                    "removed_scalar_alias_rows": scalar_removed,
                },
                artifacts={
                    "feature_aliases": FEATURE_ALIAS_RELATIVE_PATH,
                    "feature_scalar_aliases": FEATURE_SCALAR_ALIAS_RELATIVE_PATH,
                },
                target_path=Path(dataset_root) / dataset_id / "records.parquet",
                actor=_infer_sidecar_actor(default_run_id="feature-alias-prune-stale"),
            )
    return {
        "removed_vector_alias_rows": removed_vector_rows,
        "removed_scalar_alias_rows": removed_scalar_rows,
    }


def _prune_alias_table_for_expected_slots(
    path: Path,
    *,
    schema: pa.Schema,
    expected_runtime_by_slot: dict[tuple[object, ...], str],
    slot_func,
) -> int:
    if not path.exists():
        return 0
    rows = read_table_with_schema(path, schema=schema).to_pylist()
    kept_rows: list[dict[str, object]] = []
    removed = 0
    for row in rows:
        slot = slot_func(row)
        expected_runtime = expected_runtime_by_slot.get(slot)
        if expected_runtime is not None:
            if _has_expected_alias_contract(row, runtime_fingerprint_key=expected_runtime):
                kept_rows.append(row)
            else:
                removed += 1
            continue
        if _has_current_alias_contract(row):
            kept_rows.append(row)
        else:
            removed += 1
    if removed:
        write_table_atomic(_table_from_rows(kept_rows, schema=schema), path)
    return removed


def compact_feature_sidecars_to_current_aliases(
    *,
    dataset_root: str | Path,
    dataset_id: str,
) -> dict[str, int]:
    with sidecar_dataset_lock(dataset_root=dataset_root, dataset_id=dataset_id):
        vector_alias_rows = load_feature_alias_rows(dataset_root=dataset_root, dataset_id=dataset_id)
        scalar_alias_rows = load_feature_scalar_alias_rows(dataset_root=dataset_root, dataset_id=dataset_id)
        vector_keys = {
            str(row["feature_vector_key"])
            for row in vector_alias_rows
            if _has_current_alias_contract(row) and str(row.get("feature_vector_key") or "").strip()
        }
        scalar_keys = {
            str(row["feature_scalar_key"])
            for row in scalar_alias_rows
            if _has_current_alias_contract(row) and str(row.get("feature_scalar_key") or "").strip()
        }
        vector_result = _compact_payload_table_to_keys(
            feature_vector_path(dataset_root=dataset_root, dataset_id=dataset_id),
            schema=_FEATURE_VECTOR_SCHEMA,
            key_column="feature_vector_key",
            keep_keys=vector_keys,
            batch_size=_COMPACT_BATCH_SIZE,
        )
        scalar_result = _compact_payload_table_to_keys(
            feature_scalar_path(dataset_root=dataset_root, dataset_id=dataset_id),
            schema=_FEATURE_SCALAR_SCHEMA,
            key_column="feature_scalar_key",
            keep_keys=scalar_keys,
            batch_size=_COMPACT_BATCH_SIZE,
        )
    removed_vectors = vector_result["rows_before"] - vector_result["rows_after"]
    removed_scalars = scalar_result["rows_before"] - scalar_result["rows_after"]
    if removed_vectors or removed_scalars:
        dataset_dir = Path(dataset_root) / dataset_id
        Dataset(Path(dataset_root), dataset_id).log_event(
            "infer_feature_sidecars_compact",
            args={
                "removed_vector_rows": removed_vectors,
                "removed_scalar_rows": removed_scalars,
                "kept_vector_rows": vector_result["rows_after"],
                "kept_scalar_rows": scalar_result["rows_after"],
                "sidecar_contract": "sequence_view_feature_bundle",
            },
            metrics={
                "removed_vector_rows": removed_vectors,
                "removed_scalar_rows": removed_scalars,
                "kept_vector_rows": vector_result["rows_after"],
                "kept_scalar_rows": scalar_result["rows_after"],
            },
            artifacts={
                "feature_aliases": FEATURE_ALIAS_RELATIVE_PATH,
                "feature_vectors": FEATURE_VECTOR_RELATIVE_PATH,
                "feature_scalar_aliases": FEATURE_SCALAR_ALIAS_RELATIVE_PATH,
                "feature_scalars": FEATURE_SCALAR_RELATIVE_PATH,
            },
            target_path=dataset_dir / "records.parquet",
            actor=_infer_sidecar_actor(default_run_id="feature-sidecar-compaction"),
        )
    return {
        "removed_vector_rows": removed_vectors,
        "removed_scalar_rows": removed_scalars,
        "kept_vector_rows": vector_result["rows_after"],
        "kept_scalar_rows": scalar_result["rows_after"],
    }


def _compact_payload_table_to_keys(
    path: Path,
    *,
    schema: pa.Schema,
    key_column: str,
    keep_keys: set[str],
    batch_size: int,
) -> dict[str, int]:
    if not path.exists():
        return {"rows_before": 0, "rows_after": 0}
    parquet_file = pq.ParquetFile(path)
    rows_before = int(parquet_file.metadata.num_rows) if parquet_file.metadata is not None else 0
    payload_keys = {
        str(value)
        for value in pq.read_table(path, columns=[key_column]).column(key_column).to_pylist()
        if str(value).strip()
    }
    if payload_keys.issubset(keep_keys):
        return {"rows_before": rows_before, "rows_after": rows_before}
    keep_values = pa.array(sorted(keep_keys), type=pa.string())
    temp_path = atomic_parquet_temp_path(path)
    writer: pq.ParquetWriter | None = None
    rows_after = 0
    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch]).cast(schema)
            mask = pc.is_in(table.column(key_column), value_set=keep_values)
            filtered = table.filter(mask)
            if filtered.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(temp_path, schema)
            writer.write_table(filtered)
            rows_after += filtered.num_rows
        if writer is not None:
            writer.close()
            writer = None
        else:
            pq.write_table(_empty_table(schema), temp_path)
        os.replace(temp_path, path)
    finally:
        if writer is not None:
            writer.close()
        temp_path.unlink(missing_ok=True)
    return {"rows_before": rows_before, "rows_after": rows_after}


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.table({field.name: pa.array([], type=field.type) for field in schema}, schema=schema)


def persist_feature_vector_rows(rows: Iterable[dict[str, object]]) -> int:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        payload = dict(row)
        dataset_root = str(payload.pop("_dataset_root")).strip()
        dataset_id = str(payload.pop("_dataset_id")).strip()
        grouped[(dataset_root, dataset_id)].append(payload)

    total_written = 0
    for (dataset_root, dataset_id), payload_rows in grouped.items():
        path = feature_vector_path(dataset_root=dataset_root, dataset_id=dataset_id)
        group_written = 0
        with sidecar_dataset_lock(dataset_root=dataset_root, dataset_id=dataset_id):
            incoming_by_key: dict[str, dict[str, object]] = {}
            for row in payload_rows:
                key = str(row["feature_vector_key"])
                value = [float(item) for item in row["value"]]
                payload = {
                    "feature_vector_key": key,
                    "value": value,
                    "created_at": str(row["created_at"]),
                }
                current = incoming_by_key.get(key)
                if current is not None:
                    if current["value"] != value:
                        raise ValueError(f"Feature vector collision within incoming payload for '{key}'.")
                    continue
                incoming_by_key[key] = payload
            existing_keys = load_feature_vector_keys(
                dataset_root=dataset_root,
                dataset_id=dataset_id,
                keys=incoming_by_key,
            )
            existing_matches = (
                load_feature_vector_rows(
                    dataset_root=dataset_root,
                    dataset_id=dataset_id,
                    keys=existing_keys,
                )
                if existing_keys
                else {}
            )
            if set(existing_matches) != existing_keys:
                missing = sorted(existing_keys - set(existing_matches))
                sample = ", ".join(missing[:3])
                raise ValueError(f"Feature vector payload missing for existing key(s): {sample}")
            for key, existing_value in existing_matches.items():
                if existing_value != incoming_by_key[key]["value"]:
                    raise ValueError(f"Feature vector collision with different payload for '{key}'.")
            append_rows = [row for key, row in incoming_by_key.items() if key not in existing_keys]
            group_written = len(append_rows)
            if group_written:
                _append_feature_vector_rows(path, append_rows)
                total_written += group_written
        if group_written:
            Dataset(Path(dataset_root), dataset_id).log_event(
                "infer_feature_vectors_write",
                args={"rows_written": group_written},
                artifacts={"path": path.relative_to(Path(dataset_root) / dataset_id).as_posix()},
                target_path=path,
                actor=_infer_sidecar_actor(default_run_id="feature-vector-persistence"),
            )
    return total_written


def persist_feature_scalar_rows(rows: Iterable[dict[str, object]]) -> int:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        payload = dict(row)
        dataset_root = str(payload.pop("_dataset_root")).strip()
        dataset_id = str(payload.pop("_dataset_id")).strip()
        grouped[(dataset_root, dataset_id)].append(payload)

    total_written = 0
    for (dataset_root, dataset_id), payload_rows in grouped.items():
        path = feature_scalar_path(dataset_root=dataset_root, dataset_id=dataset_id)
        group_written = 0
        with sidecar_dataset_lock(dataset_root=dataset_root, dataset_id=dataset_id):
            incoming_by_key: dict[str, dict[str, object]] = {}
            for row in payload_rows:
                key = str(row["feature_scalar_key"])
                payload = {
                    "feature_scalar_key": key,
                    "value": float(row["value"]),
                    "created_at": row.get("created_at"),
                }
                current = incoming_by_key.get(key)
                if current is not None:
                    if float(current["value"]) != payload["value"]:
                        raise ValueError(f"Feature scalar collision within incoming payload for '{key}'.")
                    continue
                incoming_by_key[key] = payload
            existing_keys = load_feature_scalar_keys(
                dataset_root=dataset_root,
                dataset_id=dataset_id,
                keys=incoming_by_key,
            )
            existing_matches = (
                load_feature_scalar_rows(
                    dataset_root=dataset_root,
                    dataset_id=dataset_id,
                    keys=existing_keys,
                )
                if existing_keys
                else {}
            )
            if set(existing_matches) != existing_keys:
                missing = sorted(existing_keys - set(existing_matches))
                sample = ", ".join(missing[:3])
                raise ValueError(f"Feature scalar payload missing for existing key(s): {sample}")
            for key, existing_value in existing_matches.items():
                if existing_value != float(incoming_by_key[key]["value"]):
                    raise ValueError(f"Feature scalar collision with different value for '{key}'.")
            append_rows = [row for key, row in incoming_by_key.items() if key not in existing_keys]
            group_written = len(append_rows)
            if group_written:
                _append_feature_scalar_rows(path, append_rows)
                total_written += group_written
        if group_written:
            Dataset(Path(dataset_root), dataset_id).log_event(
                "infer_feature_scalars_write",
                args={"rows_written": group_written},
                artifacts={"path": path.relative_to(Path(dataset_root) / dataset_id).as_posix()},
                target_path=path,
                actor=_infer_sidecar_actor(default_run_id="feature-scalar-persistence"),
            )
    return total_written


def record_feature_bundle_complete(
    *,
    dataset_root: str | Path,
    dataset_id: str,
    job_id: str,
    model_id: str,
    contexts_completed: int,
    unique_forward_passes: int,
    required_vector_keys: int,
    required_scalar_keys: int,
    run_elapsed_seconds: float | None = None,
) -> None:
    dataset_dir = Path(dataset_root) / dataset_id
    args: dict[str, object] = {
        "job_id": str(job_id),
        "model_id": str(model_id),
        "contexts_completed": int(contexts_completed),
        "unique_forward_passes": int(unique_forward_passes),
        "required_vector_keys": int(required_vector_keys),
        "required_scalar_keys": int(required_scalar_keys),
        "sidecar_contract": "sequence_view_feature_bundle",
    }
    metrics: dict[str, object] = {
        "contexts_completed": int(contexts_completed),
        "unique_forward_passes": int(unique_forward_passes),
        "required_vector_keys": int(required_vector_keys),
        "required_scalar_keys": int(required_scalar_keys),
    }
    if run_elapsed_seconds is not None:
        elapsed = max(0.0, float(run_elapsed_seconds))
        args["run_elapsed_seconds"] = elapsed
        metrics["run_elapsed_seconds"] = elapsed
    Dataset(Path(dataset_root), dataset_id).log_event(
        _FEATURE_BUNDLE_COMPLETE_ACTION,
        args=args,
        metrics=metrics,
        artifacts={
            "feature_aliases": FEATURE_ALIAS_RELATIVE_PATH,
            "feature_vectors": FEATURE_VECTOR_RELATIVE_PATH,
            "feature_scalar_aliases": FEATURE_SCALAR_ALIAS_RELATIVE_PATH,
            "feature_scalars": FEATURE_SCALAR_RELATIVE_PATH,
        },
        target_path=dataset_dir / "records.parquet",
        actor=_infer_sidecar_actor(default_run_id=f"infer-{job_id}"),
    )


def record_feature_bundle_progress(
    *,
    dataset_root: str | Path,
    dataset_id: str,
    job_id: str,
    model_id: str,
    contexts_completed: int,
    contexts_total: int,
    unique_forward_passes_completed: int,
    unique_forward_passes_total: int,
    required_vector_keys: int,
    required_scalar_keys: int,
    run_elapsed_seconds: float | None = None,
) -> None:
    dataset_dir = Path(dataset_root) / dataset_id
    total_contexts = max(0, int(contexts_total))
    completed_contexts = max(0, min(int(contexts_completed), total_contexts))
    progress_pct = 100.0 if total_contexts == 0 else (float(completed_contexts) * 100.0 / float(total_contexts))
    args: dict[str, object] = {
        "job_id": str(job_id),
        "model_id": str(model_id),
        "contexts_completed": completed_contexts,
        "contexts_total": total_contexts,
        "progress_pct": round(progress_pct, 3),
        "unique_forward_passes_completed": max(0, int(unique_forward_passes_completed)),
        "unique_forward_passes_total": max(0, int(unique_forward_passes_total)),
        "required_vector_keys": int(required_vector_keys),
        "required_scalar_keys": int(required_scalar_keys),
        "sidecar_contract": "sequence_view_feature_bundle",
    }
    metrics: dict[str, object] = {
        "contexts_completed": completed_contexts,
        "contexts_total": total_contexts,
        "progress_pct": round(progress_pct, 3),
        "unique_forward_passes_completed": max(0, int(unique_forward_passes_completed)),
        "unique_forward_passes_total": max(0, int(unique_forward_passes_total)),
        "required_vector_keys": int(required_vector_keys),
        "required_scalar_keys": int(required_scalar_keys),
    }
    if run_elapsed_seconds is not None:
        elapsed = max(0.0, float(run_elapsed_seconds))
        args["run_elapsed_seconds"] = elapsed
        metrics["run_elapsed_seconds"] = elapsed
    Dataset(Path(dataset_root), dataset_id).log_event(
        _FEATURE_BUNDLE_PROGRESS_ACTION,
        args=args,
        metrics=metrics,
        artifacts={
            "feature_aliases": FEATURE_ALIAS_RELATIVE_PATH,
            "feature_vectors": FEATURE_VECTOR_RELATIVE_PATH,
            "feature_scalar_aliases": FEATURE_SCALAR_ALIAS_RELATIVE_PATH,
            "feature_scalars": FEATURE_SCALAR_RELATIVE_PATH,
        },
        target_path=dataset_dir / "records.parquet",
        actor=_infer_sidecar_actor(default_run_id=f"infer-{job_id}"),
    )


def record_feature_bundle_shard_commit(
    *,
    dataset_root: str | Path,
    dataset_id: str,
    job_id: str,
    model_id: str,
    shard_index: int,
    shard_count: int,
    contexts_committed: int,
    committed_vector_keys: int,
    committed_scalar_keys: int,
    checksum: str,
    runtime_fingerprint_key: str,
    ledger_relative_path: str,
    run_elapsed_seconds: float | None = None,
) -> None:
    dataset_dir = Path(dataset_root) / dataset_id
    args: dict[str, object] = {
        "job_id": str(job_id),
        "model_id": str(model_id),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "contexts_committed": int(contexts_committed),
        "committed_vector_keys": int(committed_vector_keys),
        "committed_scalar_keys": int(committed_scalar_keys),
        "checksum": str(checksum),
        "runtime_fingerprint_key": str(runtime_fingerprint_key),
        "ledger": str(ledger_relative_path),
        "sidecar_contract": "sequence_view_feature_bundle",
        "commit_policy": "temp_validate_promote",
    }
    metrics: dict[str, object] = {
        "contexts_committed": int(contexts_committed),
        "committed_vector_keys": int(committed_vector_keys),
        "committed_scalar_keys": int(committed_scalar_keys),
    }
    if run_elapsed_seconds is not None:
        elapsed = max(0.0, float(run_elapsed_seconds))
        args["run_elapsed_seconds"] = elapsed
        metrics["run_elapsed_seconds"] = elapsed
    Dataset(Path(dataset_root), dataset_id).log_event(
        _FEATURE_BUNDLE_SHARD_COMMIT_ACTION,
        args=args,
        metrics=metrics,
        artifacts={
            "feature_aliases": FEATURE_ALIAS_RELATIVE_PATH,
            "feature_vectors": FEATURE_VECTOR_RELATIVE_PATH,
            "feature_scalar_aliases": FEATURE_SCALAR_ALIAS_RELATIVE_PATH,
            "feature_scalars": FEATURE_SCALAR_RELATIVE_PATH,
            "ledger": str(ledger_relative_path),
        },
        target_path=dataset_dir / "records.parquet",
        actor=_infer_sidecar_actor(default_run_id=f"infer-{job_id}"),
    )


def _table_from_rows(rows: list[dict[str, object]], *, schema: pa.Schema) -> pa.Table:
    return pa.table(
        {field.name: pa.array([row.get(field.name) for row in rows], type=field.type) for field in schema},
        schema=schema,
    )


def _replace_with_appended_rows(
    path: Path,
    *,
    schema: pa.Schema,
    append_rows: list[dict[str, object]],
    batch_size: int,
) -> None:
    if not path.exists():
        write_table_atomic(_table_from_rows(append_rows, schema=schema), path)
        return

    tmp_path = atomic_parquet_temp_path(path)
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        with pq.ParquetWriter(tmp_path, schema=schema) as writer:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                writer.write_table(pa.Table.from_batches([batch], schema=schema))
            writer.write_table(_table_from_rows(append_rows, schema=schema))
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _append_feature_vector_rows(path: Path, rows: list[dict[str, object]]) -> None:
    _replace_with_appended_rows(
        path,
        schema=_FEATURE_VECTOR_SCHEMA,
        append_rows=rows,
        batch_size=_VECTOR_BATCH_SIZE,
    )


def _append_feature_scalar_rows(path: Path, rows: list[dict[str, object]]) -> None:
    _replace_with_appended_rows(
        path,
        schema=_FEATURE_SCALAR_SCHEMA,
        append_rows=rows,
        batch_size=_SCALAR_BATCH_SIZE,
    )
