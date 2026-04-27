"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/aliases.py

Sequence-view feature alias persistence for deduplicated Infer execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .cache_keys import stable_sha256

FEATURE_ALIAS_RELATIVE_PATH = "_derived/infer/feature_aliases.parquet"
FEATURE_VECTOR_RELATIVE_PATH = "_derived/infer/feature_vectors.parquet"

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


def feature_alias_path(*, dataset_root: str | Path, dataset_id: str) -> Path:
    return Path(dataset_root) / dataset_id / FEATURE_ALIAS_RELATIVE_PATH


def feature_vector_path(*, dataset_root: str | Path, dataset_id: str) -> Path:
    return Path(dataset_root) / dataset_id / FEATURE_VECTOR_RELATIVE_PATH


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


def persist_feature_alias_rows(rows: Iterable[dict[str, object]]) -> int:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        dataset_root = str(row.pop("_dataset_root")).strip()
        dataset_id = str(row.pop("_dataset_id")).strip()
        grouped[(dataset_root, dataset_id)].append(dict(row))

    total_written = 0
    for (dataset_root, dataset_id), payload_rows in grouped.items():
        path = feature_alias_path(dataset_root=dataset_root, dataset_id=dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_rows: list[dict[str, object]] = []
        if path.exists():
            existing_rows = pq.read_table(path).to_pylist()
        existing_by_id = {str(row["alias_id"]): row for row in existing_rows}
        for row in payload_rows:
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
        table = pa.table(
            {
                field.name: pa.array([row.get(field.name) for row in existing_by_id.values()], type=field.type)
                for field in _FEATURE_ALIAS_SCHEMA
            },
            schema=_FEATURE_ALIAS_SCHEMA,
        )
        pq.write_table(table, path)
    return total_written


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
    table = pq.read_table(path).cast(_FEATURE_VECTOR_SCHEMA)
    out: dict[str, list[float]] = {}
    for row in table.to_pylist():
        key = str(row["feature_vector_key"])
        if key in wanted:
            out[key] = [float(value) for value in row["value"]]
    return out


def persist_feature_vector_rows(rows: Iterable[dict[str, object]]) -> int:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        dataset_root = str(row.pop("_dataset_root")).strip()
        dataset_id = str(row.pop("_dataset_id")).strip()
        grouped[(dataset_root, dataset_id)].append(dict(row))

    total_written = 0
    for (dataset_root, dataset_id), payload_rows in grouped.items():
        path = feature_vector_path(dataset_root=dataset_root, dataset_id=dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_rows: list[dict[str, object]] = []
        if path.exists():
            existing_rows = pq.read_table(path).cast(_FEATURE_VECTOR_SCHEMA).to_pylist()
        existing_by_key = {str(row["feature_vector_key"]): row for row in existing_rows}
        for row in payload_rows:
            key = str(row["feature_vector_key"])
            value = [float(item) for item in row["value"]]
            current = existing_by_key.get(key)
            if current is not None:
                if [float(item) for item in current["value"]] != value:
                    raise ValueError(f"Feature vector collision with different payload for '{key}'.")
                continue
            existing_by_key[key] = {
                "feature_vector_key": key,
                "value": value,
                "created_at": str(row["created_at"]),
            }
            total_written += 1
        table = pa.table(
            {
                field.name: pa.array([row.get(field.name) for row in existing_by_key.values()], type=field.type)
                for field in _FEATURE_VECTOR_SCHEMA
            },
            schema=_FEATURE_VECTOR_SCHEMA,
        )
        pq.write_table(table, path)
    return total_written
