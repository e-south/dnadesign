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
import pyarrow.parquet as pq

from dnadesign.usr import Dataset

from .cache_keys import stable_sha256

FEATURE_ALIAS_RELATIVE_PATH = "_derived/infer/feature_aliases.parquet"
FEATURE_VECTOR_RELATIVE_PATH = "_derived/infer/feature_vectors.parquet"
FEATURE_SCALAR_ALIAS_RELATIVE_PATH = "_derived/infer/feature_scalar_aliases.parquet"
FEATURE_SCALAR_RELATIVE_PATH = "_derived/infer/feature_scalars.parquet"

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
_FEATURE_BUNDLE_PROGRESS_ACTION = "infer_feature_bundle_progress"
_FEATURE_BUNDLE_COMPLETE_ACTION = "infer_feature_bundle_complete"


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
        group_written = 0
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
            group_written += 1
        table = pa.table(
            {
                field.name: pa.array([row.get(field.name) for row in existing_by_id.values()], type=field.type)
                for field in _FEATURE_ALIAS_SCHEMA
            },
            schema=_FEATURE_ALIAS_SCHEMA,
        )
        pq.write_table(table, path)
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
        dataset_root = str(row.pop("_dataset_root")).strip()
        dataset_id = str(row.pop("_dataset_id")).strip()
        grouped[(dataset_root, dataset_id)].append(dict(row))

    total_written = 0
    for (dataset_root, dataset_id), payload_rows in grouped.items():
        path = feature_scalar_alias_path(dataset_root=dataset_root, dataset_id=dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_rows: list[dict[str, object]] = []
        if path.exists():
            existing_rows = pq.read_table(path).cast(_FEATURE_SCALAR_ALIAS_SCHEMA).to_pylist()
        existing_by_id = {str(row["alias_id"]): row for row in existing_rows}
        group_written = 0
        for row in payload_rows:
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
        pq.write_table(table, path)
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
    return pq.read_table(path).cast(_FEATURE_ALIAS_SCHEMA).to_pylist()


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
    return pq.read_table(path).cast(_FEATURE_SCALAR_ALIAS_SCHEMA).to_pylist()


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
        dataset_root = str(row.pop("_dataset_root")).strip()
        dataset_id = str(row.pop("_dataset_id")).strip()
        grouped[(dataset_root, dataset_id)].append(dict(row))

    total_written = 0
    for (dataset_root, dataset_id), payload_rows in grouped.items():
        path = feature_scalar_path(dataset_root=dataset_root, dataset_id=dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
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
        pq.write_table(_table_from_rows(append_rows, schema=schema), path)
        return

    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        with pq.ParquetWriter(tmp_path, schema=schema) as writer:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                writer.write_table(pa.Table.from_batches([batch], schema=schema))
            writer.write_table(_table_from_rows(append_rows, schema=schema))
        tmp_path.replace(path)
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
