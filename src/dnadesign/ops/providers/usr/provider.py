"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/providers/usr/provider.py

Provider-owned USR status builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.ops.status.artifacts import (
    file_count,
    line_count,
    namespace_column_counts,
    overlay_namespace_names,
)
from dnadesign.ops.status.parsing import required_text
from dnadesign.ops.status.paths import (
    required_path,
)


def provide_usr_sync_audit_status(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    del repo_root
    return _usr_sync_audit_status(inputs.get("sync_audit_json"))


def provide_usr_dataset_state_status(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    del repo_root
    return _usr_dataset_state_status(usr_root=inputs.get("usr_root"), dataset=inputs.get("dataset"))


def _usr_sync_audit_status(sync_audit_json: object) -> tuple[str, str, dict[str, object]]:
    resolved_audit = required_path(sync_audit_json, flag_name="--sync-audit-json", status_kind="usr-sync-audit")
    if not resolved_audit.exists():
        return (
            "missing",
            "sync audit artifact not found",
            {"sync_audit_json": str(resolved_audit)},
        )
    payload = json.loads(resolved_audit.read_text(encoding="utf-8"))
    transfer_state = str(payload.get("transfer_state") or "UNKNOWN")
    changed_flags = {
        "primary": bool((payload.get("primary") or {}).get("changed")),
        "meta": bool((payload.get("meta") or {}).get("changed")),
        "_snapshots": bool((payload.get("_snapshots") or {}).get("changed")),
        "_derived": bool((payload.get("_derived") or {}).get("changed")),
        "_auxiliary": bool((payload.get("_auxiliary") or {}).get("changed")),
    }
    has_pending_drift = any(changed_flags.values())
    is_ok = transfer_state in {"NO-OP", "TRANSFERRED"} and not has_pending_drift
    summary = f"{payload.get('dataset', '<unknown>')}: {transfer_state}"
    if has_pending_drift:
        summary += " with remaining drift"
    return (
        "ok" if is_ok else "attention",
        summary,
        {
            "sync_audit_json": str(resolved_audit),
            "action": payload.get("action"),
            "dataset": payload.get("dataset"),
            "transfer_state": transfer_state,
            "verify": dict(payload.get("verify") or {}),
            "changed_flags": changed_flags,
            "events_log": dict(payload.get(".events.log") or {}),
        },
    )


def _usr_dataset_state_status(
    *,
    usr_root: object,
    dataset: object,
) -> tuple[str, str, dict[str, object]]:
    resolved_root = required_path(usr_root, flag_name="--usr-root", status_kind="usr-dataset-state")
    dataset_id = required_text(dataset, flag_name="--dataset", status_kind="usr-dataset-state")
    dataset_dir = (resolved_root / dataset_id).resolve()
    records_path = dataset_dir / "records.parquet"
    if not records_path.exists():
        return (
            "missing",
            f"USR dataset not found: {dataset_id}",
            {
                "usr_root": str(resolved_root),
                "dataset": dataset_id,
                "records_path": str(records_path),
            },
        )

    parquet_file = pq.ParquetFile(str(records_path))
    schema = parquet_file.schema_arrow
    columns = list(schema.names)
    namespace_counts = namespace_column_counts(columns)
    overlay_namespaces = overlay_namespace_names(dataset_dir)
    events_log_path = dataset_dir / ".events.log"
    snapshots_dir = dataset_dir / "_snapshots"
    events_count = line_count(events_log_path) if events_log_path.exists() else 0
    snapshots_count = file_count(snapshots_dir) if snapshots_dir.exists() else 0
    rows = int(parquet_file.metadata.num_rows)
    cols = int(parquet_file.metadata.num_columns)
    infer_columns = int(namespace_counts.get("infer", 0))
    summary = f"{dataset_id}: {rows} rows, {cols} columns"
    if infer_columns:
        summary += f", {infer_columns} infer-derived columns"
    return (
        "ok" if rows > 0 else "attention",
        summary,
        {
            "usr_root": str(resolved_root),
            "dataset": dataset_id,
            "dataset_dir": str(dataset_dir),
            "records_path": str(records_path),
            "rows": rows,
            "columns": cols,
            "namespace_column_counts": dict(sorted(namespace_counts.items())),
            "overlay_namespaces": overlay_namespaces,
            "overlay_namespace_count": len(overlay_namespaces),
            "events_count": events_count,
            "snapshots_count": snapshots_count,
        },
    )


__all__ = ["provide_usr_dataset_state_status", "provide_usr_sync_audit_status"]
