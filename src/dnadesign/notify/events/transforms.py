"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/events/transforms.py

Event status, message, metadata, and schema-validation helpers for notify.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from ..errors import NotifyConfigError


def event_meta(
    event: dict[str, Any],
    *,
    include_args: bool,
    include_raw_event: bool,
    include_context: bool,
) -> dict[str, Any]:
    dataset_raw = event.get("dataset")
    dataset = dataset_raw if isinstance(dataset_raw, dict) else {}
    meta = {
        "usr_event_version": event.get("event_version"),
        "usr_action": event.get("action"),
        "usr_dataset_name": dataset.get("name"),
        "usr_fingerprint": event.get("fingerprint"),
        "usr_registry_hash": event.get("registry_hash"),
        "usr_timestamp": event.get("timestamp_utc"),
    }
    if include_context:
        meta["usr_dataset_root"] = dataset.get("root")
    if include_args:
        meta["usr_args"] = event.get("args")
    if include_raw_event:
        meta["usr_event"] = event
    return meta


def status_for_action(action: str, *, event: dict[str, Any] | None = None) -> str:
    del event
    action_norm = str(action or "").strip().lower()
    if not action_norm:
        return "running"
    if "fail" in action_norm or "error" in action_norm:
        return "failure"
    if action_norm == "init":
        return "started"
    if action_norm in {"materialize", "compact_overlay", "overlay_compact", "registry_freeze"}:
        return "running"
    return "running"


def event_message(
    event: dict[str, Any],
    *,
    run_id: str,
    duration_seconds: float | None,
) -> str:
    del run_id
    del duration_seconds
    action = str(event.get("action") or "event")
    dataset_raw = event.get("dataset")
    dataset = dataset_raw if isinstance(dataset_raw, dict) else {}
    dataset_name = dataset.get("name") or "unknown-dataset"
    metrics_raw = event.get("metrics")
    metrics = metrics_raw if isinstance(metrics_raw, dict) else {}

    rows_written = metrics.get("rows_written")
    if rows_written is not None:
        return f"{action} on {dataset_name} (rows_written={rows_written})"
    return f"{action} on {dataset_name}"


def validate_usr_event(
    event: dict[str, Any],
    *,
    expected_version: int,
    allow_unknown_version: bool,
) -> None:
    if not isinstance(event, dict):
        raise NotifyConfigError("event line must decode to a JSON object")
    if "event_version" not in event:
        raise NotifyConfigError("event missing required 'event_version'")
    version = event.get("event_version")
    if not isinstance(version, int):
        raise NotifyConfigError("event_version must be an integer")
    if version != expected_version and not allow_unknown_version:
        raise NotifyConfigError(f"unknown event_version={version}; expected {expected_version}")
    action = event.get("action")
    if not isinstance(action, str) or not action.strip():
        raise NotifyConfigError("event missing required 'action'")
    dataset = event.get("dataset")
    if dataset is not None and not isinstance(dataset, dict):
        raise NotifyConfigError("event field 'dataset' must be an object when provided")
    actor = event.get("actor")
    if actor is not None and not isinstance(actor, dict):
        raise NotifyConfigError("event field 'actor' must be an object when provided")
