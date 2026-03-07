"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tool_events/infer.py

Infer tool-event status/message/evaluation handlers for notify.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .types import ToolEventDecision, ToolEventState

_INFER_ATTACH_MIN_SECONDS_DEFAULT = 60.0
_INFER_ATTACH_HEARTBEAT_SECONDS_DEFAULT = 1800.0


def _infer_actor_tool(event: dict[str, Any]) -> str:
    actor_raw = event.get("actor")
    actor = actor_raw if isinstance(actor_raw, dict) else {}
    return str(actor.get("tool") or "").strip().lower()


def _is_infer_actor(event: dict[str, Any]) -> bool:
    return _infer_actor_tool(event) == "infer"


def _event_timestamp_seconds(event: dict[str, Any]) -> float | None:
    raw = event.get("timestamp_utc")
    if not isinstance(raw, str) or not raw.strip():
        return None
    ts = raw.strip()
    if ts.endswith("Z"):
        ts = f"{ts[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return float(parsed.timestamp())


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_progress_min_seconds(notify_config: dict[str, Any]) -> float:
    configured = _to_float_or_none(notify_config.get("progress_min_seconds"))
    if configured is None:
        return float(_INFER_ATTACH_MIN_SECONDS_DEFAULT)
    if configured <= 0.0:
        return float(_INFER_ATTACH_MIN_SECONDS_DEFAULT)
    return float(configured)


def _resolve_progress_heartbeat_seconds(notify_config: dict[str, Any]) -> float:
    configured = _to_float_or_none(notify_config.get("progress_heartbeat_seconds"))
    if configured is None:
        return float(_INFER_ATTACH_HEARTBEAT_SECONDS_DEFAULT)
    if configured <= 0.0:
        return float(_INFER_ATTACH_HEARTBEAT_SECONDS_DEFAULT)
    return float(configured)


def _infer_attach_status_override(event: dict[str, Any]) -> str | None:
    if not _is_infer_actor(event):
        return None
    return "running"


def _infer_attach_message(
    event: dict[str, Any],
    *,
    run_id: str,
    duration_seconds: float | None,
) -> str | None:
    del duration_seconds
    if not _is_infer_actor(event):
        return None
    dataset_raw = event.get("dataset")
    dataset = dataset_raw if isinstance(dataset_raw, dict) else {}
    dataset_name = str(dataset.get("name") or "unknown-dataset")
    args_raw = event.get("args")
    args = args_raw if isinstance(args_raw, dict) else {}
    rows_incoming = args.get("rows_incoming")
    rows_matched = args.get("rows_matched")
    rows_missing = args.get("rows_missing")
    fingerprint_raw = event.get("fingerprint")
    fingerprint = fingerprint_raw if isinstance(fingerprint_raw, dict) else {}
    workspace_rows = fingerprint.get("rows")
    lines = [f"Infer write-back progress | run={run_id} | dataset={dataset_name}"]
    if rows_incoming is not None or rows_matched is not None or rows_missing is not None:
        lines.append(
            f"- Chunk rows: incoming={rows_incoming if rows_incoming is not None else 0} "
            f"matched={rows_matched if rows_matched is not None else 0} "
            f"missing={rows_missing if rows_missing is not None else 0}"
        )
    if workspace_rows is not None:
        lines.append(f"- Workspace rows: {workspace_rows}")
    return "\n".join(lines)


def _evaluate_infer_attach_event(event: dict[str, Any], run_id: str, state: ToolEventState) -> ToolEventDecision:
    if not _is_infer_actor(event):
        return ToolEventDecision(emit=True, duration_seconds=None)
    bucket = state.get_bucket("infer_attach")
    per_run_raw = bucket.setdefault("per_run", {})
    per_run = per_run_raw if isinstance(per_run_raw, dict) else {}
    notify_config_raw = bucket.get("notify_config")
    notify_config = notify_config_raw if isinstance(notify_config_raw, dict) else {}
    min_seconds = _resolve_progress_min_seconds(notify_config)
    heartbeat_seconds = _resolve_progress_heartbeat_seconds(notify_config)
    now_seconds = _event_timestamp_seconds(event)
    if now_seconds is None:
        now_seconds = float(time.time())

    run_key = str(run_id)
    entry_raw = per_run.get(run_key)
    entry = entry_raw if isinstance(entry_raw, dict) else {}
    last_sent_raw = entry.get("last_sent")
    last_sent = _to_float_or_none(last_sent_raw)
    if last_sent is None:
        per_run[run_key] = {"last_sent": now_seconds}
        return ToolEventDecision(emit=True, duration_seconds=None)

    elapsed = now_seconds - last_sent
    if elapsed >= heartbeat_seconds:
        per_run[run_key] = {"last_sent": now_seconds}
        return ToolEventDecision(emit=True, duration_seconds=None)
    if elapsed < min_seconds:
        return ToolEventDecision(emit=False, duration_seconds=None)
    per_run[run_key] = {"last_sent": now_seconds}
    return ToolEventDecision(emit=True, duration_seconds=None)


def register_infer_handlers(
    *,
    register_status_override: Callable[[str, Callable[[dict[str, Any]], str | None]], None],
    register_message_override: Callable[
        [
            str,
            Callable[[dict[str, Any]], str] | Callable[[dict[str, Any], str, float | None], str] | Callable[..., str],
        ],
        None,
    ],
    register_evaluator: Callable[[str, Callable[[dict[str, Any], str, ToolEventState], ToolEventDecision]], None],
) -> None:
    register_status_override("attach", _infer_attach_status_override)
    register_message_override("attach", _infer_attach_message)
    register_evaluator("attach", _evaluate_infer_attach_event)
