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

from ..errors import NotifyConfigError
from .densegen_common import _duration_hhmmss
from .types import ToolEventDecision, ToolEventState

_INFER_PROGRESS_STEP_PCT_SMALL_TARGET = 25
_INFER_PROGRESS_STEP_PCT_LARGE_TARGET = 10
_INFER_SMALL_TARGET_UNITS_THRESHOLD = 200
_INFER_ATTACH_MIN_SECONDS_DEFAULT = 60.0
_INFER_ATTACH_HEARTBEAT_SECONDS_DEFAULT = 1800.0


def _eta_to_overall_complete_seconds(
    *,
    completed_units: int,
    target_units: int,
    elapsed_seconds: float | None,
) -> float | None:
    if elapsed_seconds is None:
        return None
    if elapsed_seconds <= 0.0 or completed_units <= 0:
        return None
    remaining_units = int(target_units) - int(completed_units)
    if remaining_units <= 0:
        return None
    units_per_second = float(completed_units) / float(elapsed_seconds)
    if units_per_second <= 0.0:
        return None
    return float(remaining_units) / units_per_second


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


def _to_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_args(event: dict[str, Any]) -> dict[str, Any]:
    args_raw = event.get("args")
    return args_raw if isinstance(args_raw, dict) else {}


def _infer_output(event: dict[str, Any]) -> dict[str, Any]:
    output_raw = _infer_args(event).get("infer_output")
    return output_raw if isinstance(output_raw, dict) else {}


def _infer_progress(event: dict[str, Any]) -> dict[str, Any]:
    progress_raw = _infer_args(event).get("infer_progress")
    return progress_raw if isinstance(progress_raw, dict) else {}


def _infer_output_id(event: dict[str, Any]) -> str:
    return str(_infer_output(event).get("id") or "").strip()


def _infer_output_kind(event: dict[str, Any]) -> str:
    return str(_infer_output(event).get("kind") or "").strip().lower()


def _infer_notify_suppress(event: dict[str, Any]) -> bool:
    args = _infer_args(event)
    if bool(args.get("infer_notify_suppress")):
        return True
    return _infer_output_kind(event) == "metadata"


def _resolve_progress_step_pct(progress: dict[str, Any], notify_config: dict[str, Any]) -> int:
    configured = notify_config.get("progress_step_pct")
    if configured is not None:
        value = _to_int_or_none(configured)
        if value is None or value < 1 or value > 100:
            raise NotifyConfigError("progress_step_pct must be an integer between 1 and 100")
        return value
    overall_target_units = _to_int_or_none(progress.get("overall_target_units"))
    if overall_target_units is not None and overall_target_units <= _INFER_SMALL_TARGET_UNITS_THRESHOLD:
        return _INFER_PROGRESS_STEP_PCT_SMALL_TARGET
    return _INFER_PROGRESS_STEP_PCT_LARGE_TARGET


def _ordered_family_progress_parts(progress: dict[str, Any]) -> list[str]:
    progress_map_raw = progress.get("family_progress_pct_map")
    progress_map = progress_map_raw if isinstance(progress_map_raw, dict) else {}
    if not progress_map:
        return []
    preferred = ("log_likelihood", "output_layer_mean", "intermediate_embedding")
    parts: list[str] = []
    seen: set[str] = set()
    for family in (*preferred, *sorted(progress_map)):
        if family in seen:
            continue
        seen.add(family)
        pct = _to_float_or_none(progress_map.get(family))
        if pct is None:
            continue
        label = "LL" if family == "log_likelihood" else family
        parts.append(f"{label} {pct:.1f}%")
    return parts


def _infer_attach_signature(event: dict[str, Any], *, progress_step_pct: int) -> tuple[object, ...]:
    progress = _infer_progress(event)
    overall_progress_pct = _to_float_or_none(progress.get("overall_progress_pct")) or 0.0
    overall_step = int(max(0.0, min(100.0, overall_progress_pct)) // float(progress_step_pct))
    family_parts = tuple(_ordered_family_progress_parts(progress))
    output_id = _infer_output_id(event)
    output_progress_pct = _to_float_or_none(progress.get("output_progress_pct"))
    rows_matched = _to_int_or_none(_infer_args(event).get("rows_matched"))
    return (overall_step, output_id, round(output_progress_pct or 0.0, 1), rows_matched, family_parts)


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
    if _infer_notify_suppress(event):
        return None
    return "running"


def _infer_materialize_status_override(event: dict[str, Any]) -> str | None:
    if not _is_infer_actor(event):
        return None
    return "success"


def _infer_attach_message(
    event: dict[str, Any],
    *,
    run_id: str,
    duration_seconds: float | None,
) -> str | None:
    if not _is_infer_actor(event):
        return None
    if _infer_notify_suppress(event):
        return None
    dataset_raw = event.get("dataset")
    dataset = dataset_raw if isinstance(dataset_raw, dict) else {}
    dataset_name = str(dataset.get("name") or "unknown-dataset")
    args = _infer_args(event)
    progress = _infer_progress(event)
    output = _infer_output(event)
    rows_incoming = args.get("rows_incoming")
    rows_matched = args.get("rows_matched")
    rows_missing = args.get("rows_missing")
    fingerprint_raw = event.get("fingerprint")
    fingerprint = fingerprint_raw if isinstance(fingerprint_raw, dict) else {}
    workspace_rows = fingerprint.get("rows")
    lines = [f"Infer progress | run={run_id} | dataset={dataset_name}"]
    overall_progress_pct = _to_float_or_none(progress.get("overall_progress_pct"))
    overall_completed_units = _to_int_or_none(progress.get("overall_completed_units"))
    overall_target_units = _to_int_or_none(progress.get("overall_target_units"))
    if overall_progress_pct is not None:
        overall_line = f"- Overall requested outputs: {overall_progress_pct:.1f}%"
        if overall_completed_units is not None and overall_target_units is not None:
            overall_line += f" ({overall_completed_units}/{overall_target_units} units)"
        lines.append(overall_line)
        if overall_completed_units is not None and overall_target_units is not None:
            remaining_units = max(0, overall_target_units - overall_completed_units)
            lines.append(f"- Remaining overall units: {remaining_units}")
            eta_seconds = _eta_to_overall_complete_seconds(
                completed_units=overall_completed_units,
                target_units=overall_target_units,
                elapsed_seconds=duration_seconds,
            )
            if eta_seconds is not None:
                lines.append(f"- ETA to overall complete: {eta_seconds / 3600.0:.1f}h")
    family_parts = _ordered_family_progress_parts(progress)
    if family_parts:
        lines.append(f"- Families: {' | '.join(family_parts)}")
    output_id = str(output.get("id") or "").strip()
    output_progress_pct = _to_float_or_none(progress.get("output_progress_pct"))
    completed_rows = _to_int_or_none(progress.get("completed_rows"))
    target_rows = _to_int_or_none(progress.get("target_rows"))
    if output_id:
        output_line = f"- Current output: {output_id}"
        if output_progress_pct is not None and completed_rows is not None and target_rows is not None:
            output_line += f" {output_progress_pct:.1f}% ({completed_rows}/{target_rows} rows)"
        lines.append(output_line)
    if rows_incoming is not None or rows_matched is not None or rows_missing is not None:
        lines.append(
            f"- Chunk rows: incoming={rows_incoming if rows_incoming is not None else 0} "
            f"matched={rows_matched if rows_matched is not None else 0} "
            f"missing={rows_missing if rows_missing is not None else 0}"
        )
    if workspace_rows is not None:
        lines.append(f"- Workspace rows: {workspace_rows}")
    if duration_seconds is not None:
        lines.append(f"- Elapsed: {_duration_hhmmss(duration_seconds)}")
    return "\n".join(lines)


def _evaluate_infer_attach_event(event: dict[str, Any], run_id: str, state: ToolEventState) -> ToolEventDecision:
    if not _is_infer_actor(event):
        return ToolEventDecision(emit=True, duration_seconds=None)
    if _infer_notify_suppress(event):
        return ToolEventDecision(emit=False, duration_seconds=None)
    bucket = state.get_bucket("infer_attach")
    per_run_raw = bucket.setdefault("per_run", {})
    per_run = per_run_raw if isinstance(per_run_raw, dict) else {}
    notify_config_raw = bucket.get("notify_config")
    notify_config = notify_config_raw if isinstance(notify_config_raw, dict) else {}
    min_seconds = _resolve_progress_min_seconds(notify_config)
    heartbeat_seconds = _resolve_progress_heartbeat_seconds(notify_config)
    progress = _infer_progress(event)
    overall_progress_pct = _to_float_or_none(progress.get("overall_progress_pct"))
    now_seconds = _event_timestamp_seconds(event)
    if now_seconds is None:
        now_seconds = float(time.time())

    run_key = str(run_id)
    entry_raw = per_run.get(run_key)
    entry = entry_raw if isinstance(entry_raw, dict) else {}
    started_at_raw = entry.get("started_at")
    started_at = _to_float_or_none(started_at_raw)
    if started_at is None:
        started_at = now_seconds
    last_sent_raw = entry.get("last_sent")
    last_sent = _to_float_or_none(last_sent_raw)
    if overall_progress_pct is None:
        if last_sent is None:
            per_run[run_key] = {"started_at": started_at, "last_sent": now_seconds}
            return ToolEventDecision(emit=True, duration_seconds=max(0.0, now_seconds - started_at))
        elapsed = now_seconds - last_sent
        if elapsed >= heartbeat_seconds:
            per_run[run_key] = {"started_at": started_at, "last_sent": now_seconds}
            return ToolEventDecision(emit=True, duration_seconds=max(0.0, now_seconds - started_at))
        if elapsed < min_seconds:
            return ToolEventDecision(emit=False, duration_seconds=None)
        per_run[run_key] = {"started_at": started_at, "last_sent": now_seconds}
        return ToolEventDecision(emit=True, duration_seconds=max(0.0, now_seconds - started_at))

    progress_step_pct = _resolve_progress_step_pct(progress, notify_config)
    overall_step = int(max(0.0, min(100.0, overall_progress_pct)) // float(progress_step_pct))
    last_step_raw = entry.get("last_step")
    last_step = int(last_step_raw) if last_step_raw is not None else -1
    elapsed = None if last_sent is None else (now_seconds - last_sent)
    step_trigger = overall_step > last_step
    if step_trigger and elapsed is not None and elapsed < min_seconds:
        step_trigger = False
    heartbeat_trigger = last_sent is None or (elapsed is not None and elapsed >= heartbeat_seconds)
    if not step_trigger and not heartbeat_trigger:
        return ToolEventDecision(emit=False, duration_seconds=None)
    signature = _infer_attach_signature(event, progress_step_pct=progress_step_pct)
    if signature == entry.get("last_signature") and not heartbeat_trigger:
        return ToolEventDecision(emit=False, duration_seconds=None)
    per_run[run_key] = {
        "started_at": started_at,
        "last_sent": now_seconds,
        "last_step": max(last_step, overall_step),
        "last_signature": signature,
    }
    return ToolEventDecision(emit=True, duration_seconds=max(0.0, now_seconds - started_at))


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
    for action in ("attach", "write_overlay_part"):
        register_status_override(action, _infer_attach_status_override)
        register_message_override(action, _infer_attach_message)
        register_evaluator(action, _evaluate_infer_attach_event)
    register_status_override("materialize", _infer_materialize_status_override)
