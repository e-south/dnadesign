"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tool_events_densegen.py

DenseGen-specific notify tool-event handlers and gating logic.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .errors import NotifyConfigError
from .tool_event_types import ToolEventDecision, ToolEventState

_DENSEGEN_HEALTH_PROGRESS_STEP_PCT_SMALL_QUOTA = 25
_DENSEGEN_HEALTH_PROGRESS_STEP_PCT_LARGE_QUOTA = 10
_DENSEGEN_HEALTH_SMALL_QUOTA_THRESHOLD = 200
_DENSEGEN_HEALTH_MIN_SECONDS_DEFAULT = 60.0
_DENSEGEN_HEALTH_HEARTBEAT_SECONDS = 1800.0


def _normalize_densegen_status(event: dict[str, Any]) -> str:
    args_raw = event.get("args")
    args = args_raw if isinstance(args_raw, dict) else {}
    return str(args.get("status") or "").strip().lower()


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


def _duration_hhmmss(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _densegen_metrics(event: dict[str, Any], *, required: bool) -> dict[str, Any]:
    metrics_raw = event.get("metrics")
    metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
    densegen_raw = metrics.get("densegen")
    if not isinstance(densegen_raw, dict):
        if required:
            raise NotifyConfigError("densegen_health events require metrics.densegen object")
        return {}
    return densegen_raw


def _densegen_metric_int(densegen_metrics: dict[str, Any], *, key: str) -> int:
    value = _to_int_or_none(densegen_metrics.get(key))
    if value is None:
        raise NotifyConfigError(f"densegen_health metrics.densegen.{key} must be an integer")
    return value


def _densegen_metric_float(densegen_metrics: dict[str, Any], *, key: str) -> float:
    value = _to_float_or_none(densegen_metrics.get(key))
    if value is None:
        raise NotifyConfigError(f"densegen_health metrics.densegen.{key} must be numeric")
    return value


def _resolve_progress_step_pct(*, densegen_metrics: dict[str, Any], notify_config: dict[str, Any]) -> int:
    configured = notify_config.get("progress_step_pct")
    if configured is not None:
        value = _to_int_or_none(configured)
        if value is None or value < 1 or value > 100:
            raise NotifyConfigError("progress_step_pct must be an integer between 1 and 100")
        return value
    run_quota = _to_int_or_none(densegen_metrics.get("run_quota"))
    if run_quota is not None and run_quota <= _DENSEGEN_HEALTH_SMALL_QUOTA_THRESHOLD:
        return _DENSEGEN_HEALTH_PROGRESS_STEP_PCT_SMALL_QUOTA
    return _DENSEGEN_HEALTH_PROGRESS_STEP_PCT_LARGE_QUOTA


def _resolve_progress_min_seconds(*, notify_config: dict[str, Any]) -> float:
    configured = notify_config.get("progress_min_seconds")
    if configured is None:
        return float(_DENSEGEN_HEALTH_MIN_SECONDS_DEFAULT)
    value = _to_float_or_none(configured)
    if value is None or value <= 0.0:
        raise NotifyConfigError("progress_min_seconds must be a positive number")
    return float(value)


def _densegen_health_signature(densegen_metrics: dict[str, Any], *, progress_step_pct: int) -> tuple[Any, ...]:
    quota_progress = _densegen_metric_float(densegen_metrics, key="quota_progress_pct")
    rows_written = _densegen_metric_int(densegen_metrics, key="rows_written_session")
    plans_attempted = _densegen_metric_int(densegen_metrics, key="plans_attempted")
    plans_solved = _densegen_metric_int(densegen_metrics, key="plans_solved")
    tfbs_coverage = _densegen_metric_float(densegen_metrics, key="tfbs_coverage_pct")
    step = int(max(0.0, min(100.0, quota_progress)) // float(progress_step_pct))
    return (step, round(tfbs_coverage, 2), plans_solved, plans_attempted, rows_written)


def _should_emit_densegen_running_health(
    *,
    run_id: str,
    densegen_metrics: dict[str, Any],
    event_ts_seconds: float | None,
    notify_config: dict[str, Any],
    state: dict[str, dict[str, Any]],
) -> bool:
    progress_step_pct = _resolve_progress_step_pct(densegen_metrics=densegen_metrics, notify_config=notify_config)
    progress_min_seconds = _resolve_progress_min_seconds(notify_config=notify_config)
    quota_progress = _densegen_metric_float(densegen_metrics, key="quota_progress_pct")
    quota_step = int(max(0.0, min(100.0, quota_progress)) // float(progress_step_pct))
    now_seconds = float(event_ts_seconds) if event_ts_seconds is not None else float(time.time())
    entry = state.get(run_id) or {}
    last_step_raw = entry.get("last_step")
    last_step = int(last_step_raw) if last_step_raw is not None else -1
    last_sent_raw = entry.get("last_sent")
    last_sent = float(last_sent_raw) if last_sent_raw is not None else None

    step_trigger = quota_step > last_step
    if step_trigger and last_sent is not None and (now_seconds - last_sent) < progress_min_seconds:
        step_trigger = False
    heartbeat_trigger = last_sent is None or (now_seconds - last_sent) >= float(_DENSEGEN_HEALTH_HEARTBEAT_SECONDS)
    if not step_trigger and not heartbeat_trigger:
        return False

    signature = _densegen_health_signature(densegen_metrics, progress_step_pct=progress_step_pct)
    last_signature = entry.get("last_signature")
    if signature == last_signature and not heartbeat_trigger:
        return False

    state[run_id] = {
        "last_step": max(last_step, quota_step),
        "last_sent": now_seconds,
        "last_signature": signature,
    }
    return True


def _densegen_health_status_override(event: dict[str, Any]) -> str | None:
    status = _normalize_densegen_status(event)
    if status in {"completed", "complete", "success", "succeeded"}:
        return "success"
    if status in {"failed", "failure", "error"}:
        return "failure"
    if status in {"started", "start"}:
        return "started"
    if status in {"resumed", "resume"}:
        return "running"
    return None


def _densegen_health_message(
    event: dict[str, Any],
    *,
    run_id: str,
    duration_seconds: float | None,
) -> str:
    dataset_raw = event.get("dataset")
    dataset = dataset_raw if isinstance(dataset_raw, dict) else {}
    dataset_name = str(dataset.get("name") or "unknown-dataset")
    args_raw = event.get("args")
    args = args_raw if isinstance(args_raw, dict) else {}
    status = str(args.get("status") or "running").strip().lower()
    densegen_required = status in {"running", "completed", "complete", "success", "succeeded"}
    densegen_metrics = _densegen_metrics(event, required=densegen_required)

    if status in {"started", "start"}:
        lines = [f"DenseGen started | run={run_id} | dataset={dataset_name}"]
        if densegen_metrics:
            run_quota = _to_int_or_none(densegen_metrics.get("run_quota"))
            if run_quota is not None:
                lines.append(f"- Quota target: {run_quota} rows")
        return "\n".join(lines)

    if status in {"resumed", "resume"}:
        lines = [f"DenseGen resumed | run={run_id} | dataset={dataset_name}"]
        if densegen_metrics:
            quota_progress = _to_float_or_none(densegen_metrics.get("quota_progress_pct"))
            rows_written = _to_int_or_none(densegen_metrics.get("rows_written_session"))
            run_quota = _to_int_or_none(densegen_metrics.get("run_quota"))
            if quota_progress is not None and rows_written is not None and run_quota is not None:
                lines.append(f"- Progress: {quota_progress:.1f}% ({rows_written}/{run_quota} rows)")
        return "\n".join(lines)

    if status in {"failed", "failure", "error"}:
        stage = str(args.get("plan") or args.get("input_name") or "densegen_health")
        error_text = str(args.get("error") or "").strip()
        lines = [f"DenseGen failed | run={run_id} | dataset={dataset_name}"]
        lines.append(f"- Stage: {stage}")
        if error_text:
            lines.append(f"- Error: {error_text}")
        return "\n".join(lines)

    if status in {"completed", "complete", "success", "succeeded"}:
        run_quota = _densegen_metric_int(densegen_metrics, key="run_quota")
        rows_written = _densegen_metric_int(densegen_metrics, key="rows_written_session")
        quota_progress = _densegen_metric_float(densegen_metrics, key="quota_progress_pct")
        tfbs_total = _densegen_metric_int(densegen_metrics, key="tfbs_total_library")
        tfbs_used = _densegen_metric_int(densegen_metrics, key="tfbs_unique_used")
        tfbs_coverage = _densegen_metric_float(densegen_metrics, key="tfbs_coverage_pct")
        plans_attempted = _densegen_metric_int(densegen_metrics, key="plans_attempted")
        plans_solved = _densegen_metric_int(densegen_metrics, key="plans_solved")
        success_pct = (float(plans_solved) / float(plans_attempted) * 100.0) if plans_attempted > 0 else 0.0
        elapsed = _to_float_or_none(densegen_metrics.get("run_elapsed_seconds"))
        if elapsed is None:
            elapsed = duration_seconds
        lines = [f"DenseGen completed | run={run_id} | dataset={dataset_name}"]
        if elapsed is not None:
            lines.append(f"- Duration: {_duration_hhmmss(elapsed)}")
        lines.append(f"- Quota: {quota_progress:.1f}% ({rows_written}/{run_quota} rows)")
        lines.append(f"- Plans: {plans_solved}/{plans_attempted} ({success_pct:.1f}%)")
        lines.append(f"- TFBS coverage: {tfbs_coverage:.1f}% ({tfbs_used}/{tfbs_total})")
        return "\n".join(lines)

    run_quota = _densegen_metric_int(densegen_metrics, key="run_quota")
    rows_written = _densegen_metric_int(densegen_metrics, key="rows_written_session")
    quota_progress = _densegen_metric_float(densegen_metrics, key="quota_progress_pct")
    tfbs_total = _densegen_metric_int(densegen_metrics, key="tfbs_total_library")
    tfbs_used = _densegen_metric_int(densegen_metrics, key="tfbs_unique_used")
    tfbs_coverage = _densegen_metric_float(densegen_metrics, key="tfbs_coverage_pct")
    plans_attempted = _densegen_metric_int(densegen_metrics, key="plans_attempted")
    plans_solved = _densegen_metric_int(densegen_metrics, key="plans_solved")
    success_pct = (float(plans_solved) / float(plans_attempted) * 100.0) if plans_attempted > 0 else 0.0
    elapsed = _to_float_or_none(densegen_metrics.get("run_elapsed_seconds"))
    if elapsed is None:
        elapsed = duration_seconds
    lines = [f"DenseGen progress | run={run_id} | dataset={dataset_name}"]
    lines.append(f"- Quota: {quota_progress:.1f}% ({rows_written}/{run_quota} rows)")
    lines.append(f"- Plan success: {plans_solved}/{plans_attempted} ({success_pct:.1f}%)")
    lines.append(f"- TFBS coverage: {tfbs_coverage:.1f}% ({tfbs_used}/{tfbs_total})")
    if elapsed is not None:
        lines.append(f"- Runtime: {_duration_hhmmss(elapsed)}")
    return "\n".join(lines)


def _densegen_flush_failed_message(
    event: dict[str, Any],
    *,
    run_id: str,
    duration_seconds: float | None,
) -> str:
    del run_id, duration_seconds
    dataset_raw = event.get("dataset")
    dataset = dataset_raw if isinstance(dataset_raw, dict) else {}
    dataset_name = str(dataset.get("name") or "unknown-dataset")
    metrics_raw = event.get("metrics")
    metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
    args_raw = event.get("args")
    args = args_raw if isinstance(args_raw, dict) else {}
    error_type = args.get("error_type")
    error_text = str(args.get("error") or "").strip()
    orphan_count = metrics.get("orphan_artifacts")
    parts = [f"densegen_flush_failed on {dataset_name}"]
    if error_type:
        parts.append(f"error_type={error_type}")
    if error_text:
        parts.append(f"error={error_text}")
    if orphan_count is not None:
        parts.append(f"orphan_artifacts={orphan_count}")
    return " | ".join(parts)


def _evaluate_densegen_health_event(event: dict[str, Any], run_id: str, state: ToolEventState) -> ToolEventDecision:
    bucket = state.get_bucket("densegen_health")
    started_at_raw = bucket.setdefault("started_at", {})
    running_state_raw = bucket.setdefault("running_state", {})
    started_emitted_raw = bucket.setdefault("started_emitted", {})
    terminal_emitted_raw = bucket.setdefault("terminal_emitted", {})
    started_at = started_at_raw if isinstance(started_at_raw, dict) else {}
    running_state = running_state_raw if isinstance(running_state_raw, dict) else {}
    started_emitted = started_emitted_raw if isinstance(started_emitted_raw, dict) else {}
    terminal_emitted = terminal_emitted_raw if isinstance(terminal_emitted_raw, dict) else {}
    notify_config_raw = bucket.get("notify_config")
    notify_config = notify_config_raw if isinstance(notify_config_raw, dict) else {}

    densegen_status = _normalize_densegen_status(event)
    event_ts_seconds = _event_timestamp_seconds(event)
    run_key = str(run_id)
    if densegen_status in {"started", "start"} and event_ts_seconds is not None:
        started_at[run_key] = event_ts_seconds
    elif densegen_status in {"resumed", "resume"} and event_ts_seconds is not None:
        started_at.setdefault(run_key, event_ts_seconds)

    if densegen_status in {"started", "start"}:
        if bool(started_emitted.get(run_key)):
            return ToolEventDecision(emit=False, duration_seconds=None)
        started_emitted[run_key] = True
        terminal_emitted.pop(run_key, None)
        return ToolEventDecision(emit=True, duration_seconds=None)

    if densegen_status in {"resumed", "resume"}:
        return ToolEventDecision(emit=False, duration_seconds=None)

    if densegen_status == "running":
        densegen_metrics = _densegen_metrics(event, required=True)
        if not _should_emit_densegen_running_health(
            run_id=run_key,
            densegen_metrics=densegen_metrics,
            event_ts_seconds=event_ts_seconds,
            notify_config=notify_config,
            state=running_state,
        ):
            return ToolEventDecision(emit=False, duration_seconds=None)

    if densegen_status in {"completed", "complete", "success", "succeeded"}:
        if bool(terminal_emitted.get(run_key)):
            return ToolEventDecision(emit=False, duration_seconds=None)
        densegen_metrics = _densegen_metrics(event, required=True)
        rows_written = _densegen_metric_int(densegen_metrics, key="rows_written_session")
        if rows_written <= 0:
            return ToolEventDecision(emit=False, duration_seconds=None)
        terminal_emitted[run_key] = True
        start_seconds = started_at.get(run_key)
        if start_seconds is not None and event_ts_seconds is not None:
            return ToolEventDecision(emit=True, duration_seconds=max(0.0, float(event_ts_seconds - start_seconds)))
    return ToolEventDecision(emit=True, duration_seconds=None)


def register_densegen_handlers(
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
    register_status_override("densegen_health", _densegen_health_status_override)
    register_message_override("densegen_health", _densegen_health_message)
    register_message_override("densegen_flush_failed", _densegen_flush_failed_message)
    register_evaluator("densegen_health", _evaluate_densegen_health_event)
