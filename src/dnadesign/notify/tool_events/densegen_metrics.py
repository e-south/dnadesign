"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tool_events/densegen_metrics.py

DenseGen tool-event metric extraction and running-health gating helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from typing import Any

from ..errors import NotifyConfigError
from .densegen_common import _to_float_or_none, _to_int_or_none

_DENSEGEN_HEALTH_PROGRESS_STEP_PCT_SMALL_QUOTA = 25
_DENSEGEN_HEALTH_PROGRESS_STEP_PCT_LARGE_QUOTA = 10
_DENSEGEN_HEALTH_SMALL_QUOTA_THRESHOLD = 200
_DENSEGEN_HEALTH_MIN_SECONDS_DEFAULT = 60.0
_DENSEGEN_HEALTH_HEARTBEAT_SECONDS = 1800.0


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
