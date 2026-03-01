"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tool_events/densegen_eval.py

DenseGen tool-event evaluation state machine for notify emission gating.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from .densegen_common import _event_timestamp_seconds, _normalize_densegen_status
from .densegen_metrics import _densegen_metric_int, _densegen_metrics, _should_emit_densegen_running_health
from .types import ToolEventDecision, ToolEventState


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
        densegen_data = _densegen_metrics(event, required=True)
        if not _should_emit_densegen_running_health(
            run_id=run_key,
            densegen_metrics=densegen_data,
            event_ts_seconds=event_ts_seconds,
            notify_config=notify_config,
            state=running_state,
        ):
            return ToolEventDecision(emit=False, duration_seconds=None)

    if densegen_status in {"completed", "complete", "success", "succeeded"}:
        if bool(terminal_emitted.get(run_key)):
            return ToolEventDecision(emit=False, duration_seconds=None)
        densegen_data = _densegen_metrics(event, required=True)
        rows_written = _densegen_metric_int(densegen_data, key="rows_written_session")
        if rows_written <= 0:
            return ToolEventDecision(emit=False, duration_seconds=None)
        terminal_emitted[run_key] = True
        start_seconds = started_at.get(run_key)
        if start_seconds is not None and event_ts_seconds is not None:
            return ToolEventDecision(emit=True, duration_seconds=max(0.0, float(event_ts_seconds - start_seconds)))
    return ToolEventDecision(emit=True, duration_seconds=None)
