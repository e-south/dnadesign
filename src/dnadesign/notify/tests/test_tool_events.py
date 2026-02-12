"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_tool_events.py

Tool-event handler tests for notify tool-agnostic dispatch and run-state logic.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.notify.tool_events import (
    ToolEventState,
    evaluate_tool_event,
    tool_event_message_override,
    tool_event_status_override,
)


def _event(action: str, *, status: str | None = None, timestamp: str = "2026-02-06T00:00:00+00:00") -> dict:
    args: dict[str, str] = {}
    if status is not None:
        args["status"] = status
    return {
        "event_version": 1,
        "timestamp_utc": timestamp,
        "action": action,
        "dataset": {"name": "demo", "root": "/tmp/datasets"},
        "args": args,
        "metrics": {"rows_written": 3},
        "artifacts": {"overlay": {"namespace": "densegen"}},
        "fingerprint": {"rows": 1, "cols": 2, "size_bytes": 128, "sha256": None},
        "registry_hash": "abc123",
        "actor": {"tool": "densegen", "run_id": "run-1", "host": "host", "pid": 123},
        "version": "0.1.0",
    }


def _densegen_metrics(
    *,
    run_quota: int = 100,
    rows_written_session: int = 10,
    quota_progress_pct: float = 10.0,
    tfbs_total_library: int = 80,
    tfbs_unique_used: int = 20,
    tfbs_coverage_pct: float = 25.0,
    plans_attempted: int = 10,
    plans_solved: int = 8,
    run_elapsed_seconds: float = 12.0,
) -> dict:
    return {
        "densegen": {
            "run_quota": run_quota,
            "rows_written_session": rows_written_session,
            "quota_progress_pct": quota_progress_pct,
            "tfbs_total_library": tfbs_total_library,
            "tfbs_unique_used": tfbs_unique_used,
            "tfbs_coverage_pct": tfbs_coverage_pct,
            "plans_attempted": plans_attempted,
            "plans_solved": plans_solved,
            "run_elapsed_seconds": run_elapsed_seconds,
        }
    }


def test_tool_event_status_override_handles_densegen_health_statuses() -> None:
    event_completed = _event("densegen_health", status="completed")
    event_failed = _event("densegen_health", status="failed")
    event_started = _event("densegen_health", status="started")
    event_resumed = _event("densegen_health", status="resumed")

    assert tool_event_status_override("densegen_health", event_completed) == "success"
    assert tool_event_status_override("densegen_health", event_failed) == "failure"
    assert tool_event_status_override("densegen_health", event_started) == "started"
    assert tool_event_status_override("densegen_health", event_resumed) == "running"


def test_tool_event_status_override_is_none_for_unhandled_actions() -> None:
    event = _event("materialize")
    assert tool_event_status_override("materialize", event) is None


def test_evaluate_tool_event_densegen_running_is_gated_by_progress_step() -> None:
    state = ToolEventState()
    first = _event("densegen_health", status="running")
    first["metrics"] = _densegen_metrics(quota_progress_pct=10.0)
    second = _event("densegen_health", status="running", timestamp="2026-02-06T00:00:10+00:00")
    second["metrics"] = _densegen_metrics(quota_progress_pct=10.1)

    first_decision = evaluate_tool_event("densegen_health", first, run_id="run-1", state=state)
    second_decision = evaluate_tool_event("densegen_health", second, run_id="run-1", state=state)

    assert first_decision.emit is True
    assert second_decision.emit is False


def test_evaluate_tool_event_densegen_completion_exposes_duration() -> None:
    state = ToolEventState()
    started = _event("densegen_health", status="started", timestamp="2026-02-06T00:00:00+00:00")
    started["metrics"] = _densegen_metrics(
        quota_progress_pct=0.0,
        rows_written_session=0,
        plans_attempted=0,
        plans_solved=0,
    )
    completed = _event("densegen_health", status="completed", timestamp="2026-02-06T00:00:30+00:00")
    completed["metrics"] = _densegen_metrics(
        quota_progress_pct=100.0,
        rows_written_session=100,
        plans_attempted=10,
        plans_solved=10,
    )

    _ = evaluate_tool_event("densegen_health", started, run_id="run-1", state=state)
    decision = evaluate_tool_event("densegen_health", completed, run_id="run-1", state=state)

    assert decision.emit is True
    assert decision.duration_seconds == 30.0


def test_tool_event_message_override_formats_densegen_health_message() -> None:
    event = _event("densegen_health", status="running")
    event["metrics"] = _densegen_metrics()

    msg = tool_event_message_override("densegen_health", event, run_id="run-1", duration_seconds=12.0)

    assert msg is not None
    assert "DenseGen health | run=run-1 | dataset=demo" in msg
    assert "- Quota: 10.0% (10/100 rows)" in msg


def test_tool_event_message_override_formats_densegen_flush_failed_message() -> None:
    event = _event("densegen_flush_failed")
    event["args"] = {"error_type": "OSError", "error": "disk quota exceeded"}
    event["metrics"] = {"orphan_artifacts": 2}

    msg = tool_event_message_override("densegen_flush_failed", event, run_id="run-1", duration_seconds=None)

    assert msg == "densegen_flush_failed on demo | error_type=OSError | error=disk quota exceeded | orphan_artifacts=2"
