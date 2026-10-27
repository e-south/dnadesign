"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/runtime/test_round_log_summary.py

Regression tests for round log summary OPAL runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json

import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.reporting.summary import summarize_round_log
from dnadesign.opal.src.storage.artifacts import append_round_log_event


def _event(ts: str, stage: str, **payload: object) -> dict[str, object]:
    return {
        "schema_version": "opal.progress_event.v1",
        "event_id": f"event-{ts}-{stage}",
        "phase": "run",
        "severity": "info",
        "ts": ts,
        "stage": stage,
        **payload,
    }


def test_round_log_summary_counts():
    events = [
        _event("2025-01-01T00:00:00+00:00", "start"),
        _event("2025-01-01T00:00:05+00:00", "fit_start"),
        _event("2025-01-01T00:00:06+00:00", "fit"),
        _event("2025-01-01T00:00:07+00:00", "predict_batch", rows=2),
        _event("2025-01-01T00:00:08+00:00", "done"),
    ]
    summary = summarize_round_log(events)
    assert summary["events"] == 5
    assert summary["stage_counts"]["predict_batch"] == 1
    assert summary["predict_rows"] == 2


def test_round_log_summary_latest_run_window():
    events = [
        _event("2025-01-01T00:00:00+00:00", "start"),
        _event("2025-01-01T00:00:10+00:00", "done"),
        _event("2025-01-01T00:01:00+00:00", "start"),
        _event("2025-01-01T00:01:05+00:00", "done"),
    ]
    summary = summarize_round_log(events)
    assert summary["run_count"] == 2
    assert summary["events_total"] == 4
    assert summary["events"] == 2
    assert summary["duration_sec_total"] == 5.0


def test_round_log_summary_filters_by_run_id():
    events = [
        _event("2025-01-01T00:00:00+00:00", "start"),
        _event("2025-01-01T00:00:01+00:00", "run_context", run_id="run-a"),
        _event("2025-01-01T00:00:02+00:00", "predict_batch", run_id="run-a", rows=2),
        _event("2025-01-01T00:00:05+00:00", "done", run_id="run-a"),
        _event("2025-01-01T00:01:01+00:00", "run_context", run_id="run-b"),
        _event("2025-01-01T00:01:03+00:00", "done", run_id="run-b"),
    ]

    summary = summarize_round_log(events, run_id="run-a")

    assert summary["run_id_filter_applied"] is True
    assert summary["events"] == 3
    assert summary["events_total"] == 6
    assert summary["predict_rows"] == 2
    assert summary["duration_sec_total"] == 4.0


def test_round_log_summary_rejects_missing_run_id_when_logs_are_run_scoped():
    events = [_event("2025-01-01T00:00:01+00:00", "done", run_id="run-a")]

    with pytest.raises(OpalError, match="no events for run_id"):
        summarize_round_log(events, run_id="run-b")


def test_round_log_summary_rejects_pre_contract_events():
    events = [{"ts": "2025-01-01T00:00:01+00:00", "stage": "done"}]

    with pytest.raises(OpalError, match="schema_version"):
        summarize_round_log(events)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "opal.progress_event.v0", "schema_version"),
        ("phase", "legacy", "phase must be one of"),
        ("severity", "debug", "severity must be one of"),
        ("ts", "2025-01-01T00:00:01", "must include a UTC offset"),
    ],
)
def test_round_log_summary_rejects_invalid_event_contract(field: str, value: str, message: str):
    event = _event("2025-01-01T00:00:01+00:00", "done")
    event[field] = value

    with pytest.raises(OpalError, match=message):
        summarize_round_log([event])


def test_round_log_summary_rejects_duplicate_event_ids():
    events = [
        _event("2025-01-01T00:00:00+00:00", "start"),
        _event("2025-01-01T00:00:01+00:00", "done"),
    ]
    events[1]["event_id"] = events[0]["event_id"]

    with pytest.raises(OpalError, match="duplicate event_id"):
        summarize_round_log(events)


def test_round_log_event_contract_marks_x_validation_as_preflight(tmp_path):
    log_path = tmp_path / "round.log.jsonl"

    append_round_log_event(
        log_path,
        {
            "ts": "2025-01-01T00:00:00+00:00",
            "round": 0,
            "stage": "x_validate_start",
            "attempt_id": "attempt-1",
        },
    )

    event = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert event["schema_version"] == "opal.progress_event.v1"
    assert event["phase"] == "preflight"
    assert event["attempt_id"] == "attempt-1"

    summary = summarize_round_log([event])
    assert summary["preflight_events"] == 1
    assert summary["run_events"] == 0
    assert summary["run_scope"]["attempt_ids"] == ["attempt-1"]
