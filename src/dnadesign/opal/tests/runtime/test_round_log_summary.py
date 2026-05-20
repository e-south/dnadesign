"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/runtime/test_round_log_summary.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.reporting.summary import summarize_round_log


def test_round_log_summary_counts():
    events = [
        {"ts": "2025-01-01T00:00:00+00:00", "stage": "start"},
        {"ts": "2025-01-01T00:00:05+00:00", "stage": "fit_start"},
        {"ts": "2025-01-01T00:00:06+00:00", "stage": "fit"},
        {"ts": "2025-01-01T00:00:07+00:00", "stage": "predict_batch", "rows": 2},
        {"ts": "2025-01-01T00:00:08+00:00", "stage": "done"},
    ]
    summary = summarize_round_log(events)
    assert summary["events"] == 5
    assert summary["stage_counts"]["predict_batch"] == 1
    assert summary["predict_rows"] == 2


def test_round_log_summary_latest_run_window():
    events = [
        {"ts": "2025-01-01T00:00:00+00:00", "stage": "start"},
        {"ts": "2025-01-01T00:00:10+00:00", "stage": "done"},
        {"ts": "2025-01-01T00:01:00+00:00", "stage": "start"},
        {"ts": "2025-01-01T00:01:05+00:00", "stage": "done"},
    ]
    summary = summarize_round_log(events)
    assert summary["run_count"] == 2
    assert summary["events_total"] == 4
    assert summary["events"] == 2
    assert summary["duration_sec_total"] == 5.0


def test_round_log_summary_filters_by_run_id():
    events = [
        {"ts": "2025-01-01T00:00:00+00:00", "stage": "start"},
        {"ts": "2025-01-01T00:00:01+00:00", "stage": "run_context", "run_id": "run-a"},
        {"ts": "2025-01-01T00:00:02+00:00", "stage": "predict_batch", "run_id": "run-a", "rows": 2},
        {"ts": "2025-01-01T00:00:05+00:00", "stage": "done", "run_id": "run-a"},
        {"ts": "2025-01-01T00:01:01+00:00", "stage": "run_context", "run_id": "run-b"},
        {"ts": "2025-01-01T00:01:03+00:00", "stage": "done", "run_id": "run-b"},
    ]

    summary = summarize_round_log(events, run_id="run-a")

    assert summary["run_id_filter_applied"] is True
    assert summary["events"] == 3
    assert summary["events_total"] == 6
    assert summary["predict_rows"] == 2
    assert summary["duration_sec_total"] == 4.0


def test_round_log_summary_rejects_missing_run_id_when_logs_are_run_scoped():
    events = [{"ts": "2025-01-01T00:00:01+00:00", "stage": "done", "run_id": "run-a"}]

    with pytest.raises(OpalError, match="no events for run_id"):
        summarize_round_log(events, run_id="run-b")
