"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_round_log_summary.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from dnadesign.opal.src.summary import summarize_round_log


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
