"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/tool_events/test_tool_events.py

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


def test_tool_event_status_override_is_none_for_unhandled_actions() -> None:
    event = _event("materialize")
    assert tool_event_status_override("materialize", event) is None


def test_tool_event_status_override_handles_infer_attach_events() -> None:
    event = _event("attach")
    event["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    event["args"] = {"rows_incoming": 10, "rows_matched": 10, "rows_missing": 0}

    assert tool_event_status_override("attach", event) == "running"


def test_tool_event_status_override_handles_infer_write_overlay_part_events() -> None:
    event = _event("write_overlay_part")
    event["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    event["args"] = {"rows_incoming": 10, "rows_matched": 10, "rows_missing": 0}

    assert tool_event_status_override("write_overlay_part", event) == "running"


def test_tool_event_status_override_handles_infer_materialize_as_terminal() -> None:
    event = _event("materialize")
    event["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}

    assert tool_event_status_override("materialize", event) == "success"


def test_tool_event_status_override_handles_infer_feature_sidecar_events() -> None:
    event = _event("infer_feature_vectors_write")
    event["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    event["args"] = {"rows_written": 157164}

    assert tool_event_status_override("infer_feature_vectors_write", event) == "running"


def test_tool_event_status_override_handles_infer_feature_bundle_progress() -> None:
    event = _event("infer_feature_bundle_progress")
    event["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    event["args"] = {"progress_pct": 42.0}

    assert tool_event_status_override("infer_feature_bundle_progress", event) == "running"


def test_tool_event_status_override_handles_infer_feature_bundle_complete_as_terminal() -> None:
    event = _event("infer_feature_bundle_complete")
    event["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}

    assert tool_event_status_override("infer_feature_bundle_complete", event) == "success"


def test_tool_event_message_override_formats_infer_attach_message() -> None:
    event = _event("attach")
    event["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    event["args"] = {
        "rows_incoming": 12,
        "rows_matched": 11,
        "rows_missing": 1,
        "infer_output": {
            "id": "output_layer_mean__seq_mean",
            "family": "output_layer_mean",
            "kind": "feature",
            "pool_scope": "seq_mean",
        },
        "infer_progress": {
            "target_rows": 100,
            "completed_rows": 24,
            "output_progress_pct": 24.0,
            "overall_target_units": 400,
            "overall_completed_units": 72,
            "overall_progress_pct": 18.0,
            "family_progress_pct_map": {
                "log_likelihood": 50.0,
                "output_layer_mean": 12.0,
                "intermediate_embedding": 0.0,
            },
        },
    }

    msg = tool_event_message_override("attach", event, run_id="infer-run-1", duration_seconds=3661.0)

    assert msg is not None
    assert "Infer progress | run=infer-run-1 | dataset=demo" in msg
    assert "- Overall requested outputs: 18.0% (72/400 units)" in msg
    assert "- Remaining overall units: 328" in msg
    assert "- ETA to overall complete: 4.6h" in msg
    assert "- Families: LL 50.0% | output_layer_mean 12.0% | intermediate_embedding 0.0%" in msg
    assert "- Current output: output_layer_mean__seq_mean 24.0% (24/100 rows)" in msg
    assert "incoming=12 matched=11 missing=1" in msg
    assert "- Workspace rows: 1" in msg
    assert "- Elapsed: 01:01:01" in msg


def test_tool_event_message_override_formats_infer_write_overlay_part_message() -> None:
    event = _event("write_overlay_part")
    event["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    event["args"] = {
        "rows_incoming": 12,
        "rows_matched": 11,
        "rows_missing": 1,
        "infer_output": {
            "id": "output_layer_mean__seq_mean",
            "family": "output_layer_mean",
            "kind": "feature",
            "pool_scope": "seq_mean",
        },
        "infer_progress": {
            "target_rows": 100,
            "completed_rows": 24,
            "output_progress_pct": 24.0,
            "overall_target_units": 400,
            "overall_completed_units": 72,
            "overall_progress_pct": 18.0,
            "family_progress_pct_map": {
                "log_likelihood": 50.0,
                "output_layer_mean": 12.0,
                "intermediate_embedding": 0.0,
            },
        },
    }

    msg = tool_event_message_override("write_overlay_part", event, run_id="infer-run-1", duration_seconds=3661.0)

    assert msg is not None
    assert "Infer progress | run=infer-run-1 | dataset=demo" in msg
    assert "- Current output: output_layer_mean__seq_mean 24.0% (24/100 rows)" in msg
    assert "incoming=12 matched=11 missing=1" in msg


def test_tool_event_message_override_explains_grouped_infer_writeback_semantics() -> None:
    event = _event("write_overlay_part")
    event["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    event["args"] = {
        "rows_incoming": 256,
        "rows_matched": 256,
        "rows_missing": 0,
        "infer_progress": {
            "overall_target_units": 502760,
            "overall_completed_units": 218112,
            "overall_progress_pct": 43.38292624711592,
            "family_progress_pct_map": {
                "log_likelihood": 46.3,
                "output_layer_mean": 46.3,
                "intermediate_embedding": 0.0,
            },
        },
    }

    msg = tool_event_message_override("write_overlay_part", event, run_id="infer-run-1", duration_seconds=915.0)

    assert msg is not None
    assert "- Overall requested outputs: 43.4% (218112/502760 units)" in msg
    assert "- Families: LL 46.3% | output_layer_mean 46.3% | intermediate_embedding 0.0%" in msg
    assert (
        "- Grouped writeback: family percentages reflect persisted rows still missing each family, "
        "not per-family GPU compute"
    ) in msg
    assert "- Current output:" not in msg
    assert "- ETA to overall complete: 0.3h" in msg
    assert "incoming=256 matched=256 missing=0" in msg
    assert "- Elapsed: 00:15:15" in msg


def test_tool_event_message_override_formats_infer_feature_sidecar_message() -> None:
    event = _event("infer_feature_vectors_write")
    event["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    event["args"] = {"rows_written": 157164}
    event["artifacts"] = {"path": "_derived/infer/feature_vectors.parquet"}

    msg = tool_event_message_override(
        "infer_feature_vectors_write",
        event,
        run_id="infer-run-1",
        duration_seconds=None,
    )

    assert msg is not None
    assert "Infer feature sidecar write" in msg
    assert "Rows written: 157164" in msg
    assert "_derived/infer/feature_vectors.parquet" in msg


def test_tool_event_message_override_formats_infer_feature_bundle_progress_message() -> None:
    event = _event("infer_feature_bundle_progress")
    event["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    event["args"] = {
        "job_id": "context_forward_seq_and_anchor_mean_7b",
        "model_id": "evo2_7b",
        "contexts_completed": 78640,
        "contexts_total": 157279,
        "progress_pct": 50.0,
        "unique_forward_passes_completed": 78640,
        "unique_forward_passes_total": 157279,
        "required_vector_keys": 629116,
        "required_scalar_keys": 314558,
    }

    msg = tool_event_message_override(
        "infer_feature_bundle_progress",
        event,
        run_id="infer-run-1",
        duration_seconds=1800.0,
    )

    assert msg is not None
    assert "Infer feature bundle 50.0% | job=context_forward_seq_and_anchor_mean_7b | dataset=demo" in msg
    assert "- Model: evo2_7b" in msg
    assert "- Views this run: 78640/157279 | forward passes: 78640/157279" in msg
    assert "- ETA: 00:30:00" in msg
    assert "- Sidecar keys: vectors=629116 scalars=314558" in msg
    assert "- Elapsed: 00:30:00" in msg


def test_tool_event_message_override_formats_infer_feature_bundle_complete_message() -> None:
    event = _event("infer_feature_bundle_complete")
    event["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    event["args"] = {
        "job_id": "context_forward_seq_and_anchor_mean_7b",
        "model_id": "evo2_7b",
        "contexts_completed": 314558,
        "unique_forward_passes": 157279,
        "required_vector_keys": 629116,
        "required_scalar_keys": 314558,
    }

    msg = tool_event_message_override(
        "infer_feature_bundle_complete",
        event,
        run_id="infer-run-1",
        duration_seconds=None,
    )

    assert msg is not None
    assert "Infer feature bundle complete" in msg
    assert "job=context_forward_seq_and_anchor_mean_7b" in msg
    assert "- Forward passes: 157279" in msg
    assert "- Sidecar keys: vectors=629116 scalars=314558" in msg


def test_evaluate_tool_event_infer_attach_exposes_elapsed_duration() -> None:
    state = ToolEventState()
    first = _event("attach", timestamp="2026-02-06T00:00:00+00:00")
    first["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    first["args"] = {
        "rows_incoming": 8,
        "rows_matched": 8,
        "rows_missing": 0,
        "infer_output": {"id": "log_likelihood__total", "family": "log_likelihood", "kind": "feature"},
        "infer_progress": {
            "target_rows": 100,
            "completed_rows": 8,
            "output_progress_pct": 8.0,
            "overall_target_units": 300,
            "overall_completed_units": 8,
            "overall_progress_pct": 2.7,
        },
    }
    second = _event("attach", timestamp="2026-02-06T00:01:10+00:00")
    second["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    second["args"] = {
        "rows_incoming": 8,
        "rows_matched": 8,
        "rows_missing": 0,
        "infer_output": {"id": "log_likelihood__total", "family": "log_likelihood", "kind": "feature"},
        "infer_progress": {
            "target_rows": 100,
            "completed_rows": 32,
            "output_progress_pct": 32.0,
            "overall_target_units": 100,
            "overall_completed_units": 32,
            "overall_progress_pct": 32.0,
        },
    }

    first_decision = evaluate_tool_event("attach", first, run_id="infer-run-1", state=state)
    second_decision = evaluate_tool_event("attach", second, run_id="infer-run-1", state=state)

    assert first_decision.emit is True
    assert first_decision.duration_seconds == 0.0
    assert second_decision.emit is True
    assert second_decision.duration_seconds == 70.0


def test_evaluate_tool_event_infer_write_overlay_part_exposes_elapsed_duration() -> None:
    state = ToolEventState()
    first = _event("write_overlay_part", timestamp="2026-02-06T00:00:00+00:00")
    first["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    first["args"] = {
        "rows_incoming": 8,
        "rows_matched": 8,
        "rows_missing": 0,
        "infer_output": {"id": "log_likelihood__total", "family": "log_likelihood", "kind": "feature"},
        "infer_progress": {
            "target_rows": 100,
            "completed_rows": 8,
            "output_progress_pct": 8.0,
            "overall_target_units": 300,
            "overall_completed_units": 8,
            "overall_progress_pct": 2.7,
        },
    }
    second = _event("write_overlay_part", timestamp="2026-02-06T00:01:10+00:00")
    second["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    second["args"] = {
        "rows_incoming": 8,
        "rows_matched": 8,
        "rows_missing": 0,
        "infer_output": {"id": "log_likelihood__total", "family": "log_likelihood", "kind": "feature"},
        "infer_progress": {
            "target_rows": 100,
            "completed_rows": 32,
            "output_progress_pct": 32.0,
            "overall_target_units": 100,
            "overall_completed_units": 32,
            "overall_progress_pct": 32.0,
        },
    }

    first_decision = evaluate_tool_event("write_overlay_part", first, run_id="infer-run-1", state=state)
    second_decision = evaluate_tool_event("write_overlay_part", second, run_id="infer-run-1", state=state)

    assert first_decision.emit is True
    assert first_decision.duration_seconds == 0.0
    assert second_decision.emit is True
    assert second_decision.duration_seconds == 70.0


def test_evaluate_tool_event_infer_feature_bundle_progress_is_step_throttled() -> None:
    state = ToolEventState()
    first = _event("infer_feature_bundle_progress", timestamp="2026-02-06T00:00:00+00:00")
    first["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    first["args"] = {
        "contexts_completed": 1,
        "contexts_total": 1000,
        "progress_pct": 0.1,
    }
    second = _event("infer_feature_bundle_progress", timestamp="2026-02-06T00:00:30+00:00")
    second["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    second["args"] = {
        "contexts_completed": 90,
        "contexts_total": 1000,
        "progress_pct": 9.0,
    }
    third = _event("infer_feature_bundle_progress", timestamp="2026-02-06T00:01:10+00:00")
    third["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    third["args"] = {
        "contexts_completed": 110,
        "contexts_total": 1000,
        "progress_pct": 11.0,
    }

    first_decision = evaluate_tool_event("infer_feature_bundle_progress", first, run_id="infer-run-1", state=state)
    second_decision = evaluate_tool_event("infer_feature_bundle_progress", second, run_id="infer-run-1", state=state)
    third_decision = evaluate_tool_event("infer_feature_bundle_progress", third, run_id="infer-run-1", state=state)

    assert first_decision.emit is True
    assert first_decision.duration_seconds == 0.0
    assert second_decision.emit is False
    assert third_decision.emit is True
    assert third_decision.duration_seconds == 70.0


def test_evaluate_tool_event_infer_feature_bundle_complete_exposes_duration() -> None:
    state = ToolEventState()
    progress = _event("infer_feature_bundle_progress", timestamp="2026-02-06T00:00:00+00:00")
    progress["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    progress["args"] = {
        "contexts_completed": 1,
        "contexts_total": 100,
        "progress_pct": 1.0,
    }
    complete = _event("infer_feature_bundle_complete", timestamp="2026-02-06T00:12:34+00:00")
    complete["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}

    _ = evaluate_tool_event("infer_feature_bundle_progress", progress, run_id="infer-run-1", state=state)
    decision = evaluate_tool_event("infer_feature_bundle_complete", complete, run_id="infer-run-1", state=state)

    assert decision.emit is True
    assert decision.duration_seconds == 754.0


def test_evaluate_tool_event_infer_attach_is_throttled_by_min_seconds() -> None:
    state = ToolEventState()
    first = _event("attach", timestamp="2026-02-06T00:00:00+00:00")
    first["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    first["args"] = {
        "rows_incoming": 20,
        "rows_matched": 20,
        "rows_missing": 0,
        "infer_output": {"id": "log_likelihood__total", "family": "log_likelihood", "kind": "feature"},
        "infer_progress": {
            "target_rows": 100,
            "completed_rows": 10,
            "output_progress_pct": 10.0,
            "overall_target_units": 100,
            "overall_completed_units": 10,
            "overall_progress_pct": 10.0,
        },
    }
    second = _event("attach", timestamp="2026-02-06T00:00:10+00:00")
    second["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    second["args"] = {
        "rows_incoming": 20,
        "rows_matched": 20,
        "rows_missing": 0,
        "infer_output": {"id": "log_likelihood__total", "family": "log_likelihood", "kind": "feature"},
        "infer_progress": {
            "target_rows": 100,
            "completed_rows": 19,
            "output_progress_pct": 19.0,
            "overall_target_units": 100,
            "overall_completed_units": 19,
            "overall_progress_pct": 19.0,
        },
    }
    third = _event("attach", timestamp="2026-02-06T00:01:10+00:00")
    third["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    third["args"] = {
        "rows_incoming": 20,
        "rows_matched": 20,
        "rows_missing": 0,
        "infer_output": {"id": "log_likelihood__total", "family": "log_likelihood", "kind": "feature"},
        "infer_progress": {
            "target_rows": 100,
            "completed_rows": 30,
            "output_progress_pct": 30.0,
            "overall_target_units": 100,
            "overall_completed_units": 30,
            "overall_progress_pct": 30.0,
        },
    }

    first_decision = evaluate_tool_event("attach", first, run_id="infer-run-1", state=state)
    second_decision = evaluate_tool_event("attach", second, run_id="infer-run-1", state=state)
    third_decision = evaluate_tool_event("attach", third, run_id="infer-run-1", state=state)

    assert first_decision.emit is True
    assert second_decision.emit is False
    assert third_decision.emit is True


def test_tool_event_status_override_suppresses_infer_metadata_attach_events() -> None:
    event = _event("attach")
    event["actor"] = {"tool": "infer", "run_id": "infer-run-1", "host": "host", "pid": 123}
    event["args"] = {
        "rows_incoming": 12,
        "rows_matched": 12,
        "rows_missing": 0,
        "infer_output": {"id": "metadata__sequence_id", "family": "metadata", "kind": "metadata"},
        "infer_notify_suppress": True,
    }

    assert tool_event_status_override("attach", event) is None
    assert tool_event_message_override("attach", event, run_id="infer-run-1", duration_seconds=None) is None
    assert evaluate_tool_event("attach", event, run_id="infer-run-1", state=ToolEventState()).emit is False
