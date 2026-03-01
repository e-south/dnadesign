"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tool_events/densegen_messages.py

DenseGen tool-event status and message rendering helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from .densegen_common import _duration_hhmmss, _normalize_densegen_status, _to_float_or_none, _to_int_or_none
from .densegen_metrics import _densegen_metric_float, _densegen_metric_int, _densegen_metrics


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
    densegen_data = _densegen_metrics(event, required=densegen_required)

    if status in {"started", "start"}:
        lines = [f"DenseGen started | run={run_id} | dataset={dataset_name}"]
        if densegen_data:
            run_quota = _to_int_or_none(densegen_data.get("run_quota"))
            if run_quota is not None:
                lines.append(f"- Quota target: {run_quota} rows")
        return "\n".join(lines)

    if status in {"resumed", "resume"}:
        lines = [f"DenseGen resumed | run={run_id} | dataset={dataset_name}"]
        if densegen_data:
            quota_progress = _to_float_or_none(densegen_data.get("quota_progress_pct"))
            rows_written = _to_int_or_none(densegen_data.get("rows_written_session"))
            run_quota = _to_int_or_none(densegen_data.get("run_quota"))
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
        run_quota = _densegen_metric_int(densegen_data, key="run_quota")
        rows_written = _densegen_metric_int(densegen_data, key="rows_written_session")
        quota_progress = _densegen_metric_float(densegen_data, key="quota_progress_pct")
        tfbs_total = _densegen_metric_int(densegen_data, key="tfbs_total_library")
        tfbs_used = _densegen_metric_int(densegen_data, key="tfbs_unique_used")
        tfbs_coverage = _densegen_metric_float(densegen_data, key="tfbs_coverage_pct")
        plans_attempted = _densegen_metric_int(densegen_data, key="plans_attempted")
        plans_solved = _densegen_metric_int(densegen_data, key="plans_solved")
        success_pct = (float(plans_solved) / float(plans_attempted) * 100.0) if plans_attempted > 0 else 0.0
        elapsed = _to_float_or_none(densegen_data.get("run_elapsed_seconds"))
        if elapsed is None:
            elapsed = duration_seconds
        lines = [f"DenseGen completed | run={run_id} | dataset={dataset_name}"]
        if elapsed is not None:
            lines.append(f"- Duration: {_duration_hhmmss(elapsed)}")
        lines.append(f"- Quota: {quota_progress:.1f}% ({rows_written}/{run_quota} rows)")
        lines.append(f"- Plans: {plans_solved}/{plans_attempted} ({success_pct:.1f}%)")
        lines.append(f"- TFBS coverage: {tfbs_coverage:.1f}% ({tfbs_used}/{tfbs_total})")
        return "\n".join(lines)

    run_quota = _densegen_metric_int(densegen_data, key="run_quota")
    rows_written = _densegen_metric_int(densegen_data, key="rows_written_session")
    quota_progress = _densegen_metric_float(densegen_data, key="quota_progress_pct")
    tfbs_total = _densegen_metric_int(densegen_data, key="tfbs_total_library")
    tfbs_used = _densegen_metric_int(densegen_data, key="tfbs_unique_used")
    tfbs_coverage = _densegen_metric_float(densegen_data, key="tfbs_coverage_pct")
    plans_attempted = _densegen_metric_int(densegen_data, key="plans_attempted")
    plans_solved = _densegen_metric_int(densegen_data, key="plans_solved")
    success_pct = (float(plans_solved) / float(plans_attempted) * 100.0) if plans_attempted > 0 else 0.0
    elapsed = _to_float_or_none(densegen_data.get("run_elapsed_seconds"))
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
