# ABOUTME: Renders round log summaries for OPAL CLI.
# ABOUTME: Formats log event counts and durations.
"""Round log renderers."""

from __future__ import annotations

from typing import Mapping

from ...tui import kv_table, list_table, tui_enabled
from ..core import bullet_list, kv_block


def render_round_log_summary_human(summary: Mapping[str, object]) -> str:
    if tui_enabled():
        from rich.console import Group

        head_rows = {
            "round": summary.get("round_index"),
            "path": summary.get("path"),
            "events": summary.get("events"),
            "predict_batches": summary.get("predict_batches"),
            "predict_rows": summary.get("predict_rows"),
            "duration_total_s": summary.get("duration_sec_total"),
            "duration_fit_s": summary.get("duration_sec_fit"),
        }
        if (summary.get("run_count") or 0) > 1:
            head_rows["runs_in_log"] = summary.get("run_count")
            head_rows["events_total"] = summary.get("events_total")
        head = kv_table(
            "Round log",
            head_rows,
        )
        stages = summary.get("stage_counts") or {}
        stage_lines = [f"{k}: {v}" for k, v in sorted(stages.items())] if stages else []
        stage_block = list_table("Stages", stage_lines)
        blocks = [head] if head is not None else []
        if stage_block is not None:
            blocks.append(stage_block)
        return Group(*blocks)
    head_rows = {
        "round": summary.get("round_index"),
        "path": summary.get("path"),
        "events": summary.get("events"),
        "predict_batches": summary.get("predict_batches"),
        "predict_rows": summary.get("predict_rows"),
        "duration_total_s": summary.get("duration_sec_total"),
        "duration_fit_s": summary.get("duration_sec_fit"),
    }
    if (summary.get("run_count") or 0) > 1:
        head_rows["runs_in_log"] = summary.get("run_count")
        head_rows["events_total"] = summary.get("events_total")
    head = kv_block("Round log", head_rows)
    stages = summary.get("stage_counts") or {}
    stage_lines = [f"{k}: {v}" for k, v in sorted(stages.items())] if stages else []
    stages_block = bullet_list("Stages", stage_lines)
    return "\n".join([head, "", stages_block])
