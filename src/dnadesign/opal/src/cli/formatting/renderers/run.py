"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/formatting/renderers/run.py

Renders run-related command output for OPAL CLI. Formats run summaries and.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ...tui import kv_table, tui_enabled
from ..core import _b, _dim, kv_block


def _required_summary_field(summary: dict, key: str) -> str:
    raw = summary.get(key)
    if raw is None:
        raise ValueError(f"run summary missing required field: {key}")
    val = str(raw).strip()
    if not val:
        raise ValueError(f"run summary field must be non-empty: {key}")
    return val


def render_run_summary_text(summary: dict) -> str:
    rid = summary.get("run_id", "")
    views = summary.get("selection_views")
    if not isinstance(views, dict) or not views:
        raise ValueError("run summary missing required field: selection_views")
    selection_lines = []
    for view_id, view_summary in views.items():
        tie_handling = _required_summary_field(view_summary, "tie_handling")
        objective_mode = _required_summary_field(view_summary, "objective_mode")
        selection_lines.append(
            f"{view_id}: objective={objective_mode} tie={tie_handling} | "
            f"top_k={view_summary.get('top_k_requested')} -> "
            f"selected={view_summary.get('top_k_effective')}"
        )
    if tui_enabled():
        table = kv_table(
            "Run summary",
            {
                "run_id": rid,
                "as_of_round": summary.get("as_of_round"),
                "trained_on": summary.get("trained_on"),
                "scored": summary.get("scored"),
                "selection views": "; ".join(selection_lines),
                "selection batch": summary.get("selection_batch_count"),
                "ledger": summary.get("ledger"),
            },
        )
        if table is not None:
            return table
    lines = [
        f"{_b('run_id')}: {rid}",
        f"{_b('as_of_round')}: {summary.get('as_of_round')}",
        f"{_b('trained_on')}:{' '}{summary.get('trained_on')} | {_b('scored')}:{' '}{summary.get('scored')}",
        f"{_b('selection views')}: " + "; ".join(selection_lines),
        f"{_b('selection batch')}: {summary.get('selection_batch_count')}",
        f"{_b('ledger')}: {summary.get('ledger')}",
    ]
    return "\n".join(lines)


def render_run_meta_text(row: Mapping[str, Any]) -> str:
    y_ops = row.get("training__y_ops") or []
    y_ops_str = ", ".join([p.get("name") for p in y_ops]) if y_ops else "(none)"
    raw_views = row.get("selection_views__defs_json") or "[]"
    views = raw_views if isinstance(raw_views, list) else json.loads(str(raw_views))
    view_summary = ", ".join(str(view.get("selection_view_id")) for view in views) or "(none)"
    if tui_enabled():
        from rich.console import Group

        head = kv_table(
            "Run",
            {
                "run_id": row.get("run_id"),
                "as_of_round": row.get("as_of_round"),
                "model": row.get("model__name"),
                "selection views": view_summary,
                "y_ops": y_ops_str,
                "n_train": row.get("stats__n_train"),
                "n_scored": row.get("stats__n_scored"),
            },
        )
        blocks = [head] if head is not None else []
        for view in views:
            stats = view.get("objective_summary_stats") or {}
            if stats:
                stats_block = kv_table(f"Objective summary: {view.get('selection_view_id')}", stats)
                if stats_block is not None:
                    blocks.append(stats_block)
        artifacts = row.get("artifacts") or {}
        if artifacts:
            artifacts_block = kv_table("Artifacts", artifacts)
            if artifacts_block is not None:
                blocks.append(artifacts_block)
        return Group(*blocks)
    head = kv_block(
        "Run",
        {
            "run_id": row.get("run_id"),
            "as_of_round": row.get("as_of_round"),
            "model": row.get("model__name"),
            "selection views": view_summary,
            "y_ops": y_ops_str,
            "n_train": row.get("stats__n_train"),
            "n_scored": row.get("stats__n_scored"),
        },
    )
    summaries = {
        str(view.get("selection_view_id")): view.get("objective_summary_stats") or {}
        for view in views
        if view.get("objective_summary_stats")
    }
    stats_block = kv_block("Objective summaries", summaries) if summaries else _dim("No objective summary stats.")
    artifacts = row.get("artifacts") or {}
    artifacts_block = kv_block("Artifacts", artifacts) if artifacts else _dim("No artifacts recorded.")
    return "\n".join([head, "", stats_block, "", artifacts_block])
