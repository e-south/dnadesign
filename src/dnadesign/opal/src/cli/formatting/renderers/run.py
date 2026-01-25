"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/formatting/renderers/run.py

Renders run-related command output for OPAL CLI. Formats run summaries and
run metadata displays.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

from ...tui import kv_table, tui_enabled
from ..core import _b, _dim, kv_block


def render_run_summary_human(summary: dict) -> str:
    rid = summary.get("run_id", "")
    requested = summary.get("top_k_requested")
    effective = summary.get("top_k_effective")
    tie_hint = summary.get("tie_handling") or "competition_rank"
    obj_mode = summary.get("objective_mode") or "maximize"
    sel_line = (
        "selection: "
        f"objective={obj_mode} tie={tie_hint} | "
        f"top_k={requested} (requested) → selected={effective} (effective after ties)"
    )
    if tui_enabled():
        table = kv_table(
            "Run summary",
            {
                "run_id": rid,
                "as_of_round": summary.get("as_of_round"),
                "trained_on": summary.get("trained_on"),
                "scored": summary.get("scored"),
                "selection": f"objective={obj_mode} tie={tie_hint} | top_k={requested} -> selected={effective}",
                "ledger": summary.get("ledger"),
                "top_k_source": summary.get("top_k_source"),
            },
        )
        if table is not None:
            return table
    lines = [
        f"{_b('run_id')}: {rid}",
        f"{_b('as_of_round')}: {summary.get('as_of_round')}",
        f"{_b('trained_on')}:{' '}{summary.get('trained_on')} | {_b('scored')}:{' '}{summary.get('scored')}",
        sel_line,
        f"{_b('ledger')}: {summary.get('ledger')}",
        f"{_b('top_k_source')}: {summary.get('top_k_source')}",
    ]
    return "\n".join(lines)


def render_run_meta_human(row: Mapping[str, Any]) -> str:
    y_ops = row.get("training__y_ops") or []
    y_ops_str = ", ".join([p.get("name") for p in y_ops]) if y_ops else "(none)"
    if tui_enabled():
        from rich.console import Group

        head = kv_table(
            "Run",
            {
                "run_id": row.get("run_id"),
                "as_of_round": row.get("as_of_round"),
                "model": row.get("model__name"),
                "objective": row.get("objective__name"),
                "selection": row.get("selection__name"),
                "y_ops": y_ops_str,
                "n_train": row.get("stats__n_train"),
                "n_scored": row.get("stats__n_scored"),
            },
        )
        blocks = [head] if head is not None else []
        obj_stats = row.get("objective__summary_stats") or {}
        if obj_stats:
            stats_block = kv_table("Objective summary", obj_stats)
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
            "objective": row.get("objective__name"),
            "selection": row.get("selection__name"),
            "y_ops": y_ops_str,
            "n_train": row.get("stats__n_train"),
            "n_scored": row.get("stats__n_scored"),
        },
    )
    obj_stats = row.get("objective__summary_stats") or {}
    stats_block = kv_block("Objective summary", obj_stats) if obj_stats else _dim("No objective summary stats.")
    artifacts = row.get("artifacts") or {}
    artifacts_block = kv_block("Artifacts", artifacts) if artifacts else _dim("No artifacts recorded.")
    return "\n".join([head, "", stats_block, "", artifacts_block])
