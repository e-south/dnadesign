# ABOUTME: Renders explain command output for OPAL CLI.
# ABOUTME: Formats explain summaries for human-readable display.
"""Explain command renderers."""

from __future__ import annotations

from typing import Any, Mapping

from ...tui import kv_table, list_table, tui_enabled
from ..core import _fmt_params, kv_block


def render_explain_human(info: Mapping[str, Any]) -> str:
    sel = info.get("selection", {}) or {}
    obj = sel.get("objective", {}) or {}
    yops = info.get("training_y_ops") or info.get("training_y_ops", []) or []
    yops_str = ", ".join(str(p.get("name")) for p in yops) if yops else "(none)"

    if tui_enabled():
        from rich.console import Group

        blocks = []
        head = kv_table(
            f"Round r={info.get('round_index')}",
            {
                "X column": info.get("x_column_name"),
                "Y column": info.get("y_column_name"),
                "Vector dim (X)": info.get("representation_vector_dimension"),
            },
        )
        if head is not None:
            blocks.append(head)
        model_block = kv_table(
            "Model",
            {
                "name": (info.get("model") or {}).get("name"),
                "params": _fmt_params((info.get("model") or {}).get("params")),
                "Y-ops": yops_str,
            },
        )
        if model_block is not None:
            blocks.append(model_block)
        selection_block = kv_table(
            "Selection & Objective",
            {
                "strategy": sel.get("strategy"),
                "selection.params": _fmt_params(sel.get("params")),
                "objective": obj.get("name"),
                "objective.params": _fmt_params(obj.get("params")),
            },
        )
        if selection_block is not None:
            blocks.append(selection_block)
        counts_block = kv_table(
            "Counts",
            {
                "training labels used": info.get("number_of_training_examples_used_in_round"),
                "candidates scored": info.get("number_of_candidates_scored_in_round"),
                "candidate pool total": info.get("candidate_pool_total"),
                "candidate pool filtered out": info.get("candidate_pool_filtered_out"),
            },
        )
        if counts_block is not None:
            blocks.append(counts_block)
        warn_rows = [str(w) for w in (info.get("warnings") or [])]
        warn_block = list_table("Warnings", warn_rows)
        if warn_block is not None:
            blocks.append(warn_block)
        return Group(*blocks)

    header = kv_block(
        f"Round r={info.get('round_index')}",
        {
            "X column": info.get("x_column_name"),
            "Y column": info.get("y_column_name"),
            "Vector dim (X)": info.get("representation_vector_dimension"),
        },
    )

    model_block = kv_block(
        "Model",
        {
            "name": (info.get("model") or {}).get("name"),
            "params": _fmt_params((info.get("model") or {}).get("params")),
            "Y-ops": yops_str,
        },
    )

    selection_block = kv_block(
        "Selection & Objective",
        {
            "strategy": sel.get("strategy"),
            "selection.params": _fmt_params(sel.get("params")),
            "objective": obj.get("name"),
            "objective.params": _fmt_params(obj.get("params")),
        },
    )

    counts_block = kv_block(
        "Counts",
        {
            "training labels used": info.get("number_of_training_examples_used_in_round"),
            "candidates scored": info.get("number_of_candidates_scored_in_round"),
            "candidate pool total": info.get("candidate_pool_total"),
            "candidate pool filtered out": info.get("candidate_pool_filtered_out"),
        },
    )

    warn_rows = [str(w) for w in (info.get("warnings") or [])]
    warn_block = ""
    if warn_rows:
        warn_block = "\n" + "\n".join([f"  - {w}" for w in warn_rows])

    out = "\n\n".join([header, model_block, selection_block, counts_block])
    if warn_rows:
        out += "\n\nWarnings:" + warn_block
    return out
