"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/formatting/renderers/status.py

Renders status command output for OPAL CLI. Formats campaign status summaries
and round details.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

from ...tui import kv_table, list_table, tui_enabled
from ..core import _dim, kv_block


def render_status_text(st: Mapping[str, Any]) -> str:
    if tui_enabled():
        from rich.console import Group

        blocks = []
        head = kv_table(
            "Campaign",
            {
                "name": st.get("campaign_name"),
                "slug": st.get("campaign_slug"),
                "workdir": st.get("workdir"),
                "X column": st.get("x_column_name"),
                "Y column": st.get("y_column_name"),
                "num_rounds": st.get("num_rounds"),
                "prediction records": (st.get("writeback") or {}).get("prediction_records"),
            },
        )
        if head is not None:
            blocks.append(head)

        data = st.get("data") or {}
        if data:
            data_block = kv_table(
                "Data",
                {
                    "records": data.get("records_path"),
                    "records exists": data.get("records_exists"),
                    "rows": data.get("row_count"),
                    "location": data.get("location_kind"),
                },
            )
            if data_block is not None:
                blocks.append(data_block)

        label_source = st.get("label_source") or {}
        if label_source:
            label_kv = {
                "kind": label_source.get("kind"),
                "exists": label_source.get("exists"),
                "valid": label_source.get("valid"),
                "label_count": label_source.get("label_count"),
                "available_rounds": ", ".join(map(str, label_source.get("available_rounds") or [])) or "(none)",
            }
            if label_source.get("path"):
                label_kv["path"] = label_source.get("path")
            if label_source.get("column"):
                label_kv["column"] = label_source.get("column")
            if label_source.get("error"):
                label_kv["error"] = label_source.get("error")
            label_block = kv_table("Label source", label_kv)
            if label_block is not None:
                blocks.append(label_block)

        latest = st.get("latest_round") or {}
        if not latest:
            empty = list_table("Status", ["No completed rounds."])
            if empty is not None:
                blocks.append(empty)
            return Group(*blocks)

        def _round_table(label: str, round_info: Mapping[str, Any], ledger_info: Mapping[str, Any]):
            run_id = round_info.get("run_id") or ledger_info.get("run_id")
            main = kv_table(
                label,
                {
                    "r": round_info.get("round_index"),
                    "run_id": run_id,
                    "n_train": round_info.get("number_of_training_examples_used_in_round"),
                    "n_scored": round_info.get("number_of_candidates_scored_in_round"),
                    "top_k requested": round_info.get("selection_top_k_requested"),
                    "top_k effective": round_info.get("selection_top_k_effective_after_ties"),
                    "round_dir": round_info.get("round_dir"),
                },
            )
            if main is not None:
                blocks.append(main)
            if ledger_info:
                summary = ledger_info.get("objective_summary_stats") or {}
                kv = {
                    "model": ledger_info.get("model"),
                    "objective": ledger_info.get("objective"),
                    "selection": ledger_info.get("selection"),
                    "y_ops": ", ".join([p.get("name") for p in (ledger_info.get("y_ops") or [])]) or "(none)",
                    "score_min": summary.get("score_min"),
                    "score_median": summary.get("score_median"),
                    "score_max": summary.get("score_max"),
                }
                ledger_block = kv_table(f"{label} (ledger)", kv)
                if ledger_block is not None:
                    blocks.append(ledger_block)

        _round_table("Latest round", latest, st.get("latest_round_ledger") or {})
        selected = st.get("selected_round") or {}
        if selected and int(selected.get("round_index", -1)) != int(latest.get("round_index", -1)):
            _round_table("Selected round", selected, st.get("selected_round_ledger") or {})

        return Group(*blocks)

    head = kv_block(
        "Campaign",
        {
            "name": st.get("campaign_name"),
            "slug": st.get("campaign_slug"),
            "workdir": st.get("workdir"),
            "X column": st.get("x_column_name"),
            "Y column": st.get("y_column_name"),
            "num_rounds": st.get("num_rounds"),
            "prediction records": (st.get("writeback") or {}).get("prediction_records"),
        },
    )
    data = st.get("data") or {}
    data_block = ""
    if data:
        data_block = "\n\n" + kv_block(
            "Data",
            {
                "records": data.get("records_path"),
                "records exists": data.get("records_exists"),
                "rows": data.get("row_count"),
                "location": data.get("location_kind"),
            },
        )
    label_source = st.get("label_source") or {}
    label_block = ""
    if label_source:
        label_kv = {
            "kind": label_source.get("kind"),
            "exists": label_source.get("exists"),
            "valid": label_source.get("valid"),
            "label_count": label_source.get("label_count"),
            "available_rounds": ", ".join(map(str, label_source.get("available_rounds") or [])) or "(none)",
        }
        if label_source.get("path"):
            label_kv["path"] = label_source.get("path")
        if label_source.get("column"):
            label_kv["column"] = label_source.get("column")
        if label_source.get("error"):
            label_kv["error"] = label_source.get("error")
        label_block = "\n\n" + kv_block("Label source", label_kv)

    latest = st.get("latest_round") or {}
    if not latest:
        return head + data_block + label_block + "\n\n" + _dim("No completed rounds.")

    def _round_block(label: str, round_info: Mapping[str, Any], ledger_info: Mapping[str, Any]) -> list[str]:
        run_id = round_info.get("run_id") or ledger_info.get("run_id")
        main = kv_block(
            label,
            {
                "r": round_info.get("round_index"),
                "run_id": run_id,
                "n_train": round_info.get("number_of_training_examples_used_in_round"),
                "n_scored": round_info.get("number_of_candidates_scored_in_round"),
                "top_k requested": round_info.get("selection_top_k_requested"),
                "top_k effective": round_info.get("selection_top_k_effective_after_ties"),
                "round_dir": round_info.get("round_dir"),
            },
        )
        blocks = [main]
        if ledger_info:
            summary = ledger_info.get("objective_summary_stats") or {}
            kv = {
                "model": ledger_info.get("model"),
                "objective": ledger_info.get("objective"),
                "selection": ledger_info.get("selection"),
                "y_ops": ", ".join([p.get("name") for p in (ledger_info.get("y_ops") or [])]) or "(none)",
                "score_min": summary.get("score_min"),
                "score_median": summary.get("score_median"),
                "score_max": summary.get("score_max"),
            }
            blocks.append(kv_block(f"{label} (ledger)", kv))
        return blocks

    parts = [head]
    if data_block:
        parts.append(data_block.lstrip())
    if label_block:
        parts.append(label_block.lstrip())
    latest_blocks = _round_block("Latest round", latest, st.get("latest_round_ledger") or {})
    for b in latest_blocks:
        parts.extend(["", b])

    selected = st.get("selected_round") or {}
    if selected and int(selected.get("round_index", -1)) != int(latest.get("round_index", -1)):
        sel_blocks = _round_block("Selected round", selected, st.get("selected_round_ledger") or {})
        for b in sel_blocks:
            parts.extend(["", b])

    return "\n".join(parts)
