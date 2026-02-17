"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/formatting/renderers/record.py

Renders record-show command output for OPAL CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

from ..core import _truncate, bullet_list, kv_block


def render_record_report_human(report: Mapping[str, Any]) -> str:
    if report.get("error"):
        return kv_block("record-show", {"error": report.get("error")})

    sources = report.get("sources") or {}
    src_block = ""
    if sources:
        src_block = "\n\n" + kv_block(
            "Sources",
            {
                "records": sources.get("records", ""),
                "ledger_predictions": sources.get("ledger_predictions_dir", ""),
                "ledger_runs": sources.get("ledger_runs_path", ""),
            },
        )

    head = kv_block(
        "Record",
        {
            "id": report.get("id"),
            "sequence": (_truncate(report.get("sequence", "")) if report.get("sequence") else "(hidden)"),
            "label_hist_column": report.get("label_hist_column"),
            "n_labels": (len(report.get("labels")) if isinstance(report.get("labels"), list) else 0),
        },
    )

    runs = report.get("runs") or []
    lines = []
    for r in runs:
        parts = [
            f"r={r.get('as_of_round')}",
            f"run_id={_truncate(r.get('run_id', ''), 18)}",
        ]
        for k in ("sel__is_selected", "sel__rank_competition", "pred__score_selected"):
            if k in r:
                parts.append(f"{k.split('__', 1)[-1]}={r[k]}")
        lines.append(", ".join(parts))

    runs_block = bullet_list("Runs (per as_of_round)", lines)
    return "\n".join([head + src_block, "", runs_block])
