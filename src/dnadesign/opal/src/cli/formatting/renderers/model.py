"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/formatting/renderers/model.py

Renders model command output for OPAL CLI. Formats model metadata and feature
importance summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

from ...tui import kv_table, tui_enabled
from ..core import _fmt_params, bullet_list, kv_block


def render_model_show_human(info: Mapping[str, Any]) -> str:
    if tui_enabled():
        from rich import box
        from rich.console import Group
        from rich.table import Table

        head = kv_table(
            "Model",
            {
                "type": info.get("model_type"),
                "params": _fmt_params(info.get("params", {})),
            },
        )
        blocks = [head] if head is not None else []
        top = info.get("feature_importance_top20") or []
        if top:
            table = Table(title="Top-20 feature importance (if available)", box=box.ASCII)
            table.add_column("rank", style="bold")
            table.add_column("feature_index")
            table.add_column("weight")
            for i, row in enumerate(top, start=1):
                fi = row.get("feature_index")
                w = row.get("feature_importance")
                table.add_row(str(i), str(fi), f"{w:.6g}" if w is not None else "")
            blocks.append(table)
        if blocks:
            return Group(*blocks)
    head = kv_block(
        "Model",
        {
            "type": info.get("model_type"),
            "params": _fmt_params(info.get("params", {})),
        },
    )
    top = info.get("feature_importance_top20") or []
    if not top:
        return head

    lines = []
    for i, row in enumerate(top, start=1):
        fi = row.get("feature_index")
        w = row.get("feature_importance")
        rk = row.get("feature_rank")
        lines.append(f"{i:2d}. feature_index={fi}  weight={w:.6g}  rank={rk}")
    return head + "\n\n" + bullet_list("Top-20 feature importance (if available)", lines)
