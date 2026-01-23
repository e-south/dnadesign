# ABOUTME: Renders runs list output for OPAL CLI.
# ABOUTME: Formats list of run metadata entries.
"""Runs list renderers."""

from __future__ import annotations

from typing import Mapping, Sequence

from ...tui import list_table, tui_enabled
from ..core import _dim, _truncate, bullet_list


def render_runs_list_human(rows: Sequence[Mapping[str, object]]) -> str:
    if tui_enabled():
        from rich import box
        from rich.table import Table

        if not rows:
            empty = list_table("Runs", ["No runs found."])
            if empty is not None:
                return empty
        table = Table(title="Runs", box=box.ASCII)
        table.add_column("r", style="bold")
        table.add_column("run_id")
        table.add_column("model")
        table.add_column("objective")
        table.add_column("selection")
        table.add_column("n_train")
        table.add_column("n_scored")
        for r in rows:
            table.add_row(
                str(r.get("as_of_round")),
                _truncate(r.get("run_id", ""), 18),
                str(r.get("model")),
                str(r.get("objective")),
                str(r.get("selection")),
                str(r.get("stats_n_train")),
                str(r.get("stats_n_scored")),
            )
        return table
    if not rows:
        return _dim("No runs found.")
    lines = []
    for r in rows:
        parts = [
            f"r={r.get('as_of_round')}",
            f"run_id={_truncate(r.get('run_id', ''), 18)}",
            f"model={r.get('model')}",
            f"objective={r.get('objective')}",
            f"selection={r.get('selection')}",
            f"n_train={r.get('stats_n_train')}",
            f"n_scored={r.get('stats_n_scored')}",
        ]
        lines.append(", ".join(parts))
    return bullet_list("Runs", lines)
