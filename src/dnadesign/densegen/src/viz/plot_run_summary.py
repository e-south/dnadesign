"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_run_summary.py

Summary-table helpers for DenseGen run-health plotting.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_common import _save_figure, _style


def build_run_health_summary_frame(attempts_df: pd.DataFrame, *, plan_quotas: dict[str, int]) -> pd.DataFrame:
    n_attempts = int(len(attempts_df))
    n_ok = int((attempts_df["status"] == "ok").sum())
    n_rej = int((attempts_df["status"] == "rejected").sum())
    n_dup = int((attempts_df["status"] == "duplicate").sum())
    n_fail = int((attempts_df["status"] == "failed").sum())
    waste_rate = (n_rej + n_dup + n_fail) / float(max(1, n_attempts))
    rows: list[dict[str, object]] = [
        {"scope": "run", "name": "attempts", "value": n_attempts, "unit": "count"},
        {"scope": "run", "name": "ok", "value": n_ok, "unit": "count"},
        {"scope": "run", "name": "rejected", "value": n_rej, "unit": "count"},
        {"scope": "run", "name": "duplicate", "value": n_dup, "unit": "count"},
        {"scope": "run", "name": "failed", "value": n_fail, "unit": "count"},
        {"scope": "run", "name": "waste_rate", "value": waste_rate, "unit": "fraction"},
    ]
    by_plan = (
        attempts_df.groupby("plan_name")
        .agg(
            attempts=("status", "size"),
            ok=("status", lambda s: int((s == "ok").sum())),
            rejected=("status", lambda s: int((s == "rejected").sum())),
            duplicate=("status", lambda s: int((s == "duplicate").sum())),
            failed=("status", lambda s: int((s == "failed").sum())),
        )
        .reset_index()
    )
    for row in by_plan.to_dict(orient="records"):
        plan = str(row["plan_name"])
        quota = int(plan_quotas.get(plan, 0))
        ok_count = int(row["ok"])
        rows.append(
            {
                "scope": "plan",
                "name": f"{plan}:accepted_over_quota",
                "value": (ok_count / float(quota)) if quota > 0 else np.nan,
                "unit": "fraction",
            }
        )
    return pd.DataFrame(rows)


def render_run_health_summary_table_figure(
    summary_df: pd.DataFrame,
    out_path: Path,
    *,
    style: Optional[dict] = None,
) -> None:
    style_cfg = _style(style)
    display = summary_df.copy()
    if "value" in display.columns:
        display["value"] = display["value"].map(
            lambda value: f"{float(value):.6g}" if isinstance(value, (float, np.floating)) else str(value)
        )
    fig_width = min(22.0, max(10.0, 2.2 * len(display.columns) + 1.7))
    fig_height = min(22.0, max(3.0, 0.52 * max(1, len(display)) + 1.2))
    fig, ax = plt.subplots(figsize=(float(fig_width), float(fig_height)), constrained_layout=False)
    ax.axis("off")
    table = ax.table(
        cellText=display.values.tolist(),
        colLabels=[str(col) for col in display.columns],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table_font = max(11.0, float(style_cfg.get("tick_size", style_cfg.get("font_size", 13.0) * 0.78)))
    table.set_fontsize(table_font)
    table.scale(1.03, 1.28)
    save_style = dict(style_cfg)
    save_style["save_pad_inches"] = min(float(save_style.get("save_pad_inches", 0.08)), 0.02)
    _save_figure(fig, out_path, style=save_style)
    plt.close(fig)


__all__ = ["build_run_health_summary_frame", "render_run_health_summary_table_figure"]
