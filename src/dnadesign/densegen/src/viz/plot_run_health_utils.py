"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_run_health_utils.py

Utility helpers for run-health plotting composition and panel layout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch

from .plot_run_helpers import _reason_family_label


def rate_series_from_counts(counts: pd.DataFrame) -> dict[str, np.ndarray]:
    ok = counts.get("ok", pd.Series(0.0, index=counts.index)).to_numpy(dtype=float)
    rejected = counts.get("rejected", pd.Series(0.0, index=counts.index)).to_numpy(dtype=float)
    duplicate = counts.get("duplicate", pd.Series(0.0, index=counts.index)).to_numpy(dtype=float)
    failed = counts.get("failed", pd.Series(0.0, index=counts.index)).to_numpy(dtype=float)
    totals = ok + rejected + duplicate + failed
    safe_totals = np.where(totals > 0.0, totals, 1.0)
    acceptance = ok / safe_totals
    waste = (rejected + duplicate + failed) / safe_totals
    duplicate_rate = duplicate / safe_totals
    return {
        "acceptance": acceptance,
        "waste": waste,
        "duplicate": duplicate_rate,
        "totals": totals,
    }


def subtitle(
    ax: plt.Axes,
    text: str,
    *,
    fontsize: float,
    y: float = 1.02,
    color: str = "#444444",
) -> None:
    ax.text(
        0.0,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color=color,
    )


def solver_ticks(values: np.ndarray, *, max_ticks: int = 10) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=float)
    unique = np.unique(values.astype(int))
    lo = int(unique.min())
    hi = int(unique.max())
    if lo == hi:
        return np.array([float(lo)], dtype=float)
    span = hi - lo + 1
    n_ticks = min(int(max_ticks), max(2, span))
    raw = np.linspace(lo, hi, num=n_ticks)
    ticks = np.unique(np.rint(raw).astype(int))
    if ticks.size < 2:
        ticks = np.array([lo, hi], dtype=int)
    return ticks.astype(float)


def link_panels_by_ticks(fig: plt.Figure, ax_top: plt.Axes, ax_bottom: plt.Axes, ticks: np.ndarray) -> None:
    if ticks.size == 0:
        return
    y_top = float(ax_top.get_ylim()[0])
    y_bottom = float(ax_bottom.get_ylim()[1])
    for x in ticks.tolist():
        connector = ConnectionPatch(
            xyA=(float(x), y_top),
            coordsA=ax_top.transData,
            xyB=(float(x), y_bottom),
            coordsB=ax_bottom.transData,
            axesA=ax_top,
            axesB=ax_bottom,
            linestyle="--",
            linewidth=0.55,
            color="#9a9a9a",
            alpha=0.55,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(connector)


def save_axes_subset(fig: plt.Figure, path: Path, axes: list[plt.Axes | None], *, pad: float = 0.05) -> None:
    selected = [ax for ax in axes if ax is not None]
    if not selected:
        raise ValueError(f"No axes provided for saving subset figure: {path}")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = selected[0].get_tightbbox(renderer)
    for ax in selected[1:]:
        bbox = bbox.union([bbox, ax.get_tightbbox(renderer)])
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, bbox_inches=bbox_inches.expanded(1.0 + pad, 1.0 + pad))


def aggregate_reason_pareto(problem_df: pd.DataFrame, *, top_k: int | None = 8) -> pd.DataFrame:
    if problem_df is None or problem_df.empty:
        return pd.DataFrame(columns=["rejected", "failed", "total"])
    required = {"status", "reason"}
    missing = required - set(problem_df.columns)
    if missing:
        raise ValueError(f"run_health reason analysis missing required columns: {sorted(missing)}")
    reasons = problem_df.copy()
    reasons["reason_family"] = reasons.apply(
        lambda row: _reason_family_label(
            str(row.get("status", "")),
            row.get("reason"),
            row.get("detail_json"),
        ),
        axis=1,
    )
    pivot = (
        reasons.groupby(["reason_family", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["rejected", "failed"], fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    if top_k is not None and len(pivot) > int(top_k):
        head = pivot.head(int(top_k)).copy()
        tail = pivot.iloc[int(top_k) :]
        other = pd.DataFrame(
            {
                "rejected": [float(tail["rejected"].sum())],
                "failed": [float(tail["failed"].sum())],
                "total": [float(tail["total"].sum())],
            },
            index=["other"],
        )
        pivot = pd.concat([head, other], axis=0)
    return pivot
