"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/overlap.py

Overlap summary panel plot.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dnadesign.cruncher.analysis.plots._savefig import savefig


def plot_overlap_panel(
    summary_df: pd.DataFrame,
    elite_overlap_df: pd.DataFrame,
    tf_names: Iterable[str],
    out_path: Path,
    *,
    focus_pair: tuple[str, str] | None = None,
    dpi: int,
    png_compress_level: int,
) -> None:
    tf_list = list(tf_names)
    if summary_df is None or summary_df.empty or not tf_list:
        raise ValueError("Overlap summary is required for overlap panel.")
    if elite_overlap_df is None or elite_overlap_df.empty:
        raise ValueError("Elite overlap table is required for overlap panel.")

    sns.set_style("ticks", {"axes.grid": False})
    n_tf = len(tf_list)
    n_elites = int(elite_overlap_df["id"].nunique()) if "id" in elite_overlap_df.columns else int(len(elite_overlap_df))
    n_pairs = int(len(summary_df))
    if n_tf <= 8:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax_heat, ax_hist = axes
        matrix = pd.DataFrame(0.0, index=tf_list, columns=tf_list)
        for _, row in summary_df.iterrows():
            tf_i = row.get("tf_i")
            tf_j = row.get("tf_j")
            if tf_i not in matrix.index or tf_j not in matrix.columns:
                continue
            rate = row.get("overlap_rate")
            if isinstance(rate, (int, float)):
                matrix.loc[tf_i, tf_j] = float(rate)
                matrix.loc[tf_j, tf_i] = float(rate)
        sns.heatmap(matrix, cmap="mako", vmin=0.0, vmax=1.0, square=True, ax=ax_heat, cbar=False)
        ax_heat.set_title("Overlap rate")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax_heat, ax_hist = axes
        summary_df = summary_df.copy()
        summary_df["pair"] = summary_df["tf_i"].astype(str) + "–" + summary_df["tf_j"].astype(str)
        top_pairs = summary_df.sort_values("overlap_rate", ascending=False).head(10)
        colors = ["#4c78a8"] * len(top_pairs)
        if focus_pair is not None and len(top_pairs):
            focus_label = "–".join(focus_pair)
            for idx, pair in enumerate(top_pairs["pair"]):
                if pair == focus_label or pair == "–".join(reversed(focus_pair)):
                    colors[idx] = "#f58518"
        ax_heat.bar(top_pairs["pair"], top_pairs["overlap_rate"], color=colors)
        ax_heat.set_title("Top overlap pairs")
        ax_heat.set_ylabel("Overlap rate")
        ax_heat.tick_params(axis="x", rotation=35)

    overlap_vals = pd.to_numeric(elite_overlap_df["overlap_total_bp"], errors="coerce").dropna()
    if overlap_vals.empty:
        ax_hist.axis("off")
        ax_hist.text(0.5, 0.5, "Overlap bp histogram unavailable", ha="center", va="center", fontsize=9)
    else:
        min_val = float(overlap_vals.min())
        max_val = float(overlap_vals.max())
        unique_vals = int(overlap_vals.nunique())
        if np.isclose(min_val, max_val, rtol=0.0, atol=1e-12):
            ax_hist.bar([min_val], [int(overlap_vals.size)], width=0.8, color="#4c78a8", alpha=0.85)
            ax_hist.set_xlim(min_val - 1.0, max_val + 1.0)
        else:
            bins = min(20, max(5, int(round(np.sqrt(float(overlap_vals.size))))))
            sns.histplot(overlap_vals, bins=bins, kde=unique_vals > 1, ax=ax_hist)
        ax_hist.set_title("Overlap bp per elite")
        ax_hist.set_xlabel("Total overlap bp")
        ax_hist.set_ylabel("Count")

    fig.text(
        0.99,
        0.01,
        "best-hit windows only; descriptive overlap",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.text(
        0.01,
        0.01,
        f"n_elites={n_elites}; n_pairs={n_pairs}",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
