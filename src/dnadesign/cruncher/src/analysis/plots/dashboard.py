"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/dashboard.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def _score_columns(tf_names: Iterable[str]) -> list[str]:
    return [f"score_{tf}" for tf in tf_names]


def _empty_panel(ax: plt.Axes, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color="#555555")


def plot_dashboard(
    sequences_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    tf_names: Iterable[str],
    overlap_summary_df: pd.DataFrame | None,
    elite_overlap_df: pd.DataFrame | None,
    out_path: Path,
) -> None:
    tf_list = list(tf_names)
    sns.set_style("ticks", {"axes.grid": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_trace, ax_worst, ax_heat, ax_overlap = axes.flatten()

    # ── worst-TF trace ──────────────────────────────────────────────────────
    df = sequences_df.copy()
    if "phase" in df.columns:
        df = df[df["phase"] == "draw"]
    score_cols = _score_columns(tf_list)
    if not score_cols or any(col not in df.columns for col in score_cols) or df.empty:
        _empty_panel(ax_trace, "Worst-TF trace unavailable")
    else:
        df = df.copy()
        df["min_score"] = df[score_cols].min(axis=1)
        if "chain" in df.columns and "draw" in df.columns:
            for _, chain_df in df.groupby("chain"):
                ax_trace.plot(chain_df["draw"], chain_df["min_score"], alpha=0.4)
            median_by_draw = df.groupby("draw")["min_score"].median()
            ax_trace.plot(median_by_draw.index, median_by_draw.values, color="black", linewidth=1.8)
        else:
            ax_trace.plot(df["min_score"].to_numpy(), alpha=0.7)
        ax_trace.set_title("Worst-TF trace (min score)")
        ax_trace.set_xlabel("Draw")
        ax_trace.set_ylabel("Min scaled score")

    # ── worst-TF frequency ──────────────────────────────────────────────────
    if not score_cols or any(col not in df.columns for col in score_cols) or df.empty:
        _empty_panel(ax_worst, "Worst-TF identity unavailable")
    else:
        scores = df[score_cols].to_numpy(dtype=float)
        worst_idx = np.argmin(scores, axis=1)
        worst_labels = [tf_list[idx] for idx in worst_idx]
        counts = pd.Series(worst_labels).value_counts().reindex(tf_list, fill_value=0)
        ax_worst.bar(counts.index, counts.values, color="#4c78a8")
        ax_worst.set_title("Worst-TF frequency")
        ax_worst.set_xlabel("TF")
        ax_worst.set_ylabel("Count")
        ax_worst.tick_params(axis="x", rotation=25)

    # ── overlap heatmap ─────────────────────────────────────────────────────
    if overlap_summary_df is None or overlap_summary_df.empty or not tf_list:
        _empty_panel(ax_heat, "Overlap heatmap unavailable")
    else:
        matrix = pd.DataFrame(0.0, index=tf_list, columns=tf_list)
        for _, row in overlap_summary_df.iterrows():
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

    # ── overlap distribution ────────────────────────────────────────────────
    if elite_overlap_df is None or elite_overlap_df.empty:
        _empty_panel(ax_overlap, "Overlap distribution unavailable")
    else:
        sns.histplot(elite_overlap_df["overlap_total_bp"], bins=20, kde=True, ax=ax_overlap)
        ax_overlap.set_title("Overlap bp per elite")
        ax_overlap.set_xlabel("Total overlap bp")
        ax_overlap.set_ylabel("Count")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
