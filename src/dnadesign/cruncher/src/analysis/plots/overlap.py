"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/overlap.py

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

from dnadesign.cruncher.analysis.overlap import extract_elite_hits
from dnadesign.cruncher.analysis.plots._savefig import savefig

logger = logging.getLogger(__name__)


def plot_overlap_heatmap(
    summary_df: pd.DataFrame,
    tf_names: Iterable[str],
    out_path: Path,
    *,
    dpi: int,
    png_compress_level: int,
) -> None:
    tf_list = list(tf_names)
    if summary_df is None or summary_df.empty or not tf_list:
        logger.warning("Skipping overlap heatmap: missing overlap_summary or TF list.")
        return
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
    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, cmap="mako", vmin=0.0, vmax=1.0, square=True, ax=ax)
    ax.set_title("TF overlap rate (best-hit windows)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)


def plot_overlap_bp_distribution(
    elite_overlap_df: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int,
    png_compress_level: int,
) -> None:
    if elite_overlap_df is None or elite_overlap_df.empty:
        logger.warning("Skipping overlap distribution: empty elite overlap table.")
        return
    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(elite_overlap_df["overlap_total_bp"], bins=30, kde=True, ax=ax)
    ax.set_title("Total overlap bp per elite")
    ax.set_xlabel("Total overlapping bp")
    ax.set_ylabel("Count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)


def plot_overlap_strand_combos(
    summary_df: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int,
    png_compress_level: int,
) -> None:
    if summary_df is None or summary_df.empty:
        logger.warning("Skipping overlap strand combos: empty overlap summary.")
        return
    combo_cols = ["strand_pp", "strand_pm", "strand_mp", "strand_mm"]
    if not all(col in summary_df.columns for col in combo_cols):
        logger.warning("Skipping overlap strand combos: missing strand count columns.")
        return
    df = summary_df.copy()
    df["pair"] = df["tf_i"].astype(str) + "–" + df["tf_j"].astype(str)
    df = df.sort_values("pair")
    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.6), 4))
    bottoms = np.zeros(len(df))
    labels = ["++", "+-", "-+", "--"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for label, col, color in zip(labels, combo_cols, colors):
        values = df[col].to_numpy(dtype=float)
        ax.bar(df["pair"], values, bottom=bottoms, label=label, color=color, alpha=0.85)
        bottoms += values
    ax.set_title("Strand-orientation combos (best hits)")
    ax.set_xlabel("TF pair")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate(rotation=45, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)


def plot_motif_offset_rug(
    elites_df: pd.DataFrame,
    tf_names: Iterable[str],
    out_path: Path,
    *,
    pwm_widths: dict[str, int] | None = None,
    dpi: int,
    png_compress_level: int,
) -> None:
    hits_df = extract_elite_hits(elites_df, tf_names, pwm_widths=pwm_widths)
    if hits_df.empty:
        logger.warning("Skipping motif offset rug: no elite hits available.")
        return
    hits_df["strand"] = hits_df["strand"].fillna("?")
    hits_df["offset_1"] = hits_df["offset"].astype(float) + 1
    tf_list = list(tf_names)
    n = len(tf_list)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    sns.set_style("ticks", {"axes.grid": False})
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 2.8 * nrows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, tf in enumerate(tf_list):
        ax = axes_list[i]
        subset = hits_df[hits_df["tf"] == tf]
        if subset.empty:
            ax.axis("off")
            continue
        sns.stripplot(
            data=subset,
            x="offset_1",
            y="strand",
            hue="strand",
            ax=ax,
            jitter=0.25,
            size=3,
            palette="Set2",
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title(f"{tf} best-hit offsets")
        ax.set_xlabel("Start position (1-based)")
        ax.set_ylabel("Strand")
    for j in range(i + 1, len(axes_list)):
        axes_list[j].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
