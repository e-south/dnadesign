"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/optimization.py

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

from dnadesign.cruncher.analysis.plots._savefig import savefig

logger = logging.getLogger(__name__)


def _score_columns(tf_names: Iterable[str]) -> list[str]:
    return [f"score_{tf}" for tf in tf_names]


def plot_worst_tf_trace(
    sequences_df: pd.DataFrame,
    tf_names: Iterable[str],
    out_path: Path,
    *,
    dpi: int,
    png_compress_level: int,
) -> None:
    df = sequences_df.copy()
    if "phase" in df.columns:
        df = df[df["phase"] == "draw"]
    score_cols = _score_columns(tf_names)
    if not score_cols or any(col not in df.columns for col in score_cols):
        logger.warning("Skipping worst-tf trace: missing per-TF score columns.")
        return
    if df.empty:
        logger.warning("Skipping worst-tf trace: no draw samples.")
        return
    df = df.copy()
    df["min_score"] = df[score_cols].min(axis=1)
    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(8, 4))
    if "chain" in df.columns and "draw" in df.columns:
        for chain_id, chain_df in df.groupby("chain"):
            ax.plot(chain_df["draw"], chain_df["min_score"], alpha=0.5, label=f"chain {chain_id}")
        median_by_draw = df.groupby("draw")["min_score"].median()
        ax.plot(median_by_draw.index, median_by_draw.values, color="black", linewidth=2.0, label="median")
    else:
        ax.plot(df["min_score"].to_numpy(), alpha=0.7, label="min score")
    ax.set_title("Worst-TF score trace (min across TFs)")
    ax.set_xlabel("Draw")
    ax.set_ylabel("Min scaled score")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)


def plot_worst_tf_identity(
    sequences_df: pd.DataFrame,
    tf_names: Iterable[str],
    out_path: Path,
    *,
    dpi: int,
    png_compress_level: int,
) -> None:
    tf_list = list(tf_names)
    df = sequences_df.copy()
    if "phase" in df.columns:
        df = df[df["phase"] == "draw"]
    score_cols = _score_columns(tf_list)
    if not score_cols or any(col not in df.columns for col in score_cols):
        logger.warning("Skipping worst-tf identity: missing per-TF score columns.")
        return
    if df.empty:
        logger.warning("Skipping worst-tf identity: no draw samples.")
        return
    scores = df[score_cols].to_numpy(dtype=float)
    worst_idx = np.argmin(scores, axis=1)
    df = df.copy()
    df["worst_tf"] = [tf_list[idx] for idx in worst_idx]
    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if "chain" in df.columns and "draw" in df.columns:
        for tf in tf_list:
            subset = df[df["worst_tf"] == tf]
            ax.scatter(
                subset["draw"],
                subset["chain"],
                s=10,
                alpha=0.7,
                label=tf,
            )
        ax.set_ylabel("Chain")
    else:
        for tf in tf_list:
            subset = df[df["worst_tf"] == tf]
            ax.scatter(
                subset.index.to_numpy(),
                np.zeros(len(subset)),
                s=10,
                alpha=0.7,
                label=tf,
            )
        ax.set_yticks([])
    ax.set_title("Worst-TF identity over time (argmin TF)")
    ax.set_xlabel("Draw")
    ax.legend(frameon=False, fontsize=8, ncol=min(4, len(tf_list)))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)


def plot_elite_filter_waterfall(
    counts: dict[str, int],
    out_path: Path,
    *,
    dpi: int,
    png_compress_level: int,
) -> None:
    stages = [
        ("total_draws_seen", "Total draws"),
        ("passed_pre_filter", "Filter pass"),
        ("kept_after_mmr", "MMR elites"),
    ]
    values = []
    labels = []
    for key, label in stages:
        if key not in counts:
            continue
        labels.append(label)
        values.append(int(counts[key]))
    if not values:
        logger.warning("Skipping elite filter waterfall: missing counts.")
        return
    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(values)), values, marker="o")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Elite filter waterfall")
    for idx, val in enumerate(values):
        ax.text(idx, val, str(val), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
