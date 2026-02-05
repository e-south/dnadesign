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

from dnadesign.cruncher.analysis.plots._savefig import savefig

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
    out_path: Path,
    *,
    dpi: int,
    png_compress_level: int,
) -> None:
    tf_list = list(tf_names)
    plt.style.use("seaborn-v0_8-ticks")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_learning, ax_elites, ax_unique, ax_worst = axes.flatten()

    df = sequences_df.copy()
    if "phase" in df.columns:
        df = df[df["phase"] == "draw"].copy()
    score_cols = _score_columns(tf_list)

    # ── learning curve ─────────────────────────────────────────────────────
    if "combined_score_final" not in df.columns or df.empty:
        _empty_panel(ax_learning, "Learning curve unavailable")
    else:
        df = df.copy()
        df["combined_score_final"] = pd.to_numeric(df["combined_score_final"], errors="coerce")
        df = df.dropna(subset=["combined_score_final"])
        if df.empty:
            _empty_panel(ax_learning, "Learning curve unavailable")
        else:
            best_by_draw = df.groupby("draw")["combined_score_final"].max().sort_index()
            cummax = best_by_draw.cummax()
            ax_learning.plot(cummax.index, cummax.values, color="#4c78a8", linewidth=2.0)
            ax_learning.set_title("Best score vs draw")
            ax_learning.set_xlabel("Draw")
            ax_learning.set_ylabel("Best combined score")

    # ── elite score snapshot ───────────────────────────────────────────────
    if elites_df is None or elites_df.empty or "min_norm" not in elites_df.columns:
        _empty_panel(ax_elites, "Elite score summary unavailable")
    else:
        elite_min = pd.to_numeric(elites_df["min_norm"], errors="coerce").dropna()
        if elite_min.empty:
            _empty_panel(ax_elites, "Elite score summary unavailable")
        else:
            ax_elites.boxplot(elite_min.to_numpy(), vert=True, patch_artist=True)
            ax_elites.set_title("Elite min-per-TF norm")
            ax_elites.set_ylabel("Min-per-TF norm")
            ax_elites.set_xticks([])

    # ── unique fraction ────────────────────────────────────────────────────
    if df.empty or "sequence" not in df.columns:
        _empty_panel(ax_unique, "Unique fraction unavailable")
    else:
        total = int(len(df))
        if total == 0:
            _empty_panel(ax_unique, "Unique fraction unavailable")
        else:
            if "canonical_sequence" in df.columns:
                unique = int(df["canonical_sequence"].astype(str).nunique())
            else:
                unique = int(df["sequence"].astype(str).nunique())
            frac = unique / float(total)
            ax_unique.bar(["unique"], [frac], color="#72b7b2")
            ax_unique.set_ylim(0.0, 1.0)
            ax_unique.set_title("Unique fraction")
            ax_unique.set_ylabel("Fraction")
            ax_unique.text(
                0,
                min(frac + 0.05, 0.95),
                f"{unique}/{total}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # ── worst-TF identity ──────────────────────────────────────────────────
    if not score_cols or any(col not in df.columns for col in score_cols) or df.empty:
        _empty_panel(ax_worst, "Worst-TF identity unavailable")
    else:
        scores = df[score_cols].to_numpy(dtype=float)
        worst_idx = np.argmin(scores, axis=1)
        worst_labels = [tf_list[idx] for idx in worst_idx]
        counts = pd.Series(worst_labels).value_counts().reindex(tf_list, fill_value=0)
        ax_worst.bar(counts.index, counts.values, color="#f58518")
        ax_worst.set_title("Worst-TF identity")
        ax_worst.set_xlabel("TF")
        ax_worst.set_ylabel("Count")
        ax_worst.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
