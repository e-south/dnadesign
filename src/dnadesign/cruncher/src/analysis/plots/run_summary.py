"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/run_summary.py

Publication-ready run summary figure.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.plots._savefig import savefig


def _score_columns(tf_names: Iterable[str]) -> list[str]:
    return [f"score_{tf}" for tf in tf_names]


def _empty_panel(ax: plt.Axes, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color="#555555")


def _best_so_far_min_norm(df: pd.DataFrame) -> pd.Series | None:
    if df.empty or "min_per_tf_norm" not in df.columns:
        return None
    subset = df.copy()
    subset["min_per_tf_norm"] = pd.to_numeric(subset["min_per_tf_norm"], errors="coerce")
    subset = subset.dropna(subset=["min_per_tf_norm"])
    if subset.empty:
        return None
    if "draw" in subset.columns:
        best_by_draw = subset.groupby("draw")["min_per_tf_norm"].max().sort_index()
        return best_by_draw.cummax()
    return subset["min_per_tf_norm"].expanding().max()


def _min_norm_from_scores(df: pd.DataFrame, tf_names: list[str]) -> pd.Series | None:
    score_cols = _score_columns(tf_names)
    missing = [col for col in score_cols if col not in df.columns]
    if missing:
        return None
    scores = df[score_cols].to_numpy(dtype=float)
    return pd.Series(np.nanmin(scores, axis=1), index=df.index)


def _plot_swap_summary(ax: plt.Axes, optimizer_stats: dict[str, object] | None) -> None:
    attempts = []
    accepts = []
    if isinstance(optimizer_stats, dict):
        attempts = optimizer_stats.get("swap_attempts_by_pair") or []
        accepts = optimizer_stats.get("swap_accepts_by_pair") or []
    attempts = [int(v) for v in attempts] if attempts else []
    accepts = [int(v) for v in accepts] if accepts else []
    if not attempts or len(attempts) != len(accepts):
        _empty_panel(ax, "Swap acceptance unavailable")
        return
    total_attempts = sum(attempts)
    total_accepts = sum(accepts)
    rate = total_accepts / total_attempts if total_attempts else 0.0
    ax.bar([0], [rate], color="#4c78a8", width=0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("PT swap acceptance")
    ax.set_xticks([])
    ax.set_ylabel("Acceptance rate")
    ax.text(0, min(rate + 0.07, 0.95), f"{rate:.2f}", ha="center", va="bottom", fontsize=9)


def plot_run_summary(
    sequences_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    nn_df: pd.DataFrame,
    tf_names: Iterable[str],
    out_path: Path,
    *,
    optimizer_stats: dict[str, object] | None,
    score_scale: str | None,
    dpi: int,
    png_compress_level: int,
) -> None:
    tf_list = list(tf_names)
    plt.style.use("seaborn-v0_8-ticks")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_learning, ax_dist, ax_div, ax_swap = axes.flatten()

    # ── Tile A: best-so-far min-TF score ──────────────────────────────────
    df = sequences_df.copy()
    if "phase" in df.columns and df["phase"].isin(["tune", "draw"]).any():
        df = df.sort_values("draw")
    best_series = _best_so_far_min_norm(df)
    if best_series is None:
        _empty_panel(ax_learning, "Learning curve unavailable")
    else:
        ax_learning.plot(best_series.index, best_series.values, color="#4c78a8", linewidth=2.0)
        ax_learning.set_title("Best-so-far min-TF score")
        ax_learning.set_xlabel("Sweep")
        ax_learning.set_ylabel(f"Min-per-TF ({score_scale})" if score_scale else "Min-per-TF")
        if "phase" in df.columns:
            tune_mask = df["phase"] == "tune"
            draw_mask = df["phase"] == "draw"
            if tune_mask.any() and draw_mask.any():
                first_draw = df.loc[draw_mask, "draw"].min()
                ax_learning.axvline(first_draw, linestyle="--", color="#888888", linewidth=1.0)
        elite_min = None
        if elites_df is not None and not elites_df.empty:
            if "min_norm" in elites_df.columns:
                elite_min = pd.to_numeric(elites_df["min_norm"], errors="coerce").dropna()
            elif "min_per_tf_norm" in elites_df.columns:
                elite_min = pd.to_numeric(elites_df["min_per_tf_norm"], errors="coerce").dropna()
        if elite_min is not None and not elite_min.empty:
            ax_learning.text(
                0.02,
                0.92,
                f"elite median={elite_min.median():.3f}",
                transform=ax_learning.transAxes,
                fontsize=9,
                color="#333333",
            )

    # ── Tile B: elite vs baseline distribution ────────────────────────────
    if baseline_df is None or baseline_df.empty:
        _empty_panel(ax_dist, "Baseline distribution unavailable")
    else:
        baseline_min = _min_norm_from_scores(baseline_df, tf_list)
        elite_min = None
        if elites_df is not None and not elites_df.empty:
            if "min_norm" in elites_df.columns:
                elite_min = pd.to_numeric(elites_df["min_norm"], errors="coerce").dropna()
            else:
                elite_min = _min_norm_from_scores(elites_df, tf_list)
        if baseline_min is None or baseline_min.empty:
            _empty_panel(ax_dist, "Baseline distribution unavailable")
        else:
            ax_dist.hist(baseline_min, bins=20, color="#c9c9c9", alpha=0.7, label="baseline")
            if elite_min is not None and not elite_min.empty:
                ax_dist.hist(elite_min, bins=20, color="#4c78a8", alpha=0.7, label="elites")
            ax_dist.set_title("Elite vs baseline (min-per-TF)")
            ax_dist.set_xlabel(f"Min-per-TF ({score_scale})" if score_scale else "Min-per-TF")
            ax_dist.set_ylabel("Count")
            ax_dist.legend(frameon=False, fontsize=8)

    # ── Tile C: elite diversity headline ──────────────────────────────────
    if nn_df is None or nn_df.empty or "nn_dist" not in nn_df.columns:
        _empty_panel(ax_div, "Elite diversity unavailable")
    else:
        nn_vals = pd.to_numeric(nn_df["nn_dist"], errors="coerce").dropna()
        if nn_vals.empty:
            _empty_panel(ax_div, "Elite diversity unavailable")
        else:
            ax_div.axis("off")
            ax_div.text(
                0.5,
                0.60,
                f"median NN={nn_vals.median():.3f}",
                ha="center",
                va="center",
                fontsize=12,
                color="#333333",
            )
            ax_div.text(
                0.5,
                0.40,
                f"min NN={nn_vals.min():.3f}",
                ha="center",
                va="center",
                fontsize=11,
                color="#555555",
            )
            ax_div.set_title("Elite diversity")

    # ── Tile D: PT swap acceptance summary ────────────────────────────────
    _plot_swap_summary(ax_swap, optimizer_stats)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
