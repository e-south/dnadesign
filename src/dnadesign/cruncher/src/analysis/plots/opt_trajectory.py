"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/opt_trajectory.py

Optimization trajectory plot with baseline cloud.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from dnadesign.cruncher.analysis.plots._savefig import savefig
from dnadesign.cruncher.analysis.trajectory import project_scores


def _axis_label(metric: str, score_scale: str | None) -> str:
    if metric.startswith("score_"):
        label = f"{metric.replace('score_', '')} score"
    elif metric == "worst_tf_score":
        label = "worst-TF score"
    elif metric == "second_worst_tf_score":
        label = "2nd-worst TF score"
    else:
        label = metric
    if score_scale:
        return f"{label} ({score_scale})"
    return label


def _line_collection(
    segments: np.ndarray,
    *,
    color: tuple[float, float, float],
    alphas: np.ndarray | None,
    linestyle: str = "solid",
) -> LineCollection:
    if segments.size == 0:
        return LineCollection([], linewidths=0)
    if alphas is None:
        alphas = np.full(len(segments), 0.8, dtype=float)
    colors = np.column_stack(
        [np.full(len(segments), color[0]), np.full(len(segments), color[1]), np.full(len(segments), color[2]), alphas]
    )
    collection = LineCollection(segments, colors=colors, linewidths=1.5)
    collection.set_linestyle(linestyle)
    return collection


def plot_opt_trajectory(
    trajectory_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    tf_names: Iterable[str],
    out_path: Path,
    *,
    identity_mode: str | None,
    elite_centroid: tuple[float, float] | None,
    score_scale: str | None,
    dpi: int,
    png_compress_level: int,
) -> None:
    if trajectory_df is None or trajectory_df.empty:
        raise ValueError("Trajectory points are required for opt trajectory plot.")
    if baseline_df is None or baseline_df.empty:
        raise ValueError("Baseline table is required for opt trajectory plot.")

    x = trajectory_df["x"].astype(float).to_numpy()
    y = trajectory_df["y"].astype(float).to_numpy()
    x_metric = str(trajectory_df["x_metric"].iloc[0])
    y_metric = str(trajectory_df["y_metric"].iloc[0])

    base_x, base_y, _, _, _, _ = project_scores(baseline_df, tf_names)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(base_x, base_y, s=8, c="#c9c9c9", alpha=0.35, edgecolors="none", label="baseline")

    if x.size >= 2:
        points = np.column_stack([x, y])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        phase = trajectory_df["phase"] if "phase" in trajectory_df.columns else None
        color = (0.24, 0.44, 0.65)
        if phase is not None and phase.size >= 2:
            phase_start = phase.iloc[:-1].to_numpy()
            tune_mask = phase_start == "tune"
            draw_mask = phase_start == "draw"
            if np.any(tune_mask):
                tune_segments = segments[tune_mask]
                ax.add_collection(
                    _line_collection(
                        tune_segments,
                        color=color,
                        alphas=np.full(len(tune_segments), 0.35),
                        linestyle="dashed",
                    )
                )
            if np.any(draw_mask):
                draw_segments = segments[draw_mask]
                alphas = np.linspace(0.2, 0.95, len(draw_segments))
                ax.add_collection(_line_collection(draw_segments, color=color, alphas=alphas))
        else:
            alphas = np.linspace(0.2, 0.95, len(segments))
            ax.add_collection(_line_collection(segments, color=color, alphas=alphas))

        if phase is not None and phase.size >= 1:
            phase_series = phase.astype(str)
            draw_indices = np.where(phase_series == "draw")[0]
            if draw_indices.size:
                first_draw_idx = int(draw_indices[0])
                ax.scatter(
                    [x[first_draw_idx]],
                    [y[first_draw_idx]],
                    s=40,
                    marker="o",
                    color="#333333",
                    zorder=5,
                    label="tune→draw",
                )

    if elite_centroid is not None:
        ax.scatter(
            [elite_centroid[0]],
            [elite_centroid[1]],
            s=80,
            marker="*",
            color="#f58518",
            zorder=6,
            label="elite median",
        )

    ax.set_xlabel(_axis_label(x_metric, score_scale))
    ax.set_ylabel(_axis_label(y_metric, score_scale))
    tf_label = ", ".join(str(tf) for tf in tf_names)
    scale_label = f", scale={score_scale}" if score_scale else ""
    ax.set_title(f"Optimization trajectory ({tf_label}{scale_label}; cold chain)")
    if identity_mode:
        ax.text(
            0.02,
            0.98,
            f"identity={identity_mode}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#444444",
        )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
