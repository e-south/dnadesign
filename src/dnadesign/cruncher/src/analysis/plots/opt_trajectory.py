"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/opt_trajectory.py

Optimization trajectory plots for story and debug views.

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
from dnadesign.cruncher.analysis.trajectory import compute_best_so_far_path, project_scores


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


def _require_df(df: pd.DataFrame | None, *, name: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"{name} data is required for opt trajectory plot.")
    return df


def _first_metric(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        raise ValueError(f"Trajectory points missing required column '{col}'.")
    return str(df[col].iloc[0])


def _sample_rows(df: pd.DataFrame, *, stride: int, group_col: str = "chain") -> pd.DataFrame:
    if stride <= 1 or df.empty:
        return df
    sampled_parts: list[pd.DataFrame] = []
    if group_col in df.columns:
        for _, group in df.groupby(group_col, sort=True, dropna=False):
            group = group.sort_values("sweep").reset_index(drop=True)
            idx = list(range(0, len(group), stride))
            if len(group) - 1 not in idx:
                idx.append(len(group) - 1)
            sampled_parts.append(group.iloc[sorted(set(idx))])
    else:
        ordered = df.sort_values("sweep").reset_index(drop=True)
        idx = list(range(0, len(ordered), stride))
        if len(ordered) - 1 not in idx:
            idx.append(len(ordered) - 1)
        sampled_parts.append(ordered.iloc[sorted(set(idx))])
    return pd.concat(sampled_parts).sort_values(["sweep", group_col]).reset_index(drop=True)


def _legend_labels(ax: plt.Axes) -> list[str]:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return []
    ax.legend(frameon=False, fontsize=8)
    return labels


def _set_axes_and_title(
    ax: plt.Axes,
    *,
    x_metric: str,
    y_metric: str,
    tf_names: Iterable[str],
    score_scale: str | None,
    suffix: str,
    identity_mode: str | None = None,
) -> None:
    ax.set_xlabel(_axis_label(x_metric, score_scale))
    ax.set_ylabel(_axis_label(y_metric, score_scale))
    tf_label = ", ".join(str(tf) for tf in tf_names)
    scale_label = f", scale={score_scale}" if score_scale else ""
    ax.set_title(f"Optimization trajectory ({tf_label}{scale_label}; {suffix})")
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


def plot_opt_trajectory_story(
    *,
    trajectory_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    selected_df: pd.DataFrame | None,
    consensus_anchors: list[dict[str, object]] | None,
    tf_names: Iterable[str],
    out_path: Path,
    score_scale: str | None,
    dpi: int,
    png_compress_level: int,
    stride: int = 50,
    baseline_mode: str = "hexbin",
) -> dict[str, object]:
    trajectory = _require_df(trajectory_df, name="Trajectory")
    baseline = _require_df(baseline_df, name="Baseline")
    x_metric = _first_metric(trajectory, "x_metric")
    y_metric = _first_metric(trajectory, "y_metric")
    if "objective_scalar" not in trajectory.columns:
        raise ValueError("Trajectory points must include objective_scalar for story plot.")
    if baseline_mode not in {"hexbin", "scatter"}:
        raise ValueError(f"Unsupported baseline_mode '{baseline_mode}'. Use 'hexbin' or 'scatter'.")

    base_x, base_y, _, _, _, _ = project_scores(baseline, tf_names)
    fig, ax = plt.subplots(figsize=(6.8, 5.1))
    if baseline_mode == "hexbin":
        ax.hexbin(
            base_x,
            base_y,
            gridsize=52,
            bins="log",
            mincnt=1,
            cmap="Greys",
            linewidths=0,
            alpha=0.55,
            label=f"random baseline (n={len(base_x)})",
        )
    else:
        ax.scatter(base_x, base_y, s=8, c="#bdbdbd", alpha=0.30, edgecolors="none", label="random baseline")

    draw_points = _sample_rows(trajectory, stride=max(1, int(stride)))
    if not draw_points.empty:
        sweep = pd.to_numeric(draw_points["sweep"], errors="coerce").astype(float)
        sweep_min = float(sweep.min())
        sweep_range = float(sweep.max() - sweep_min)
        if sweep_range <= 0:
            alphas = np.full(len(draw_points), 0.60, dtype=float)
        else:
            alphas = 0.20 + 0.75 * ((sweep - sweep_min) / sweep_range)
        colors = np.column_stack(
            [
                np.full(len(draw_points), 0.13),
                np.full(len(draw_points), 0.40),
                np.full(len(draw_points), 0.73),
                np.asarray(alphas, dtype=float),
            ]
        )
        ax.scatter(
            draw_points["x"].astype(float),
            draw_points["y"].astype(float),
            s=18,
            c=colors,
            edgecolors="none",
            zorder=4,
            label="optimization states",
        )

    best_path = compute_best_so_far_path(trajectory, objective_col="objective_scalar", sweep_col="sweep")
    ax.plot(
        best_path["x"].astype(float),
        best_path["y"].astype(float),
        color="#0b4f9c",
        linewidth=1.7,
        zorder=5,
        label="best-so-far",
    )
    marker_stride = max(1, int(stride // 2))
    markers = _sample_rows(best_path.assign(chain=0), stride=marker_stride)
    ax.scatter(
        markers["x"].astype(float),
        markers["y"].astype(float),
        s=22,
        color="#0b4f9c",
        edgecolors="none",
        zorder=6,
    )

    selected = selected_df.copy() if selected_df is not None else pd.DataFrame()
    if not selected.empty:
        ax.scatter(
            selected["x"].astype(float),
            selected["y"].astype(float),
            s=56,
            facecolor="#2ca02c",
            edgecolor="#111111",
            linewidth=0.9,
            zorder=8,
            label="selected top-k",
        )

    anchor_payload: list[dict[str, object]] = []
    if consensus_anchors:
        anchor_x = [float(item["x"]) for item in consensus_anchors]
        anchor_y = [float(item["y"]) for item in consensus_anchors]
        ax.scatter(
            anchor_x,
            anchor_y,
            s=120,
            marker="X",
            facecolor="#f58518",
            edgecolor="#111111",
            linewidth=0.8,
            zorder=9,
            label="consensus anchors",
        )
        for item in consensus_anchors:
            x = float(item["x"])
            y = float(item["y"])
            label = str(item.get("label") or item.get("tf") or "consensus")
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(6, -8),
                textcoords="offset points",
                fontsize=8,
                color="#3f3f3f",
            )
            anchor_payload.append(
                {
                    "tf": str(item.get("tf")),
                    "label": label,
                    "x": x,
                    "y": y,
                }
            )

    _set_axes_and_title(
        ax,
        x_metric=x_metric,
        y_metric=y_metric,
        tf_names=tf_names,
        score_scale=score_scale,
        suffix="story",
    )
    legend_labels = _legend_labels(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
    return {
        "mode": "story",
        "legend_labels": legend_labels,
        "consensus_anchors": anchor_payload,
        "best_path_points": int(len(best_path)),
    }


def _plot_chain_debug(
    ax: plt.Axes,
    chain_df: pd.DataFrame,
    *,
    label: str | None,
    color: str,
    alpha: float,
) -> None:
    x = chain_df["x"].astype(float).to_numpy()
    y = chain_df["y"].astype(float).to_numpy()
    if x.size == 0:
        return
    if x.size >= 2:
        ax.plot(x, y, color=color, alpha=alpha, linewidth=1.1, zorder=4)
    ax.scatter(x, y, s=14, color=color, alpha=min(0.95, alpha + 0.12), edgecolors="none", zorder=5, label=label)


def plot_opt_trajectory_debug(
    *,
    trajectory_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    tf_names: Iterable[str],
    out_path: Path,
    identity_mode: str | None,
    elite_centroid: tuple[float, float] | None,
    score_scale: str | None,
    dpi: int,
    png_compress_level: int,
    stride: int = 10,
    show_all_chains: bool = False,
) -> dict[str, object]:
    trajectory = _require_df(trajectory_df, name="Trajectory")
    baseline = _require_df(baseline_df, name="Baseline")
    x_metric = _first_metric(trajectory, "x_metric")
    y_metric = _first_metric(trajectory, "y_metric")
    if "is_cold_chain" not in trajectory.columns:
        raise ValueError("Trajectory points must include is_cold_chain for debug plot.")
    if "chain" not in trajectory.columns:
        raise ValueError("Trajectory points must include chain for debug plot.")

    base_x, base_y, _, _, _, _ = project_scores(baseline, tf_names)
    fig, ax = plt.subplots(figsize=(6.8, 5.1))
    ax.scatter(base_x, base_y, s=8, c="#c9c9c9", alpha=0.28, edgecolors="none", label="random baseline")

    plot_df = trajectory.copy().sort_values(["chain", "sweep"]).reset_index(drop=True)
    cold_rows = plot_df[plot_df["is_cold_chain"].astype(int) == 1]
    if cold_rows.empty:
        raise ValueError("Debug trajectory plot requires at least one cold-chain row.")
    cold_chain = int(cold_rows["chain"].iloc[0])
    chain_ids = sorted(int(v) for v in plot_df["chain"].unique())
    if show_all_chains:
        draw_chain_ids = chain_ids
    else:
        hot_candidates = [cid for cid in chain_ids if cid != cold_chain]
        draw_chain_ids = [cold_chain]
        if hot_candidates:
            draw_chain_ids.append(hot_candidates[0])

    first_context_label = True
    for chain_id in draw_chain_ids:
        chain_df = plot_df[plot_df["chain"].astype(int) == int(chain_id)].copy()
        chain_df = _sample_rows(chain_df, stride=max(1, int(stride)), group_col="chain")
        is_cold = int(chain_id) == cold_chain
        label = None
        if is_cold:
            label = f"cold chain (beta≈1, id={cold_chain})"
            _plot_chain_debug(ax, chain_df, label=label, color="#2f69a8", alpha=0.80)
        else:
            if first_context_label:
                label = "context chain(s)"
                first_context_label = False
            _plot_chain_debug(ax, chain_df, label=label, color="#7f7f7f", alpha=0.35)

    cold_phase_df = plot_df[plot_df["chain"].astype(int) == cold_chain]
    if "phase" in cold_phase_df.columns and not cold_phase_df.empty:
        phase_series = cold_phase_df["phase"].astype(str)
        has_tune = bool((phase_series == "tune").any())
        draw_indices = np.where(phase_series == "draw")[0]
        if has_tune and draw_indices.size:
            first_draw_idx = int(draw_indices[0])
            ax.scatter(
                [float(cold_phase_df["x"].iloc[first_draw_idx])],
                [float(cold_phase_df["y"].iloc[first_draw_idx])],
                s=40,
                marker="o",
                color="#222222",
                zorder=7,
                label="first draw (cold chain)",
            )

    if elite_centroid is not None:
        ax.scatter(
            [elite_centroid[0]],
            [elite_centroid[1]],
            s=84,
            marker="*",
            color="#f58518",
            zorder=8,
            label="elite median",
        )

    _set_axes_and_title(
        ax,
        x_metric=x_metric,
        y_metric=y_metric,
        tf_names=tf_names,
        score_scale=score_scale,
        suffix="debug",
        identity_mode=identity_mode,
    )
    legend_labels = _legend_labels(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
    return {
        "mode": "debug",
        "legend_labels": legend_labels,
        "cold_chain_id": cold_chain,
    }


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
    style: str = "story",
    selected_df: pd.DataFrame | None = None,
    consensus_anchors: list[dict[str, object]] | None = None,
    stride_story: int = 50,
    stride_debug: int = 10,
    baseline_mode: str = "hexbin",
    show_all_chains: bool = False,
) -> dict[str, object]:
    style_name = str(style).strip().lower()
    if style_name == "story":
        return plot_opt_trajectory_story(
            trajectory_df=trajectory_df,
            baseline_df=baseline_df,
            selected_df=selected_df,
            consensus_anchors=consensus_anchors,
            tf_names=tf_names,
            out_path=out_path,
            score_scale=score_scale,
            dpi=dpi,
            png_compress_level=png_compress_level,
            stride=stride_story,
            baseline_mode=baseline_mode,
        )
    if style_name == "debug":
        return plot_opt_trajectory_debug(
            trajectory_df=trajectory_df,
            baseline_df=baseline_df,
            tf_names=tf_names,
            out_path=out_path,
            identity_mode=identity_mode,
            elite_centroid=elite_centroid,
            score_scale=score_scale,
            dpi=dpi,
            png_compress_level=png_compress_level,
            stride=stride_debug,
            show_all_chains=show_all_chains,
        )
    raise ValueError(f"Unsupported trajectory plot style '{style}'. Use 'story' or 'debug'.")
