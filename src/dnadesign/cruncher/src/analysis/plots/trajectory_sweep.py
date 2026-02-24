"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/trajectory_sweep.py

Render sweep-space trajectory plots for optimization chains.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from dnadesign.cruncher.analysis.plots._savefig import savefig
from dnadesign.cruncher.analysis.plots._style import apply_axes_style
from dnadesign.cruncher.analysis.plots.trajectory_common import (
    _CHAIN_COLORS,
    _CHAIN_MARKERS,
    _best_update_indices,
    _prepare_chain_df,
    _require_numeric,
    _stride_indices,
)


def _score_scale_label(objective_config: dict[str, object] | None) -> str:
    cfg = objective_config if isinstance(objective_config, dict) else {}
    scale = str(cfg.get("score_scale") or "normalized-llr").strip().lower()
    if scale in {"llr", "raw-llr", "raw_llr"}:
        return "raw-LLR"
    if scale in {"normalized-llr", "norm-llr", "norm_llr"}:
        return "norm-LLR"
    if scale == "logp":
        return "logp"
    return scale


def _objective_scalar_semantics(objective_config: dict[str, object] | None) -> str:
    cfg = objective_config if isinstance(objective_config, dict) else {}
    combine = str(cfg.get("combine") or "min").strip().lower()
    scale_label = _score_scale_label(cfg)
    softmin_cfg = cfg.get("softmin")
    softmin_enabled = isinstance(softmin_cfg, dict) and bool(softmin_cfg.get("enabled"))
    if combine == "sum":
        return f"sum TF best-window {scale_label}"
    if combine == "min" and softmin_enabled:
        return f"soft-min TF best-window {scale_label}"
    return f"min TF best-window {scale_label}"


def _sweep_ylabel(
    y_column: str,
    mode: str,
    *,
    objective_config: dict[str, object] | None,
) -> str:
    if mode not in {"raw", "best_so_far", "all"}:
        raise ValueError(f"Unsupported sweep mode '{mode}'.")
    if y_column == "objective_scalar":
        semantics = _objective_scalar_semantics(objective_config)
        return semantics[:1].upper() + semantics[1:]
    if y_column == "raw_llr_objective":
        return "Replay objective (raw-LLR)"
    if y_column == "norm_llr_objective":
        return "Replay objective (norm-LLR)"
    return str(y_column)


def _cooling_markers(cooling_config: dict[str, object] | None) -> list[tuple[int, float]]:
    if not isinstance(cooling_config, dict):
        return []
    kind = str(cooling_config.get("kind") or "").strip().lower()
    if kind != "piecewise":
        return []
    stages = cooling_config.get("stages")
    if not isinstance(stages, list):
        return []
    points: list[tuple[int, float]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        sweeps = stage.get("sweeps")
        beta = stage.get("beta")
        if not isinstance(sweeps, (int, float)) or not isinstance(beta, (int, float)):
            continue
        points.append((int(sweeps), float(beta)))
    return sorted(points, key=lambda item: item[0])


def _summary_source_mode(mode: str) -> str:
    if mode == "raw":
        return "raw"
    if mode in {"best_so_far", "all"}:
        return "best_so_far"
    raise ValueError(f"Unsupported sweep mode '{mode}' for summary overlay.")


def _build_sweep_summary(
    rows: list[dict[str, float]],
    *,
    stride: int,
) -> tuple[pd.DataFrame, float | None]:
    if not rows:
        return pd.DataFrame(), None
    summary_df = pd.DataFrame(rows)
    grouped = summary_df.groupby("sweep_idx", sort=True, dropna=False)["y"]
    stats = grouped.quantile([0.25, 0.5, 0.75]).unstack(level=1).reset_index()
    stats = stats.rename(columns={0.25: "q25", 0.5: "median", 0.75: "q75"})
    stats = stats.sort_values("sweep_idx").reset_index(drop=True)
    peak_value = float(stats["median"].max()) if not stats.empty else None
    priority: list[int] = []
    if not stats.empty and np.isfinite(stats["median"].to_numpy(dtype=float)).any():
        priority = [int(np.nanargmax(stats["median"].to_numpy(dtype=float)))]
    keep_idx = _stride_indices(len(stats), stride=max(1, int(stride)), priority_indices=priority)
    return stats.iloc[keep_idx].reset_index(drop=True), peak_value


def _chain_sweep_arrays(
    chain_df: pd.DataFrame,
    *,
    y_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = chain_df.sort_values("sweep_idx")
    sweeps = ordered["sweep_idx"].astype(float).to_numpy()
    raw_values = ordered[y_column].astype(float).to_numpy()
    best_values = np.maximum.accumulate(raw_values) if raw_values.size else raw_values
    return sweeps, raw_values, best_values


def _sample_chain_sweep_arrays(
    chain_df: pd.DataFrame,
    *,
    y_column: str,
    stride: int,
    preserve_raw_max: bool = True,
    retain_best_updates: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sweeps, raw_values, best_values = _chain_sweep_arrays(chain_df, y_column=y_column)
    priority: list[int] = []
    if preserve_raw_max and raw_values.size and np.isfinite(raw_values).any():
        priority.append(int(np.nanargmax(raw_values)))
    if retain_best_updates:
        priority.extend(_best_update_indices(raw_values))
    keep_idx = _stride_indices(raw_values.size, stride=max(1, int(stride)), priority_indices=priority)
    return sweeps[keep_idx], raw_values[keep_idx], best_values[keep_idx]


def plot_chain_trajectory_sweep(
    *,
    trajectory_df: pd.DataFrame,
    y_column: str,
    y_mode: str = "best_so_far",
    objective_config: dict[str, object] | None = None,
    cooling_config: dict[str, object] | None = None,
    tune_sweeps: int | None = None,
    objective_caption: str | None = None,
    out_path: Path,
    dpi: int,
    png_compress_level: int,
    stride: int = 10,
    alpha_min: float = 0.25,
    alpha_max: float = 0.95,
    chain_overlay: bool = False,
    summary_overlay: bool = False,
) -> dict[str, object]:
    if not isinstance(alpha_min, (int, float)) or not isinstance(alpha_max, (int, float)):
        raise ValueError("Chain alpha bounds must be numeric.")
    alpha_lo = float(alpha_min)
    alpha_hi = float(alpha_max)
    if alpha_lo < 0 or alpha_hi > 1 or alpha_lo > alpha_hi:
        raise ValueError("Chain alpha bounds must satisfy 0 <= min <= max <= 1.")
    mode = str(y_mode).strip().lower()
    if mode not in {"raw", "best_so_far", "all"}:
        raise ValueError("y_mode must be one of: raw, best_so_far, all.")

    plot_df = _prepare_chain_df(trajectory_df)
    plot_df[y_column] = _require_numeric(plot_df, y_column, context="Trajectory")
    plot_df = plot_df.sort_values(["chain", "sweep_idx"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    chain_ids = sorted(int(v) for v in plot_df["chain"].unique())
    marker_map = {chain_id: _CHAIN_MARKERS[idx % len(_CHAIN_MARKERS)] for idx, chain_id in enumerate(chain_ids)}
    legend_handles: list[Line2D] = []
    best_final_by_chain: dict[int, float] = {}
    sampled_sweep_indices_by_chain: dict[int, list[int]] = {}
    summary_rows: list[dict[str, float]] = []
    summary_source = _summary_source_mode(mode)
    summary_overlay_enabled = False
    summary_points = 0
    summary_peak_value: float | None = None
    best_drawstyle = "steps-post" if mode in {"best_so_far", "all"} else "default"
    for idx, chain_id in enumerate(chain_ids):
        chain_df = plot_df[plot_df["chain"].astype(int) == chain_id].sort_values("sweep_idx")
        full_sweeps, full_raw_values, full_best_values = _chain_sweep_arrays(chain_df, y_column=y_column)
        if summary_overlay:
            summary_values = full_raw_values if summary_source == "raw" else full_best_values
            for sweep_value, y_value in zip(full_sweeps, summary_values):
                summary_rows.append({"sweep_idx": float(sweep_value), "y": float(y_value)})
        sweeps, raw_values, best_values = _sample_chain_sweep_arrays(
            chain_df,
            y_column=y_column,
            stride=max(1, int(stride)),
            retain_best_updates=mode in {"best_so_far", "all"},
        )
        sampled_sweep_indices_by_chain[int(chain_id)] = [int(v) for v in sweeps.tolist()]
        if best_values.size:
            best_final_by_chain[int(chain_id)] = float(best_values[-1])
        color = _CHAIN_COLORS[idx % len(_CHAIN_COLORS)]
        marker = marker_map[int(chain_id)]
        if mode == "raw":
            plot_values = raw_values
            ax.plot(sweeps, plot_values, color=color, linewidth=1.5, alpha=alpha_hi, zorder=3)
        elif mode == "best_so_far":
            plot_values = best_values
            ax.step(
                sweeps,
                plot_values,
                where="post",
                color=color,
                linewidth=1.8,
                alpha=alpha_hi,
                zorder=4,
            )
        else:
            plot_values = best_values
            ax.plot(sweeps, raw_values, color=color, linewidth=1.0, alpha=max(0.08, alpha_lo * 0.6), zorder=2)
            ax.step(
                sweeps,
                plot_values,
                where="post",
                color=color,
                linewidth=1.9,
                alpha=alpha_hi,
                zorder=4,
            )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.8,
                label=f"Chain {chain_id}",
            )
        )
        if chain_overlay and sweeps.size:
            ax.scatter(
                [float(sweeps[0])],
                [float(plot_values[0])],
                s=34,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=1.0,
                zorder=5,
            )
            ax.scatter(
                [float(sweeps[-1])],
                [float(plot_values[-1])],
                s=42,
                marker=marker,
                c=color,
                edgecolors="#111111",
                linewidths=0.6,
                zorder=5,
            )

    if summary_overlay and len(chain_ids) >= 2:
        summary_plot, summary_peak_value = _build_sweep_summary(summary_rows, stride=max(1, int(stride)))
        if not summary_plot.empty:
            sweeps_summary = summary_plot["sweep_idx"].astype(float).to_numpy()
            q25_summary = summary_plot["q25"].astype(float).to_numpy()
            q75_summary = summary_plot["q75"].astype(float).to_numpy()
            median_summary = summary_plot["median"].astype(float).to_numpy()
            ax.fill_between(
                sweeps_summary,
                q25_summary,
                q75_summary,
                color="#6f6f6f",
                alpha=0.10,
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(
                sweeps_summary,
                median_summary,
                color="#5a5a5a",
                linewidth=1.1,
                linestyle="--",
                alpha=0.72,
                zorder=1,
            )
            summary_overlay_enabled = True
            summary_points = int(len(summary_plot))
            if summary_source == "best_so_far":
                summary_label = "Median best-so-far across chains"
            else:
                summary_label = "Median across chains"
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#5a5a5a",
                    linewidth=1.1,
                    linestyle="--",
                    label=summary_label,
                )
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#6f6f6f",
                    linewidth=6.0,
                    alpha=0.20,
                    label="IQR across chains",
                )
            )

    ylabel = _sweep_ylabel(y_column, mode, objective_config=objective_config)
    ax.set_xlabel("Sweep index", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    title_suffix = {
        "raw": "raw",
        "best_so_far": "best-so-far",
        "all": "raw + best-so-far",
    }[mode]
    if y_column == "objective_scalar":
        title_base = "Soft-min TF best-window score over sweeps"
    elif y_column == "raw_llr_objective":
        title_base = "Replay objective over sweeps (raw-LLR)"
    elif y_column == "norm_llr_objective":
        title_base = "Replay objective over sweeps (normalized-LLR)"
    else:
        title_base = "Objective over sweeps"
    ax.set_title(f"{title_base} ({title_suffix})", fontsize=14)

    tune_boundary = None
    if isinstance(tune_sweeps, int) and tune_sweeps > 0:
        tune_boundary = int(tune_sweeps)
        ax.axvline(tune_boundary, color="#777777", linestyle="--", linewidth=1.0, alpha=0.75)
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        y_text = y_min + ((y_max - y_min) * 0.12)
        x_text = tune_boundary + ((x_max - x_min) * 0.008)
        ax.text(
            x_text,
            y_text,
            "Tune end",
            ha="left",
            va="bottom",
            fontsize=10,
            color="#555555",
        )

    cooling_markers = _cooling_markers(cooling_config)
    if cooling_markers:
        y_min, y_max = ax.get_ylim()
        y_text = y_min + ((y_max - y_min) * 0.02)
        for sweep_boundary, beta_value in cooling_markers[:-1]:
            ax.axvline(sweep_boundary, color="#aaaaaa", linestyle=":", linewidth=0.9, alpha=0.7)
            ax.text(
                sweep_boundary,
                y_text,
                f" β={beta_value:g}",
                ha="left",
                va="bottom",
                fontsize=10,
                color="#666666",
            )

    apply_axes_style(ax, ygrid=True, xgrid=False, tick_labelsize=12, title_size=14, label_size=14)
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, fontsize=10, loc="center right")
    legend_labels = [handle.get_label() for handle in legend_handles]
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level, bbox_inches=None)
    plt.close(fig)
    return {
        "mode": "chain_sweep",
        "y_mode": mode,
        "legend_labels": legend_labels,
        "chain_count": len(chain_ids),
        "y_column": y_column,
        "y_label": ylabel,
        "best_final_by_chain": best_final_by_chain,
        "summary_overlay_enabled": summary_overlay_enabled,
        "summary_points": summary_points,
        "summary_source": summary_source,
        "summary_peak_value": summary_peak_value,
        "sampled_sweep_indices_by_chain": sampled_sweep_indices_by_chain,
        "best_drawstyle": best_drawstyle,
        "tune_boundary_sweep": tune_boundary,
        "cooling_stage_count": len(cooling_markers),
        "objective_caption": str(objective_caption or ""),
    }
