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

from dnadesign.cruncher.analysis.objective_labels import (
    objective_scalar_semantics,
)
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


def _sweep_ylabel(
    y_column: str,
    mode: str,
    *,
    objective_config: dict[str, object] | None,
) -> str:
    if mode not in {"raw", "best_so_far", "all"}:
        raise ValueError(f"Unsupported sweep mode '{mode}'.")
    if y_column == "objective_scalar":
        semantics = objective_scalar_semantics(objective_config)
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


def _append_summary_rows(
    *,
    summary_rows: list[dict[str, float]],
    sweeps: np.ndarray,
    values: np.ndarray,
) -> None:
    for sweep_value, y_value in zip(sweeps, values):
        summary_rows.append({"sweep_idx": float(sweep_value), "y": float(y_value)})


def _plot_chain_curve(
    *,
    ax: plt.Axes,
    mode: str,
    sweeps: np.ndarray,
    raw_values: np.ndarray,
    best_values: np.ndarray,
    color: str,
    alpha_lo: float,
    alpha_hi: float,
) -> np.ndarray:
    if mode == "raw":
        plot_values = raw_values
        ax.plot(sweeps, plot_values, color=color, linewidth=1.5, alpha=alpha_hi, zorder=3)
        return plot_values
    if mode == "best_so_far":
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
        return plot_values
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
    return plot_values


def _plot_chain_endpoints(
    *,
    ax: plt.Axes,
    sweeps: np.ndarray,
    plot_values: np.ndarray,
    marker: str,
    color: str,
) -> None:
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


def _add_summary_overlay(
    *,
    ax: plt.Axes,
    summary_rows: list[dict[str, float]],
    stride: int,
    summary_source: str,
    legend_handles: list[Line2D],
) -> tuple[bool, int, float | None]:
    summary_plot, summary_peak_value = _build_sweep_summary(summary_rows, stride=max(1, int(stride)))
    if summary_plot.empty:
        return False, 0, summary_peak_value
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
    return True, int(len(summary_plot)), summary_peak_value


def _sweep_title_base(y_column: str) -> str:
    if y_column == "objective_scalar":
        return "Soft-min TF best-window score over sweeps"
    if y_column == "raw_llr_objective":
        return "Replay objective over sweeps (raw-LLR)"
    if y_column == "norm_llr_objective":
        return "Replay objective over sweeps (normalized-LLR)"
    return "Objective over sweeps"


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


def _process_chain_plot(
    *,
    ax: plt.Axes,
    chain_df: pd.DataFrame,
    chain_id: int,
    chain_idx: int,
    y_column: str,
    stride: int,
    mode: str,
    alpha_lo: float,
    alpha_hi: float,
    marker: str,
    chain_overlay: bool,
    summary_overlay: bool,
    summary_source: str,
    summary_rows: list[dict[str, float]],
    legend_handles: list[Line2D],
) -> tuple[list[int], float | None]:
    full_sweeps, full_raw_values, full_best_values = _chain_sweep_arrays(chain_df, y_column=y_column)
    if summary_overlay:
        summary_values = full_raw_values if summary_source == "raw" else full_best_values
        _append_summary_rows(summary_rows=summary_rows, sweeps=full_sweeps, values=summary_values)
    sweeps, raw_values, best_values = _sample_chain_sweep_arrays(
        chain_df,
        y_column=y_column,
        stride=max(1, int(stride)),
        retain_best_updates=mode in {"best_so_far", "all"},
    )
    color = _CHAIN_COLORS[chain_idx % len(_CHAIN_COLORS)]
    plot_values = _plot_chain_curve(
        ax=ax,
        mode=mode,
        sweeps=sweeps,
        raw_values=raw_values,
        best_values=best_values,
        color=color,
        alpha_lo=alpha_lo,
        alpha_hi=alpha_hi,
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
        _plot_chain_endpoints(
            ax=ax,
            sweeps=sweeps,
            plot_values=plot_values,
            marker=marker,
            color=color,
        )
    best_final = float(best_values[-1]) if best_values.size else None
    return [int(v) for v in sweeps.tolist()], best_final


def _annotate_tune_boundary(ax: plt.Axes, tune_sweeps: int | None) -> int | None:
    if not isinstance(tune_sweeps, int) or tune_sweeps <= 0:
        return None
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
    return tune_boundary


def _annotate_cooling_markers(ax: plt.Axes, cooling_config: dict[str, object] | None) -> list[tuple[int, float]]:
    cooling_markers = _cooling_markers(cooling_config)
    if not cooling_markers:
        return cooling_markers
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
    return cooling_markers


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
        marker = marker_map[int(chain_id)]
        sampled_sweeps, best_final = _process_chain_plot(
            ax=ax,
            chain_df=chain_df,
            chain_id=int(chain_id),
            chain_idx=idx,
            y_column=y_column,
            mode=mode,
            stride=max(1, int(stride)),
            alpha_lo=alpha_lo,
            alpha_hi=alpha_hi,
            marker=marker,
            chain_overlay=chain_overlay,
            summary_overlay=summary_overlay,
            summary_source=summary_source,
            summary_rows=summary_rows,
            legend_handles=legend_handles,
        )
        sampled_sweep_indices_by_chain[int(chain_id)] = sampled_sweeps
        if best_final is not None:
            best_final_by_chain[int(chain_id)] = best_final

    if summary_overlay and len(chain_ids) >= 2:
        summary_overlay_enabled, summary_points, summary_peak_value = _add_summary_overlay(
            ax=ax,
            summary_rows=summary_rows,
            stride=max(1, int(stride)),
            summary_source=summary_source,
            legend_handles=legend_handles,
        )

    ylabel = _sweep_ylabel(y_column, mode, objective_config=objective_config)
    ax.set_xlabel("Sweep index", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    title_suffix = {
        "raw": "raw",
        "best_so_far": "best-so-far",
        "all": "raw + best-so-far",
    }[mode]
    title_base = _sweep_title_base(y_column)
    ax.set_title(f"{title_base} ({title_suffix})", fontsize=14)

    tune_boundary = _annotate_tune_boundary(ax, tune_sweeps)
    cooling_markers = _annotate_cooling_markers(ax, cooling_config)

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
