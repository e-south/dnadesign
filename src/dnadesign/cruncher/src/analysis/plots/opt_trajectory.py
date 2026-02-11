"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/opt_trajectory.py

Plot optimization trajectories in TF score-space and sweep-space for independent chains.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.plots._savefig import savefig
from dnadesign.cruncher.analysis.plots._style import apply_axes_style, place_figure_caption


def _require_df(df: pd.DataFrame | None, *, name: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"{name} data is required for trajectory plot.")
    return df


def _require_numeric(df: pd.DataFrame, column: str, *, context: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"{context} missing required column '{column}'.")
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{context} column '{column}' must be numeric.")
    return values.astype(float)


def _prepare_chain_df(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = _require_df(trajectory_df, name="Trajectory").copy()
    if "chain" not in plot_df.columns:
        raise ValueError("Trajectory points must include chain for chain trajectory plotting.")
    chain_values = _require_numeric(plot_df, "chain", context="Trajectory")
    if "sweep_idx" in plot_df.columns:
        sweep_values = _require_numeric(plot_df, "sweep_idx", context="Trajectory")
    elif "sweep" in plot_df.columns:
        sweep_values = _require_numeric(plot_df, "sweep", context="Trajectory")
    else:
        raise ValueError("Trajectory points must include sweep_idx for trajectory plotting.")
    plot_df["chain"] = chain_values.astype(int)
    plot_df["sweep_idx"] = sweep_values.astype(int)
    return plot_df


def _sample_rows(df: pd.DataFrame, *, stride: int, group_col: str = "chain") -> pd.DataFrame:
    if stride <= 1 or df.empty:
        return df
    if group_col not in df.columns:
        raise ValueError(f"Trajectory points missing required group column '{group_col}'.")
    sampled_parts: list[pd.DataFrame] = []
    for _, group in df.groupby(group_col, sort=True, dropna=False):
        ordered = group.sort_values("sweep_idx").reset_index(drop=True)
        idx = list(range(0, len(ordered), stride))
        if len(ordered) - 1 not in idx:
            idx.append(len(ordered) - 1)
        sampled_parts.append(ordered.iloc[sorted(set(idx))])
    return pd.concat(sampled_parts).sort_values([group_col, "sweep_idx"]).reset_index(drop=True)


def _select_best_so_far_updates(sampled: pd.DataFrame, *, objective_column: str = "objective_scalar") -> pd.DataFrame:
    updates = sampled.copy()
    updates[objective_column] = _require_numeric(updates, objective_column, context="Trajectory")
    parts: list[pd.DataFrame] = []
    for _, chain_df in updates.groupby("chain", sort=True, dropna=False):
        ordered = chain_df.sort_values("sweep_idx").copy()
        scores = ordered[objective_column].astype(float)
        previous_best = scores.cummax().shift(1, fill_value=float("-inf"))
        keep = scores > previous_best
        parts.append(ordered.loc[keep])
    return pd.concat(parts).sort_values(["chain", "sweep_idx"]).reset_index(drop=True)


def _resolve_scatter_columns(*, tf_pair: tuple[str, str], scatter_scale: str) -> tuple[str, str, str]:
    scale = str(scatter_scale).strip().lower()
    if scale in {"llr", "raw-llr", "raw_llr"}:
        return f"raw_llr_{tf_pair[0]}", f"raw_llr_{tf_pair[1]}", "llr"
    if scale in {"normalized-llr", "norm-llr", "norm_llr"}:
        return f"norm_llr_{tf_pair[0]}", f"norm_llr_{tf_pair[1]}", "normalized-llr"
    raise ValueError("scatter_scale must be 'llr' or 'normalized-llr'.")


def _clean_anchor_label(label: object) -> str:
    text = str(label or "consensus anchor").strip()
    return text.replace("(max)", "").replace("  ", " ").strip()


def _sweep_ylabel(y_column: str, mode: str) -> str:
    base = {
        "raw_llr_objective": "best-window raw LLR",
        "norm_llr_objective": "best-window normalized LLR",
        "objective_scalar": "optimizer scalar objective",
    }.get(y_column, y_column)
    suffix = {
        "raw": "per sweep",
        "best_so_far": "best-so-far",
        "all": "raw + best-so-far",
    }[mode]
    return f"Joint score ({base}; {suffix})"


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


def _legend_labels(ax: plt.Axes, *, location: str = "best") -> list[str]:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8, loc=location)
    return labels


def _draw_chain_overlay(ax: plt.Axes, sampled: pd.DataFrame, *, x_col: str, y_col: str) -> None:
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    chain_ids = sorted(int(chain) for chain in sampled["chain"].astype(int).unique())
    marker_map = {chain: marker_cycle[idx % len(marker_cycle)] for idx, chain in enumerate(chain_ids)}
    for chain_id in chain_ids:
        chain_df = sampled[sampled["chain"].astype(int) == chain_id]
        ax.scatter(
            chain_df[x_col].astype(float),
            chain_df[y_col].astype(float),
            s=20,
            marker=marker_map[chain_id],
            facecolors="none",
            edgecolors="#222222",
            linewidth=0.45,
            alpha=0.35,
            zorder=6,
        )
    mapping = ", ".join(f"{chain}:{marker}" for chain, marker in marker_map.items())
    ax.text(
        0.02,
        0.02,
        f"chain marker map: {mapping}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#4a4a4a",
    )


def _overlay_selected_elites(
    ax: plt.Axes,
    elites_df: pd.DataFrame | None,
    *,
    x_col: str,
    y_col: str,
) -> int:
    if elites_df is None or elites_df.empty:
        return 0
    if x_col not in elites_df.columns or y_col not in elites_df.columns:
        return 0
    x_vals = pd.to_numeric(elites_df[x_col], errors="coerce")
    y_vals = pd.to_numeric(elites_df[y_col], errors="coerce")
    valid = x_vals.notna() & y_vals.notna()
    if not bool(valid.any()):
        return 0
    x = x_vals[valid].astype(float).to_numpy()
    y = y_vals[valid].astype(float).to_numpy()
    elite_rows = elites_df.loc[valid].copy()
    ax.scatter(
        x,
        y,
        s=42,
        marker="o",
        c="#2f7f3f",
        edgecolors="#111111",
        linewidths=0.8,
        zorder=9,
        label=f"selected elites (n={len(elite_rows)})",
    )
    if "rank" in elite_rows.columns:
        for x_val, y_val, rank in zip(x, y, elite_rows["rank"], strict=False):
            try:
                rank_label = int(rank)
            except (TypeError, ValueError):
                continue
            ax.annotate(
                str(rank_label),
                xy=(float(x_val), float(y_val)),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#1f1f1f",
            )
    return int(len(elite_rows))


def _draw_chain_lineage(
    ax: plt.Axes,
    sampled: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    alpha_lo: float,
    alpha_hi: float,
    chain_label_limit: int = 8,
) -> tuple[list[int], dict[int, int]]:
    chain_ids = sorted(int(v) for v in sampled["chain"].unique())
    if not chain_ids:
        raise ValueError("Trajectory plot requires at least one chain.")
    points_by_chain: dict[int, int] = {}
    cmap = plt.get_cmap("tab20", max(1, len(chain_ids)))
    label_each_chain = len(chain_ids) <= int(chain_label_limit)
    for idx, chain_id in enumerate(chain_ids):
        chain_df = sampled[sampled["chain"].astype(int) == chain_id].sort_values("sweep_idx")
        if chain_df.empty:
            points_by_chain[int(chain_id)] = 0
            continue
        x = chain_df[x_col].astype(float).to_numpy()
        y = chain_df[y_col].astype(float).to_numpy()
        points_by_chain[int(chain_id)] = int(x.size)
        sweeps = chain_df["sweep_idx"].astype(float).to_numpy()
        sweep_min = float(sweeps.min())
        sweep_span = float(sweeps.max() - sweep_min)
        rgb = cmap(idx % cmap.N)
        if x.size >= 2:
            for seg_idx in range(1, x.size):
                if sweep_span <= 0:
                    alpha = 0.60
                else:
                    t = (float(sweeps[seg_idx]) - sweep_min) / sweep_span
                    alpha = alpha_lo + (alpha_hi - alpha_lo) * t
                ax.plot(
                    x[seg_idx - 1 : seg_idx + 1],
                    y[seg_idx - 1 : seg_idx + 1],
                    color=(rgb[0], rgb[1], rgb[2], alpha),
                    linewidth=1.2,
                    zorder=4,
                )
        if sweep_span <= 0:
            alphas = np.full(x.size, 0.60, dtype=float)
        else:
            alphas = alpha_lo + (alpha_hi - alpha_lo) * ((sweeps - sweep_min) / sweep_span)
        colors = np.column_stack(
            [
                np.full(x.size, rgb[0], dtype=float),
                np.full(x.size, rgb[1], dtype=float),
                np.full(x.size, rgb[2], dtype=float),
                np.asarray(alphas, dtype=float),
            ]
        )
        ax.scatter(x, y, s=16, c=colors, edgecolors="none", zorder=5)
        ax.scatter(
            [x[-1]],
            [y[-1]],
            s=38,
            facecolors=[rgb],
            edgecolors="#111111",
            linewidths=0.8,
            zorder=7,
        )
        if label_each_chain:
            ax.plot([], [], color=rgb, linewidth=1.4, label=f"chain {chain_id}")
        elif idx == 0:
            ax.plot([], [], color=rgb, linewidth=1.4, label=f"chains (n={len(chain_ids)})")
    return chain_ids, points_by_chain


def plot_chain_trajectory_scatter(
    *,
    trajectory_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    elites_df: pd.DataFrame | None = None,
    tf_pair: tuple[str, str],
    scatter_scale: str,
    consensus_anchors: list[dict[str, object]] | None,
    objective_caption: str | None = None,
    out_path: Path,
    dpi: int,
    png_compress_level: int,
    stride: int = 10,
    alpha_min: float = 0.15,
    alpha_max: float = 0.95,
    slot_overlay: bool = False,
    chain_label_limit: int = 8,
) -> dict[str, object]:
    if not isinstance(alpha_min, (int, float)) or not isinstance(alpha_max, (int, float)):
        raise ValueError("Chain alpha bounds must be numeric.")
    alpha_lo = float(alpha_min)
    alpha_hi = float(alpha_max)
    if alpha_lo < 0 or alpha_hi > 1 or alpha_lo > alpha_hi:
        raise ValueError("Chain alpha bounds must satisfy 0 <= min <= max <= 1.")
    if len(tf_pair) != 2:
        raise ValueError("tf_pair must contain exactly two TF names.")

    x_col, y_col, normalized_scale = _resolve_scatter_columns(tf_pair=tf_pair, scatter_scale=scatter_scale)
    plot_df = _prepare_chain_df(trajectory_df)
    plot_df[x_col] = _require_numeric(plot_df, x_col, context="Trajectory")
    plot_df[y_col] = _require_numeric(plot_df, y_col, context="Trajectory")
    sampled = plot_df.sort_values(["chain", "sweep_idx"]).reset_index(drop=True)
    best_updates = _select_best_so_far_updates(sampled, objective_column="objective_scalar")
    if best_updates.empty:
        raise ValueError("Trajectory plot requires at least one best-so-far update per chain.")

    baseline = _require_df(baseline_df, name="Baseline").copy()
    baseline[x_col] = _require_numeric(baseline, x_col, context="Baseline")
    baseline[y_col] = _require_numeric(baseline, y_col, context="Baseline")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(
        baseline[x_col].astype(float),
        baseline[y_col].astype(float),
        s=7,
        c="#c9c9c9",
        alpha=0.20,
        edgecolors="none",
        zorder=1,
        label=f"random baseline (n={len(baseline)})",
    )

    chain_ids, points_by_chain = _draw_chain_lineage(
        ax,
        best_updates,
        x_col=x_col,
        y_col=y_col,
        alpha_lo=alpha_lo,
        alpha_hi=alpha_hi,
        chain_label_limit=chain_label_limit,
    )

    elite_points_plotted = _overlay_selected_elites(ax, elites_df, x_col=x_col, y_col=y_col)

    if consensus_anchors:
        anchor_x = [float(item["x"]) for item in consensus_anchors]
        anchor_y = [float(item["y"]) for item in consensus_anchors]
        ax.scatter(
            anchor_x,
            anchor_y,
            s=120,
            marker="*",
            facecolor="#f58518",
            edgecolor="#111111",
            linewidth=0.8,
            zorder=8,
            label="consensus anchors",
        )
        for item in consensus_anchors:
            x = float(item["x"])
            y = float(item["y"])
            label = _clean_anchor_label(item.get("label") or item.get("tf") or "consensus anchor")
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(6, -8),
                textcoords="offset points",
                fontsize=8,
                color="#3f3f3f",
            )

    if slot_overlay:
        _draw_chain_overlay(ax, best_updates, x_col=x_col, y_col=y_col)

    scale_label = "raw LLR" if normalized_scale == "llr" else "normalized LLR"
    ax.set_xlabel(f"{tf_pair[0]} best-window {scale_label}")
    ax.set_ylabel(f"{tf_pair[1]} best-window {scale_label}")
    ax.set_title(f"Chain trajectory ({tf_pair[0]} vs {tf_pair[1]})")
    apply_axes_style(ax, ygrid=True)
    legend_labels = _legend_labels(ax, location="upper left")
    place_figure_caption(fig, objective_caption)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
    return {
        "mode": "chain_scatter",
        "scatter_mode": "best_so_far_updates",
        "legend_labels": legend_labels,
        "chain_count": len(chain_ids),
        "x_column": x_col,
        "y_column": y_col,
        "plotted_points_by_chain": points_by_chain,
        "elite_points_plotted": elite_points_plotted,
        "objective_caption": str(objective_caption or ""),
    }


def plot_chain_trajectory_sweep(
    *,
    trajectory_df: pd.DataFrame,
    y_column: str,
    y_mode: str = "best_so_far",
    cooling_config: dict[str, object] | None = None,
    tune_sweeps: int | None = None,
    objective_caption: str | None = None,
    out_path: Path,
    dpi: int,
    png_compress_level: int,
    stride: int = 10,
    alpha_min: float = 0.15,
    alpha_max: float = 0.95,
    slot_overlay: bool = False,
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
    sampled = _sample_rows(plot_df, stride=max(1, int(stride)))
    sampled_plot = sampled.copy()
    sampled_plot["_y_plot"] = np.nan

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    chain_ids = sorted(int(v) for v in sampled["chain"].unique())
    cmap = plt.get_cmap("tab20", max(1, len(chain_ids)))
    for idx, chain_id in enumerate(chain_ids):
        chain_df = sampled[sampled["chain"].astype(int) == chain_id].sort_values("sweep_idx")
        sweeps = chain_df["sweep_idx"].astype(float).to_numpy()
        raw_values = chain_df[y_column].astype(float).to_numpy()
        best_values = np.maximum.accumulate(raw_values) if raw_values.size else raw_values
        rgb = cmap(idx % cmap.N)
        if mode == "raw":
            plot_values = raw_values
            ax.plot(sweeps, plot_values, color=rgb, linewidth=1.5, alpha=0.8, label=f"chain {chain_id}")
            if sweeps.size:
                alphas = np.linspace(alpha_lo, alpha_hi, sweeps.size)
                colors = np.column_stack(
                    [
                        np.full(sweeps.size, rgb[0], dtype=float),
                        np.full(sweeps.size, rgb[1], dtype=float),
                        np.full(sweeps.size, rgb[2], dtype=float),
                        np.asarray(alphas, dtype=float),
                    ]
                )
                ax.scatter(sweeps, plot_values, s=18, c=colors, edgecolors="none", zorder=4)
        elif mode == "best_so_far":
            plot_values = best_values
            ax.plot(sweeps, plot_values, color=rgb, linewidth=1.6, alpha=0.9, label=f"chain {chain_id}")
            if sweeps.size:
                alphas = np.linspace(alpha_lo, alpha_hi, sweeps.size)
                colors = np.column_stack(
                    [
                        np.full(sweeps.size, rgb[0], dtype=float),
                        np.full(sweeps.size, rgb[1], dtype=float),
                        np.full(sweeps.size, rgb[2], dtype=float),
                        np.asarray(alphas, dtype=float),
                    ]
                )
                ax.scatter(sweeps, plot_values, s=18, c=colors, edgecolors="none", zorder=4)
        else:
            plot_values = best_values
            ax.plot(sweeps, raw_values, color=rgb, linewidth=1.0, alpha=0.28, zorder=2)
            ax.plot(sweeps, plot_values, color=rgb, linewidth=1.8, alpha=0.95, label=f"chain {chain_id}", zorder=4)
            if sweeps.size:
                raw_alphas = np.linspace(max(0.08, alpha_lo * 0.5), max(0.2, alpha_hi * 0.5), sweeps.size)
                raw_colors = np.column_stack(
                    [
                        np.full(sweeps.size, rgb[0], dtype=float),
                        np.full(sweeps.size, rgb[1], dtype=float),
                        np.full(sweeps.size, rgb[2], dtype=float),
                        np.asarray(raw_alphas, dtype=float),
                    ]
                )
                best_alphas = np.linspace(alpha_lo, alpha_hi, sweeps.size)
                best_colors = np.column_stack(
                    [
                        np.full(sweeps.size, rgb[0], dtype=float),
                        np.full(sweeps.size, rgb[1], dtype=float),
                        np.full(sweeps.size, rgb[2], dtype=float),
                        np.asarray(best_alphas, dtype=float),
                    ]
                )
                ax.scatter(sweeps, raw_values, s=10, c=raw_colors, edgecolors="none", zorder=3)
                ax.scatter(sweeps, plot_values, s=18, c=best_colors, edgecolors="none", zorder=5)

        sampled_plot.loc[chain_df.index, "_y_plot"] = plot_values

    if slot_overlay:
        _draw_chain_overlay(ax, sampled_plot, x_col="sweep_idx", y_col="_y_plot")

    ylabel = _sweep_ylabel(y_column, mode)
    ax.set_xlabel("Sweep index")
    ax.set_ylabel(ylabel)
    title_suffix = {
        "raw": "raw",
        "best_so_far": "best-so-far",
        "all": "raw + best-so-far",
    }[mode]
    ax.set_title(f"Joint objective over sweeps ({title_suffix})")

    tune_boundary = None
    if isinstance(tune_sweeps, int) and tune_sweeps > 0:
        tune_boundary = int(tune_sweeps)
        ax.axvline(tune_boundary, color="#777777", linestyle="--", linewidth=1.0, alpha=0.75)
        y_max = ax.get_ylim()[1]
        ax.text(
            tune_boundary,
            y_max,
            " tune end",
            ha="left",
            va="top",
            fontsize=8,
            color="#555555",
        )

    cooling_markers = _cooling_markers(cooling_config)
    if cooling_markers:
        y_max = ax.get_ylim()[1]
        for sweep_boundary, beta_value in cooling_markers[:-1]:
            ax.axvline(sweep_boundary, color="#aaaaaa", linestyle=":", linewidth=0.9, alpha=0.7)
            ax.text(
                sweep_boundary,
                y_max,
                f" β={beta_value:g}",
                ha="left",
                va="bottom",
                fontsize=7,
                color="#666666",
            )

    apply_axes_style(ax, ygrid=True)
    legend_labels = _legend_labels(ax, location="upper left")
    place_figure_caption(fig, objective_caption)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
    return {
        "mode": "chain_sweep",
        "y_mode": mode,
        "legend_labels": legend_labels,
        "chain_count": len(chain_ids),
        "y_column": y_column,
        "y_label": ylabel,
        "tune_boundary_sweep": tune_boundary,
        "cooling_stage_count": len(cooling_markers),
        "objective_caption": str(objective_caption or ""),
    }
