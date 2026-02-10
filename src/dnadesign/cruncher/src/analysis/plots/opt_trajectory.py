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


def _resolve_scatter_columns(*, tf_pair: tuple[str, str], scatter_scale: str) -> tuple[str, str, str]:
    scale = str(scatter_scale).strip().lower()
    if scale in {"llr", "raw-llr", "raw_llr"}:
        return f"raw_llr_{tf_pair[0]}", f"raw_llr_{tf_pair[1]}", "llr"
    if scale in {"normalized-llr", "norm-llr", "norm_llr"}:
        return f"norm_llr_{tf_pair[0]}", f"norm_llr_{tf_pair[1]}", "normalized-llr"
    raise ValueError("scatter_scale must be 'llr' or 'normalized-llr'.")


def _legend_labels(ax: plt.Axes) -> list[str]:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8, loc="best")
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


def _draw_chain_lineage(
    ax: plt.Axes,
    sampled: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    alpha_lo: float,
    alpha_hi: float,
) -> list[int]:
    chain_ids = sorted(int(v) for v in sampled["chain"].unique())
    if not chain_ids:
        raise ValueError("Trajectory plot requires at least one chain.")
    cmap = plt.get_cmap("tab20", max(1, len(chain_ids)))
    for idx, chain_id in enumerate(chain_ids):
        chain_df = sampled[sampled["chain"].astype(int) == chain_id].sort_values("sweep_idx")
        if chain_df.empty:
            continue
        x = chain_df[x_col].astype(float).to_numpy()
        y = chain_df[y_col].astype(float).to_numpy()
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
        ax.plot([], [], color=rgb, linewidth=1.4, label=f"chain {chain_id}")
    return chain_ids


def plot_chain_trajectory_scatter(
    *,
    trajectory_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    tf_pair: tuple[str, str],
    scatter_scale: str,
    consensus_anchors: list[dict[str, object]] | None,
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
    if len(tf_pair) != 2:
        raise ValueError("tf_pair must contain exactly two TF names.")

    x_col, y_col, normalized_scale = _resolve_scatter_columns(tf_pair=tf_pair, scatter_scale=scatter_scale)
    plot_df = _prepare_chain_df(trajectory_df)
    plot_df[x_col] = _require_numeric(plot_df, x_col, context="Trajectory")
    plot_df[y_col] = _require_numeric(plot_df, y_col, context="Trajectory")
    sampled = _sample_rows(plot_df, stride=max(1, int(stride)))

    baseline = _require_df(baseline_df, name="Baseline").copy()
    baseline[x_col] = _require_numeric(baseline, x_col, context="Baseline")
    baseline[y_col] = _require_numeric(baseline, y_col, context="Baseline")

    fig, ax = plt.subplots(figsize=(6.8, 5.1))
    ax.scatter(
        baseline[x_col].astype(float),
        baseline[y_col].astype(float),
        s=8,
        c="#c9c9c9",
        alpha=0.28,
        edgecolors="none",
        zorder=1,
        label=f"random baseline (n={len(baseline)})",
    )

    chain_ids = _draw_chain_lineage(
        ax,
        sampled,
        x_col=x_col,
        y_col=y_col,
        alpha_lo=alpha_lo,
        alpha_hi=alpha_hi,
    )

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
            label = str(item.get("label") or item.get("tf") or "consensus")
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(6, -8),
                textcoords="offset points",
                fontsize=8,
                color="#3f3f3f",
            )

    if slot_overlay:
        _draw_chain_overlay(ax, sampled, x_col=x_col, y_col=y_col)

    scale_label = "raw LLR" if normalized_scale == "llr" else "normalized LLR"
    ax.set_xlabel(f"{tf_pair[0]} ({scale_label})")
    ax.set_ylabel(f"{tf_pair[1]} ({scale_label})")
    ax.set_title(f"Chain trajectory ({tf_pair[0]} vs {tf_pair[1]})")
    legend_labels = _legend_labels(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
    return {
        "mode": "chain_scatter",
        "legend_labels": legend_labels,
        "chain_count": len(chain_ids),
        "x_column": x_col,
        "y_column": y_col,
    }


def plot_chain_trajectory_sweep(
    *,
    trajectory_df: pd.DataFrame,
    y_column: str,
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

    plot_df = _prepare_chain_df(trajectory_df)
    plot_df[y_column] = _require_numeric(plot_df, y_column, context="Trajectory")
    sampled = _sample_rows(plot_df, stride=max(1, int(stride)))

    fig, ax = plt.subplots(figsize=(6.8, 5.1))
    chain_ids = sorted(int(v) for v in sampled["chain"].unique())
    cmap = plt.get_cmap("tab20", max(1, len(chain_ids)))
    for idx, chain_id in enumerate(chain_ids):
        chain_df = sampled[sampled["chain"].astype(int) == chain_id].sort_values("sweep_idx")
        sweeps = chain_df["sweep_idx"].astype(float).to_numpy()
        values = chain_df[y_column].astype(float).to_numpy()
        rgb = cmap(idx % cmap.N)
        ax.plot(sweeps, values, color=rgb, linewidth=1.5, alpha=0.8, label=f"chain {chain_id}")
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
            ax.scatter(sweeps, values, s=18, c=colors, edgecolors="none", zorder=4)

    if slot_overlay:
        _draw_chain_overlay(ax, sampled, x_col="sweep_idx", y_col=y_column)

    ylabel = {
        "raw_llr_objective": "Raw LLR objective",
        "norm_llr_objective": "Normalized LLR objective",
        "objective_scalar": "Optimization score",
    }.get(y_column, y_column)
    ax.set_xlabel("Sweep index")
    ax.set_ylabel(ylabel)
    ax.set_title("Chain trajectory over sweeps")
    ax.text(
        0.02,
        0.98,
        "y = combined objective across TFs per sweep",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#4a4a4a",
    )
    legend_labels = _legend_labels(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
    return {
        "mode": "chain_sweep",
        "legend_labels": legend_labels,
        "chain_count": len(chain_ids),
        "y_column": y_column,
    }
