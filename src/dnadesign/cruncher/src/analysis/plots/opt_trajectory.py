"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/opt_trajectory.py

Plot optimization trajectory lineage in TF score-space and sweep-space views.

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
        raise ValueError(f"{name} data is required for opt trajectory plot.")
    return df


def _require_numeric(df: pd.DataFrame, column: str, *, context: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"{context} missing required column '{column}'.")
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{context} column '{column}' must be numeric.")
    return values.astype(float)


def _prepare_lineage_df(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = _require_df(trajectory_df, name="Trajectory").copy()
    particle_values = _require_numeric(plot_df, "particle_id", context="Trajectory")
    if "sweep_idx" in plot_df.columns:
        sweep_values = _require_numeric(plot_df, "sweep_idx", context="Trajectory")
    elif "sweep" in plot_df.columns:
        sweep_values = _require_numeric(plot_df, "sweep", context="Trajectory")
    else:
        raise ValueError("Trajectory points must include sweep_idx for trajectory plotting.")
    plot_df["particle_id"] = particle_values.astype(int)
    plot_df["sweep_idx"] = sweep_values.astype(int)
    if "slot_id" in plot_df.columns:
        slot_values = pd.to_numeric(plot_df["slot_id"], errors="coerce")
        if slot_values.notna().any() and slot_values.isna().any():
            raise ValueError("Trajectory slot_id must be numeric when provided.")
        plot_df["slot_id"] = slot_values
    else:
        plot_df["slot_id"] = np.nan
    return plot_df


def _sample_rows(df: pd.DataFrame, *, stride: int, group_col: str = "particle_id") -> pd.DataFrame:
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


def _sample_rows_by_sweep_stride(df: pd.DataFrame, *, stride: int) -> pd.DataFrame:
    if stride <= 1 or df.empty:
        return df
    ordered = df.sort_values("sweep_idx").reset_index(drop=True)
    idx = list(range(0, len(ordered), stride))
    if len(ordered) - 1 not in idx:
        idx.append(len(ordered) - 1)
    return ordered.iloc[sorted(set(idx))].reset_index(drop=True)


def _extract_cold_slot_path(df: pd.DataFrame) -> pd.DataFrame:
    if "is_cold_chain" not in df.columns:
        raise ValueError("Trajectory missing required column 'is_cold_chain' for cold-slot sweep trajectory.")
    cold_values = pd.to_numeric(df["is_cold_chain"], errors="coerce")
    if cold_values.isna().any():
        raise ValueError("Trajectory column 'is_cold_chain' must be numeric.")
    cold_mask = cold_values.astype(int)
    invalid = ~cold_mask.isin([0, 1])
    if invalid.any():
        raise ValueError("Trajectory column 'is_cold_chain' must contain only 0/1 values.")
    cold_df = df.loc[cold_mask == 1].copy()
    if cold_df.empty:
        raise ValueError("Trajectory contains no cold-slot rows (is_cold_chain=1).")
    cold_df = cold_df.sort_values("sweep_idx").drop_duplicates(["sweep_idx"], keep="last").reset_index(drop=True)
    return cold_df


def _infer_bottleneck_tf(
    cold_df: pd.DataFrame,
    *,
    y_column: str,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float, float, float]]]:
    prefix_by_y = {
        "raw_llr_objective": "raw_llr_",
        "norm_llr_objective": "norm_llr_",
        "objective_scalar": "score_",
    }
    prefix = prefix_by_y.get(y_column)
    if prefix is None:
        return cold_df, {}
    tf_cols = [col for col in cold_df.columns if col.startswith(prefix) and not col.endswith("_objective")]
    if not tf_cols:
        return cold_df, {}
    tf_score_df = pd.DataFrame(index=cold_df.index)
    for col in tf_cols:
        tf_score_df[col] = pd.to_numeric(cold_df[col], errors="coerce")
    valid_rows = ~tf_score_df.isna().any(axis=1)
    if not valid_rows.any():
        return cold_df, {}
    bottleneck_col = tf_score_df[valid_rows].idxmin(axis=1)
    cold_df = cold_df.copy()
    cold_df.loc[valid_rows, "bottleneck_tf"] = bottleneck_col.str[len(prefix) :]
    unique_tfs = sorted(str(tf) for tf in cold_df["bottleneck_tf"].dropna().astype(str).unique())
    if not unique_tfs:
        return cold_df, {}
    cmap = plt.get_cmap("tab10", max(1, len(unique_tfs)))
    color_map: dict[str, tuple[float, float, float, float]] = {}
    for idx, tf_name in enumerate(unique_tfs):
        color_map[tf_name] = tuple(cmap(idx % cmap.N))
    return cold_df, color_map


def _legend_labels(ax: plt.Axes) -> list[str]:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8, loc="best")
    return labels


def _resolve_scatter_columns(*, tf_pair: tuple[str, str], scatter_scale: str) -> tuple[str, str, str]:
    scale = str(scatter_scale).strip().lower()
    if scale in {"llr", "raw-llr", "raw_llr"}:
        return f"raw_llr_{tf_pair[0]}", f"raw_llr_{tf_pair[1]}", "llr"
    if scale in {"normalized-llr", "norm-llr", "norm_llr"}:
        return f"norm_llr_{tf_pair[0]}", f"norm_llr_{tf_pair[1]}", "normalized-llr"
    raise ValueError("scatter_scale must be 'llr' or 'normalized-llr'.")


def _draw_particle_lineage(
    ax: plt.Axes,
    sampled: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    alpha_lo: float,
    alpha_hi: float,
) -> tuple[list[int], float, float]:
    particle_ids = sorted(int(v) for v in sampled["particle_id"].unique())
    if not particle_ids:
        raise ValueError("Trajectory plot requires at least one particle_id.")
    sweep_float = sampled["sweep_idx"].astype(float)
    sweep_min = float(sweep_float.min())
    sweep_span = float(sweep_float.max() - sweep_min)
    cmap = plt.get_cmap("tab20", max(1, len(particle_ids)))
    for idx, particle_id in enumerate(particle_ids):
        particle_df = sampled[sampled["particle_id"].astype(int) == particle_id].sort_values("sweep_idx")
        if particle_df.empty:
            continue
        x = particle_df[x_col].astype(float).to_numpy()
        y = particle_df[y_col].astype(float).to_numpy()
        sweeps = particle_df["sweep_idx"].astype(float).to_numpy()
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
    return particle_ids, sweep_min, sweep_span


def _draw_slot_overlay(
    ax: plt.Axes,
    sampled: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
) -> None:
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    slot_rows = sampled.dropna(subset=["slot_id"]).copy()
    if slot_rows.empty:
        return
    slot_rows["slot_id"] = slot_rows["slot_id"].astype(int)
    slot_ids = sorted(int(slot) for slot in slot_rows["slot_id"].unique())
    marker_map = {slot: marker_cycle[idx % len(marker_cycle)] for idx, slot in enumerate(slot_ids)}
    for slot_id in slot_ids:
        slot_df = slot_rows[slot_rows["slot_id"].astype(int) == slot_id]
        ax.scatter(
            slot_df[x_col].astype(float),
            slot_df[y_col].astype(float),
            s=20,
            marker=marker_map[slot_id],
            facecolors="none",
            edgecolors="#222222",
            linewidth=0.45,
            alpha=0.35,
            zorder=6,
        )
    mapping = ", ".join(f"{slot}:{marker}" for slot, marker in marker_map.items())
    ax.text(
        0.02,
        0.02,
        f"slot marker map: {mapping}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#4a4a4a",
    )


def _draw_sweep_context_points(ax: plt.Axes, sampled: pd.DataFrame, *, y_col: str) -> None:
    ax.scatter(
        sampled["sweep_idx"].astype(float),
        sampled[y_col].astype(float),
        s=14,
        c="#b8b8b8",
        alpha=0.16,
        edgecolors="none",
        zorder=1,
        label="all particle states",
    )


def _draw_cold_slot_progression(
    ax: plt.Axes,
    cold_df: pd.DataFrame,
    *,
    y_col: str,
    alpha_lo: float,
    alpha_hi: float,
    color_map: dict[str, tuple[float, float, float, float]],
) -> int:
    if cold_df.empty:
        raise ValueError("Cold-slot progression requires non-empty trajectory rows.")
    x_vals = cold_df["sweep_idx"].astype(float).to_numpy()
    y_vals = cold_df[y_col].astype(float).to_numpy()
    particle_ids = cold_df["particle_id"].astype(int).to_numpy()
    sweep_min = float(x_vals.min())
    sweep_span = float(x_vals.max() - sweep_min)
    handoff_count = 0
    point_colors: list[tuple[float, float, float, float]] = []
    for tf_name in cold_df.get("bottleneck_tf", pd.Series(index=cold_df.index, dtype=object)).astype(object):
        if tf_name is None or (isinstance(tf_name, float) and np.isnan(tf_name)):
            point_colors.append((0.84, 0.15, 0.16, 0.95))
            continue
        point_colors.append(color_map.get(str(tf_name), (0.84, 0.15, 0.16, 0.95)))
    if x_vals.size >= 2:
        for seg_idx in range(1, x_vals.size):
            if sweep_span <= 0:
                alpha = 0.65
            else:
                t = (float(x_vals[seg_idx]) - sweep_min) / sweep_span
                alpha = alpha_lo + (alpha_hi - alpha_lo) * t
            handoff = int(particle_ids[seg_idx - 1]) != int(particle_ids[seg_idx])
            handoff_count += int(handoff)
            seg_color = point_colors[seg_idx]
            ax.plot(
                x_vals[seg_idx - 1 : seg_idx + 1],
                y_vals[seg_idx - 1 : seg_idx + 1],
                color=(seg_color[0], seg_color[1], seg_color[2], alpha),
                linewidth=1.8,
                linestyle="--" if handoff else "-",
                zorder=6,
            )
    ax.scatter(x_vals, y_vals, s=26, c=point_colors, edgecolors="none", zorder=7)
    ax.plot([], [], color="#333333", linewidth=1.8, label="cold-slot progression")
    ax.plot([], [], color="#333333", linewidth=1.4, linestyle="--", label="lineage handoff (slot swap)")
    for tf_name in sorted(color_map):
        ax.scatter([], [], s=28, c=[color_map[tf_name]], label=f"bottleneck TF: {tf_name}")
    return handoff_count


def plot_opt_trajectory(
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
        raise ValueError("Particle alpha bounds must be numeric.")
    alpha_lo = float(alpha_min)
    alpha_hi = float(alpha_max)
    if alpha_lo < 0 or alpha_hi > 1 or alpha_lo > alpha_hi:
        raise ValueError("Particle alpha bounds must satisfy 0 <= min <= max <= 1.")
    if len(tf_pair) != 2:
        raise ValueError("tf_pair must contain exactly two TF names.")

    x_col, y_col, normalized_scale = _resolve_scatter_columns(tf_pair=tf_pair, scatter_scale=scatter_scale)
    plot_df = _prepare_lineage_df(trajectory_df)
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

    particle_ids, _, _ = _draw_particle_lineage(
        ax,
        sampled,
        x_col=x_col,
        y_col=y_col,
        alpha_lo=alpha_lo,
        alpha_hi=alpha_hi,
    )
    max_particle = max(particle_ids)
    ax.plot([], [], color="#333333", linewidth=1.2, label=f"particle lineage (id=0..{max_particle})")

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
        _draw_slot_overlay(ax, sampled, x_col=x_col, y_col=y_col)

    scale_label = "raw LLR" if normalized_scale == "llr" else "normalized LLR"
    ax.set_xlabel(f"{tf_pair[0]} ({scale_label})")
    ax.set_ylabel(f"{tf_pair[1]} ({scale_label})")
    ax.set_title(f"Optimization trajectory ({tf_pair[0]} vs {tf_pair[1]}; particle lineage)")
    legend_labels = _legend_labels(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
    return {
        "mode": "particle_scatter",
        "legend_labels": legend_labels,
        "particle_count": len(particle_ids),
        "x_column": x_col,
        "y_column": y_col,
    }


def plot_opt_trajectory_sweep(
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
        raise ValueError("Particle alpha bounds must be numeric.")
    alpha_lo = float(alpha_min)
    alpha_hi = float(alpha_max)
    if alpha_lo < 0 or alpha_hi > 1 or alpha_lo > alpha_hi:
        raise ValueError("Particle alpha bounds must satisfy 0 <= min <= max <= 1.")

    plot_df = _prepare_lineage_df(trajectory_df)
    plot_df[y_column] = _require_numeric(plot_df, y_column, context="Trajectory")
    sampled = _sample_rows(plot_df, stride=max(1, int(stride)))
    cold_df = _extract_cold_slot_path(plot_df)
    cold_df = _sample_rows_by_sweep_stride(cold_df, stride=max(1, int(stride)))
    cold_df[y_column] = _require_numeric(cold_df, y_column, context="Cold-slot trajectory")
    cold_df, color_map = _infer_bottleneck_tf(cold_df, y_column=y_column)

    fig, ax = plt.subplots(figsize=(6.8, 5.1))
    _draw_sweep_context_points(ax, sampled, y_col=y_column)
    handoff_count = _draw_cold_slot_progression(
        ax,
        cold_df,
        y_col=y_column,
        alpha_lo=alpha_lo,
        alpha_hi=alpha_hi,
        color_map=color_map,
    )
    if slot_overlay:
        _draw_slot_overlay(ax, sampled, x_col="sweep_idx", y_col=y_column)

    ylabel = {
        "raw_llr_objective": "Raw LLR objective",
        "norm_llr_objective": "Normalized LLR objective",
        "objective_scalar": "Optimization score",
    }.get(y_column, y_column)
    ax.set_xlabel("Sweep index")
    ax.set_ylabel(ylabel)
    ax.set_title("Optimization trajectory (cold-slot progress over sweeps)")
    ax.text(
        0.02,
        0.98,
        "y = combined objective across TFs per sweep; bottleneck TF may change",
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
        "mode": "cold_slot_sweep",
        "legend_labels": legend_labels,
        "cold_point_count": int(len(cold_df)),
        "cold_handoff_count": int(handoff_count),
        "particle_count": int(plot_df["particle_id"].nunique()),
        "y_column": y_column,
    }
