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
from matplotlib.colors import to_rgba

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
from dnadesign.cruncher.analysis.plots.trajectory_sweep import (
    plot_chain_trajectory_sweep as _plot_chain_trajectory_sweep,
)


def _resolve_scatter_columns(*, tf_pair: tuple[str, str], scatter_scale: str) -> tuple[str, str, str]:
    scale = str(scatter_scale).strip().lower()
    if scale in {"llr", "raw-llr", "raw_llr"}:
        return f"raw_llr_{tf_pair[0]}", f"raw_llr_{tf_pair[1]}", "llr"
    if scale in {"normalized-llr", "norm-llr", "norm_llr"}:
        return f"norm_llr_{tf_pair[0]}", f"norm_llr_{tf_pair[1]}", "normalized-llr"
    raise ValueError("scatter_scale must be 'llr' or 'normalized-llr'.")


def _clean_anchor_label(label: object) -> str:
    text = str(label or "consensus anchor").strip()
    return text


def _display_tf_name(tf_name: str) -> str:
    tf_text = str(tf_name).strip()
    if not tf_text:
        return tf_text
    return tf_text[0].upper() + tf_text[1:]


def _legend_labels(ax: plt.Axes, *, location: str = "best", fontsize: int = 10) -> list[str]:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=fontsize, loc=location)
    return labels


def _draw_chain_overlay(ax: plt.Axes, sampled: pd.DataFrame, *, x_col: str, y_col: str) -> None:
    chain_ids = sorted(int(chain) for chain in sampled["chain"].astype(int).unique())
    marker_map = {chain: _CHAIN_MARKERS[idx % len(_CHAIN_MARKERS)] for idx, chain in enumerate(chain_ids)}
    for chain_id in chain_ids:
        chain_df = sampled[sampled["chain"].astype(int) == chain_id]
        ax.scatter(
            chain_df[x_col].astype(float),
            chain_df[y_col].astype(float),
            s=34,
            marker=marker_map[chain_id],
            facecolors="none",
            edgecolors="#222222",
            linewidth=0.65,
            alpha=0.45,
            zorder=6,
        )


def _resolve_scatter_objective_column(plot_df: pd.DataFrame, objective_column: str) -> str:
    resolved = str(objective_column).strip()
    if not resolved:
        raise ValueError("Trajectory scatter objective column must be a non-empty string.")
    _require_numeric(plot_df, resolved, context="Trajectory")
    return resolved


def _resolve_elite_match_column(trajectory_df: pd.DataFrame, elites_df: pd.DataFrame | None) -> str | None:
    if elites_df is None or elites_df.empty:
        return None
    best_column: str | None = None
    best_overlap = 0
    for column in ("sequence_hash", "sequence"):
        if column in trajectory_df.columns and column in elites_df.columns:
            traj_values = {value for value in trajectory_df[column].astype(str).str.strip().tolist() if value}
            elite_values = {value for value in elites_df[column].astype(str).str.strip().tolist() if value}
            overlap = len(traj_values & elite_values)
            if overlap > best_overlap:
                best_overlap = overlap
                best_column = column
    return best_column


def _prepare_elite_points(
    elites_df: pd.DataFrame | None,
    *,
    trajectory_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    retain_elites: bool,
) -> tuple[pd.DataFrame, dict[int, set[int]], dict[str, int]]:
    stats = {
        "total": 0,
        "rendered_points": 0,
        "unique_coordinates": 0,
        "coordinate_collisions": 0,
        "exact_mapped": 0,
    }
    retain_sweeps_by_chain: dict[int, set[int]] = {}
    if elites_df is None or elites_df.empty:
        return pd.DataFrame(), retain_sweeps_by_chain, stats
    if x_col not in elites_df.columns or y_col not in elites_df.columns:
        return pd.DataFrame(), retain_sweeps_by_chain, stats

    elite_rows = elites_df.copy()
    elite_rows["_x"] = pd.to_numeric(elite_rows[x_col], errors="coerce")
    elite_rows["_y"] = pd.to_numeric(elite_rows[y_col], errors="coerce")
    elite_rows = elite_rows[elite_rows["_x"].notna() & elite_rows["_y"].notna()].copy()
    if elite_rows.empty:
        return pd.DataFrame(), retain_sweeps_by_chain, stats
    elite_rows["_x"] = elite_rows["_x"].astype(float)
    elite_rows["_y"] = elite_rows["_y"].astype(float)
    elite_rows["_exact_mapped"] = False
    elite_rows["_mapped_chain"] = np.nan
    elite_rows["_mapped_sweep_idx"] = np.nan

    match_column = _resolve_elite_match_column(trajectory_df, elite_rows)
    if match_column is not None:
        trajectory_lookup = trajectory_df.copy()
        trajectory_lookup["_match_key"] = trajectory_lookup[match_column].astype(str).str.strip()
        trajectory_lookup = trajectory_lookup[trajectory_lookup["_match_key"].ne("")]
        if not trajectory_lookup.empty:
            trajectory_lookup = trajectory_lookup.sort_values(["sweep_idx"]).drop_duplicates("_match_key", keep="last")
            map_cols = ["_match_key", "chain", "sweep_idx", x_col]
            if y_col != x_col:
                map_cols.append(y_col)
            mapping = trajectory_lookup[map_cols].set_index("_match_key")
            elite_keys = elite_rows[match_column].astype(str).str.strip()
            for idx, key in elite_keys.items():
                if not key or key not in mapping.index:
                    continue
                match_row = mapping.loc[key]
                x_value = float(match_row[x_col])
                y_value = float(match_row[y_col]) if y_col != x_col else x_value
                elite_rows.at[idx, "_x"] = x_value
                elite_rows.at[idx, "_y"] = y_value
                elite_rows.at[idx, "_exact_mapped"] = True
                elite_rows.at[idx, "_mapped_chain"] = int(match_row["chain"])
                elite_rows.at[idx, "_mapped_sweep_idx"] = int(match_row["sweep_idx"])
                if retain_elites:
                    chain_id = int(match_row["chain"])
                    sweep_idx = int(match_row["sweep_idx"])
                    retain_sweeps_by_chain.setdefault(chain_id, set()).add(sweep_idx)

    grouped = elite_rows.groupby(["_x", "_y"], sort=False, dropna=False).size().reset_index(name="n_elites")
    stats["total"] = int(len(elite_rows))
    stats["exact_mapped"] = int(elite_rows["_exact_mapped"].astype(bool).sum())
    stats["unique_coordinates"] = int(len(grouped))
    stats["coordinate_collisions"] = int((grouped["n_elites"].astype(int) > 1).sum())
    return elite_rows.reset_index(drop=True), retain_sweeps_by_chain, stats


def _sample_scatter_backbone(
    plot_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    objective_column: str,
    stride: int,
    retain_sweeps_by_chain: dict[int, set[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampled_parts: list[pd.DataFrame] = []
    best_parts: list[pd.DataFrame] = []
    for chain_id, chain_df in plot_df.groupby("chain", sort=True, dropna=False):
        ordered = chain_df.sort_values("sweep_idx").reset_index(drop=True)
        if ordered.empty:
            continue
        scores = ordered[objective_column].astype(float).to_numpy()
        update_idx = _best_update_indices(scores)
        priority_idx: list[int] = [0, len(ordered) - 1]
        priority_idx.extend(update_idx)
        sweeps_to_retain = retain_sweeps_by_chain.get(int(chain_id), set())
        if sweeps_to_retain:
            for idx, sweep_value in enumerate(ordered["sweep_idx"].astype(int).tolist()):
                if int(sweep_value) in sweeps_to_retain:
                    priority_idx.append(int(idx))
        keep_idx = _stride_indices(len(ordered), stride=max(1, int(stride)), priority_indices=priority_idx)
        sampled_chain = ordered.iloc[keep_idx].copy()
        sampled_parts.append(sampled_chain)
        update_sweeps = set(int(ordered.iloc[i]["sweep_idx"]) for i in update_idx)
        best_parts.append(sampled_chain[sampled_chain["sweep_idx"].astype(int).isin(update_sweeps)].copy())
    if not sampled_parts:
        return pd.DataFrame(columns=plot_df.columns), pd.DataFrame(columns=plot_df.columns)
    sampled = pd.concat(sampled_parts).sort_values(["chain", "sweep_idx"]).reset_index(drop=True)
    best_updates = pd.concat(best_parts).sort_values(["chain", "sweep_idx"]).reset_index(drop=True)
    return sampled, best_updates


def _sample_scatter_best_progression(
    plot_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    objective_column: str,
    stride: int,
    retain_sweeps_by_chain: dict[int, set[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampled_parts: list[pd.DataFrame] = []
    best_parts: list[pd.DataFrame] = []
    for chain_id, chain_df in plot_df.groupby("chain", sort=True, dropna=False):
        ordered = chain_df.sort_values("sweep_idx").reset_index(drop=True)
        if ordered.empty:
            continue
        scores = ordered[objective_column].astype(float).to_numpy()
        update_idx = _best_update_indices(scores)
        priority_idx: list[int] = [0, len(ordered) - 1]
        priority_idx.extend(update_idx)
        sweeps_to_retain = retain_sweeps_by_chain.get(int(chain_id), set())
        sweep_values = ordered["sweep_idx"].astype(int).tolist()
        if sweeps_to_retain:
            for idx, sweep_value in enumerate(sweep_values):
                if int(sweep_value) in sweeps_to_retain:
                    priority_idx.append(int(idx))
        keep_time_idx = _stride_indices(len(ordered), stride=max(1, int(stride)), priority_indices=priority_idx)
        best_prefix_idx = np.zeros(len(ordered), dtype=int)
        running_best = float("-inf")
        running_best_idx = 0
        for idx, value in enumerate(scores):
            if float(value) > running_best:
                running_best = float(value)
                running_best_idx = int(idx)
            best_prefix_idx[idx] = int(running_best_idx)
        sampled_rows: list[dict[str, object]] = []
        for time_idx in keep_time_idx.tolist():
            best_idx = int(best_prefix_idx[int(time_idx)])
            time_row = ordered.iloc[int(time_idx)]
            best_row = ordered.iloc[best_idx]
            sweep_value = int(time_row["sweep_idx"])
            sampled_rows.append(
                {
                    "chain": int(time_row["chain"]),
                    "sweep_idx": sweep_value,
                    x_col: float(best_row[x_col]),
                    y_col: float(best_row[y_col]),
                    objective_column: float(best_row[objective_column]),
                }
            )
        sampled_chain = pd.DataFrame(sampled_rows).sort_values("sweep_idx").reset_index(drop=True)
        sampled_parts.append(sampled_chain)
        update_sweeps = set(int(ordered.iloc[i]["sweep_idx"]) for i in update_idx)
        best_parts.append(sampled_chain[sampled_chain["sweep_idx"].astype(int).isin(update_sweeps)].copy())
    if not sampled_parts:
        columns = ["chain", "sweep_idx", x_col, y_col, objective_column]
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)
    sampled = pd.concat(sampled_parts).sort_values(["chain", "sweep_idx"]).reset_index(drop=True)
    best_updates = pd.concat(best_parts).sort_values(["chain", "sweep_idx"]).reset_index(drop=True)
    return sampled, best_updates


def _draw_chain_backbone(
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
        base_rgba = to_rgba(_CHAIN_COLORS[idx % len(_CHAIN_COLORS)])
        if x.size >= 2:
            for seg_idx in range(1, x.size):
                if sweep_span <= 0:
                    alpha = 0.68
                else:
                    t = (float(sweeps[seg_idx]) - sweep_min) / sweep_span
                    alpha = alpha_lo + (alpha_hi - alpha_lo) * t
                ax.plot(
                    x[seg_idx - 1 : seg_idx + 1],
                    y[seg_idx - 1 : seg_idx + 1],
                    color=(base_rgba[0], base_rgba[1], base_rgba[2], alpha),
                    linewidth=2.0,
                    zorder=4,
                )
        marker = _CHAIN_MARKERS[idx % len(_CHAIN_MARKERS)]
        ax.scatter(
            [float(x[0])],
            [float(y[0])],
            s=36,
            marker=marker,
            c=[(base_rgba[0], base_rgba[1], base_rgba[2], 0.78)],
            edgecolors="none",
            zorder=5,
        )
        ax.scatter(
            [float(x[-1])],
            [float(y[-1])],
            s=40,
            marker=marker,
            c=[(base_rgba[0], base_rgba[1], base_rgba[2], 0.97)],
            edgecolors="none",
            zorder=5,
        )
        if label_each_chain:
            ax.plot(
                [],
                [],
                color=(base_rgba[0], base_rgba[1], base_rgba[2], 1.0),
                linewidth=1.8,
                marker=">",
                markersize=6.0,
                markerfacecolor=(base_rgba[0], base_rgba[1], base_rgba[2], 0.9),
                markeredgecolor="none",
                label=f"Chain {chain_id}",
            )
        elif idx == 0:
            ax.plot(
                [],
                [],
                color=(base_rgba[0], base_rgba[1], base_rgba[2], 1.0),
                linewidth=1.8,
                marker=">",
                markersize=6.0,
                markerfacecolor=(base_rgba[0], base_rgba[1], base_rgba[2], 0.9),
                markeredgecolor="none",
                label=f"Chains (n={len(chain_ids)})",
            )
    return chain_ids, points_by_chain


def _draw_best_update_highlights(
    ax: plt.Axes,
    best_updates: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
) -> dict[int, int]:
    points_by_chain: dict[int, int] = {}
    if best_updates.empty:
        return points_by_chain
    chain_ids = sorted(int(v) for v in best_updates["chain"].unique())
    for idx, chain_id in enumerate(chain_ids):
        chain_df = best_updates[best_updates["chain"].astype(int) == chain_id].sort_values("sweep_idx")
        points_by_chain[chain_id] = int(len(chain_df))
        if chain_df.empty:
            continue
        x_vals = chain_df[x_col].astype(float).to_numpy()
        y_vals = chain_df[y_col].astype(float).to_numpy()
        color = _CHAIN_COLORS[idx % len(_CHAIN_COLORS)]
        ax.scatter(
            x_vals,
            y_vals,
            s=34,
            marker="D",
            c=color,
            edgecolors="#111111",
            linewidths=0.65,
            alpha=0.9,
            zorder=7,
            label="Best-so-far updates" if idx == 0 else None,
        )
    return points_by_chain


def _offset_point_in_display_space(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    dx_points: float,
    dy_points: float,
) -> tuple[float, float]:
    points_to_pixels = float(ax.figure.dpi) / 72.0
    xy_pixels = ax.transData.transform((x, y))
    shifted_pixels = xy_pixels + np.asarray([dx_points * points_to_pixels, dy_points * points_to_pixels], dtype=float)
    shifted_data = ax.transData.inverted().transform(shifted_pixels)
    return float(shifted_data[0]), float(shifted_data[1])


def _nearest_point(points: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    if points.size == 0:
        return None
    deltas = points - np.asarray([x, y], dtype=float)
    idx = int(np.argmin(np.sum(deltas * deltas, axis=1)))
    nearest = points[idx]
    return float(nearest[0]), float(nearest[1])


def _build_chain_sweep_path_lookup(
    points_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
) -> dict[tuple[int, int], tuple[float, float]]:
    if points_df.empty:
        return {}
    required_cols = {"chain", "sweep_idx", x_col, y_col}
    missing = [column for column in required_cols if column not in points_df.columns]
    if missing:
        raise ValueError(f"Trajectory points missing required columns for path lookup: {missing}")
    path_points = points_df[["chain", "sweep_idx"]].copy()
    path_points["_path_x"] = pd.to_numeric(points_df[x_col], errors="coerce")
    path_points["_path_y"] = pd.to_numeric(points_df[y_col], errors="coerce")
    path_points["chain"] = pd.to_numeric(path_points["chain"], errors="coerce").astype("Int64")
    path_points["sweep_idx"] = pd.to_numeric(path_points["sweep_idx"], errors="coerce").astype("Int64")
    path_points = path_points.dropna(subset=["chain", "sweep_idx", "_path_x", "_path_y"])
    path_points = path_points.drop_duplicates(["chain", "sweep_idx"], keep="last")
    return {
        (int(row["chain"]), int(row["sweep_idx"])): (float(row["_path_x"]), float(row["_path_y"]))
        for _, row in path_points.iterrows()
    }


def _draw_elite_context_stubs(
    ax: plt.Axes,
    *,
    trajectory_df: pd.DataFrame,
    elite_rows: pd.DataFrame,
    x_col: str,
    y_col: str,
    context_radius: int,
    alpha_lo: float,
    alpha_hi: float,
    chain_label_limit: int = 8,
) -> tuple[list[int], dict[int, int], dict[int, list[int]], dict[str, object]]:
    if context_radius < 1:
        raise ValueError("trajectory scatter elite context radius must be >= 1.")
    if elite_rows.empty:
        return [], {}, {}, {"center_count": 0, "segment_count": 0}

    mapped = elite_rows[["_mapped_chain", "_mapped_sweep_idx"]].copy()
    mapped = mapped.dropna(subset=["_mapped_chain", "_mapped_sweep_idx"])
    if mapped.empty:
        return [], {}, {}, {"center_count": 0, "segment_count": 0}
    mapped["_mapped_chain"] = pd.to_numeric(mapped["_mapped_chain"], errors="coerce").astype("Int64")
    mapped["_mapped_sweep_idx"] = pd.to_numeric(mapped["_mapped_sweep_idx"], errors="coerce").astype("Int64")
    mapped = mapped.dropna(subset=["_mapped_chain", "_mapped_sweep_idx"]).drop_duplicates()
    if mapped.empty:
        return [], {}, {}, {"center_count": 0, "segment_count": 0}

    centers_by_chain: dict[int, list[int]] = {}
    for _, row in mapped.iterrows():
        chain_id = int(row["_mapped_chain"])
        sweep_idx = int(row["_mapped_sweep_idx"])
        centers_by_chain.setdefault(chain_id, []).append(sweep_idx)
    for chain_id in centers_by_chain:
        centers_by_chain[chain_id] = sorted(set(centers_by_chain[chain_id]))

    chain_ids = sorted(centers_by_chain.keys())
    label_each_chain = len(chain_ids) <= int(chain_label_limit)
    points_by_chain: dict[int, int] = {}
    sweeps_by_chain: dict[int, list[int]] = {}
    segment_count = 0
    direction_marker_count = 0
    direction_marker_sweeps_by_chain: dict[int, list[int]] = {}

    for idx, chain_id in enumerate(chain_ids):
        chain_df = trajectory_df[trajectory_df["chain"].astype(int) == int(chain_id)].copy()
        chain_df = chain_df.sort_values("sweep_idx").reset_index(drop=True)
        if chain_df.empty:
            points_by_chain[int(chain_id)] = 0
            sweeps_by_chain[int(chain_id)] = []
            continue
        chain_sweeps = chain_df["sweep_idx"].astype(int).to_numpy()
        selected_idx: set[int] = set()
        for center_sweep in centers_by_chain[chain_id]:
            match_idx = np.where(chain_sweeps == int(center_sweep))[0]
            if match_idx.size == 0:
                continue
            center_idx = int(match_idx[-1])
            lo = max(0, center_idx - int(context_radius))
            hi = min(len(chain_df) - 1, center_idx + int(context_radius))
            selected_idx.update(range(lo, hi + 1))
        if not selected_idx:
            points_by_chain[int(chain_id)] = 0
            sweeps_by_chain[int(chain_id)] = []
            continue

        selected_sorted = sorted(selected_idx)
        local_df = chain_df.iloc[selected_sorted].copy()
        x_vals = local_df[x_col].astype(float).to_numpy()
        y_vals = local_df[y_col].astype(float).to_numpy()
        sweep_vals = local_df["sweep_idx"].astype(int).to_numpy()
        points_by_chain[int(chain_id)] = int(len(local_df))
        sweeps_by_chain[int(chain_id)] = [int(v) for v in sweep_vals.tolist()]
        if len(local_df) < 2:
            continue
        color = to_rgba(_CHAIN_COLORS[idx % len(_CHAIN_COLORS)])
        centers = np.asarray(centers_by_chain[chain_id], dtype=float)
        center_sweeps = set(centers_by_chain[chain_id])
        for seg_idx in range(1, len(local_df)):
            dist_prev = float(np.min(np.abs(centers - float(sweep_vals[seg_idx - 1]))))
            dist_curr = float(np.min(np.abs(centers - float(sweep_vals[seg_idx]))))
            dist = min(dist_prev, dist_curr)
            closeness = max(0.0, 1.0 - (dist / float(context_radius + 1)))
            alpha = alpha_lo + (alpha_hi - alpha_lo) * closeness
            ax.plot(
                x_vals[seg_idx - 1 : seg_idx + 1],
                y_vals[seg_idx - 1 : seg_idx + 1],
                color=(color[0], color[1], color[2], alpha),
                linewidth=1.25,
                zorder=4,
            )
            segment_count += 1
        marker_sweeps: list[int] = []
        marker_alpha = min(0.46, max(0.24, alpha_lo * 1.25))
        for joint_idx in range(1, len(local_df) - 1):
            sweep_curr = int(sweep_vals[joint_idx])
            if sweep_curr in center_sweeps:
                continue
            dx = float(x_vals[joint_idx + 1] - x_vals[joint_idx])
            dy = float(y_vals[joint_idx + 1] - y_vals[joint_idx])
            if np.isclose(dx, 0.0, atol=1e-12) and np.isclose(dy, 0.0, atol=1e-12):
                continue
            angle = float(np.degrees(np.arctan2(dy, dx)) - 90.0)
            ax.scatter(
                [float(x_vals[joint_idx])],
                [float(y_vals[joint_idx])],
                s=24.0,
                marker=(3, 0, angle),
                c=[(color[0], color[1], color[2], marker_alpha)],
                edgecolors="none",
                zorder=5,
            )
            marker_sweeps.append(sweep_curr)
            direction_marker_count += 1
        direction_marker_sweeps_by_chain[int(chain_id)] = marker_sweeps
        if label_each_chain:
            ax.plot(
                [],
                [],
                color=(color[0], color[1], color[2], 0.9),
                linewidth=1.25,
                marker=">",
                markersize=6.0,
                markerfacecolor=(color[0], color[1], color[2], 0.85),
                markeredgecolor="none",
                label=f"Chain {chain_id}",
            )
        elif idx == 0:
            ax.plot(
                [],
                [],
                color=(color[0], color[1], color[2], 0.9),
                linewidth=1.25,
                marker=">",
                markersize=6.0,
                markerfacecolor=(color[0], color[1], color[2], 0.85),
                markeredgecolor="none",
                label=f"Elite context stubs (chains={len(chain_ids)})",
            )
    return (
        chain_ids,
        points_by_chain,
        sweeps_by_chain,
        {
            "center_count": int(len(mapped)),
            "segment_count": int(segment_count),
            "direction_marker_count": int(direction_marker_count),
            "direction_marker_sweeps_by_chain": direction_marker_sweeps_by_chain,
        },
    )


def _overlay_selected_elites(
    ax: plt.Axes,
    elite_rows: pd.DataFrame,
    *,
    sampled_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    collision_strategy: str,
    link_mode: str,
    snap_exact_mapped_to_path: bool,
) -> dict[str, int]:
    stats = {
        "total": 0,
        "rendered_points": 0,
        "unique_coordinates": 0,
        "coordinate_collisions": 0,
        "collision_annotation_count": 0,
        "exact_mapped": 0,
        "path_link_count": 0,
        "snapped_to_path_count": 0,
    }
    if elite_rows.empty:
        return stats
    plotted = elite_rows.copy()
    plotted["_plot_x"] = plotted["_x"].astype(float)
    plotted["_plot_y"] = plotted["_y"].astype(float)
    plotted["_anchor_x"] = plotted["_x"].astype(float)
    plotted["_anchor_y"] = plotted["_y"].astype(float)
    path_lookup = _build_chain_sweep_path_lookup(sampled_df, x_col=x_col, y_col=y_col)
    if path_lookup:
        for idx, row in plotted.iterrows():
            mapped_chain = row.get("_mapped_chain")
            mapped_sweep = row.get("_mapped_sweep_idx")
            if pd.isna(mapped_chain) or pd.isna(mapped_sweep):
                continue
            anchor = path_lookup.get((int(mapped_chain), int(mapped_sweep)))
            if anchor is None:
                continue
            anchor_x, anchor_y = float(anchor[0]), float(anchor[1])
            plotted.at[idx, "_anchor_x"] = anchor_x
            plotted.at[idx, "_anchor_y"] = anchor_y
            elite_x = float(row["_x"])
            elite_y = float(row["_y"])
            if snap_exact_mapped_to_path:
                if not (np.isclose(anchor_x, elite_x) and np.isclose(anchor_y, elite_y)):
                    stats["snapped_to_path_count"] += 1
                plotted.at[idx, "_plot_x"] = anchor_x
                plotted.at[idx, "_plot_y"] = anchor_y
                continue
    if link_mode == "nearest":
        backbone_points = (
            np.column_stack(
                [
                    sampled_df[x_col].to_numpy(dtype=float),
                    sampled_df[y_col].to_numpy(dtype=float),
                ]
            )
            if not sampled_df.empty
            else np.asarray([], dtype=float)
        )
        for idx, row in plotted.iterrows():
            if bool(row.get("_exact_mapped", False)):
                continue
            nearest = _nearest_point(backbone_points, float(row["_x"]), float(row["_y"]))
            if nearest is None:
                continue
            plotted.at[idx, "_anchor_x"] = float(nearest[0])
            plotted.at[idx, "_anchor_y"] = float(nearest[1])
            ax.plot(
                [float(row["_x"]), float(nearest[0])],
                [float(row["_y"]), float(nearest[1])],
                color="#1b9e77",
                linestyle="--",
                linewidth=0.7,
                alpha=0.35,
                zorder=8,
            )
            stats["path_link_count"] += 1

    grouped = plotted.groupby(["_plot_x", "_plot_y"], sort=False, dropna=False).size().reset_index(name="n_elites")
    render_x: list[float] = []
    render_y: list[float] = []
    if collision_strategy == "none_or_count":
        render_x = grouped["_plot_x"].astype(float).tolist()
        render_y = grouped["_plot_y"].astype(float).tolist()
    elif collision_strategy == "offset_points_with_connector":
        for _, row in grouped.iterrows():
            center_x = float(row["_plot_x"])
            center_y = float(row["_plot_y"])
            count = int(row["n_elites"])
            if count <= 1:
                render_x.append(center_x)
                render_y.append(center_y)
                continue
            angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
            radius_points = 5.0
            for angle in angles:
                dx_points = radius_points * float(np.cos(angle))
                dy_points = radius_points * float(np.sin(angle))
                off_x, off_y = _offset_point_in_display_space(
                    ax,
                    center_x,
                    center_y,
                    dx_points=dx_points,
                    dy_points=dy_points,
                )
                ax.plot([center_x, off_x], [center_y, off_y], color="#1b9e77", linewidth=0.7, alpha=0.35, zorder=8)
                render_x.append(off_x)
                render_y.append(off_y)
    else:
        raise ValueError(
            "trajectory scatter elite collision strategy must be one of: none_or_count, offset_points_with_connector"
        )

    ax.scatter(
        render_x,
        render_y,
        s=56.0,
        marker="o",
        facecolors="#ffffff",
        edgecolors="#7a7a7a",
        linewidths=1.1,
        zorder=9,
        label=f"Selected elites (n={len(plotted)})",
    )
    stats["total"] = int(len(plotted))
    stats["rendered_points"] = int(len(render_x))
    stats["unique_coordinates"] = int(len(grouped))
    stats["coordinate_collisions"] = int((grouped["n_elites"].astype(int) > 1).sum())
    stats["exact_mapped"] = int(plotted["_exact_mapped"].astype(bool).sum())
    return stats


def plot_elite_score_space_context(
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
    retain_elites: bool = True,
    score_space_mode: str = "pair",
    tf_names: list[str] | None = None,
    tf_pairs_grid: list[tuple[str, str]] | None = None,
    consensus_anchors_by_pair: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    mode = str(score_space_mode).strip().lower()
    if mode not in {"pair", "worst_vs_second_worst", "all_pairs_grid"}:
        raise ValueError("score_space_mode must be one of: pair, worst_vs_second_worst, all_pairs_grid.")

    plot_df = _prepare_chain_df(trajectory_df)
    if baseline_df is None:
        baseline = pd.DataFrame()
    elif not isinstance(baseline_df, pd.DataFrame):
        raise ValueError("Baseline must be provided as a pandas DataFrame when present.")
    else:
        baseline = baseline_df.copy()
    scale = str(scatter_scale).strip().lower()
    if scale in {"llr", "raw-llr", "raw_llr"}:
        score_prefix = "raw_llr_"
        scale_label = "raw LLR"
    elif scale in {"normalized-llr", "norm-llr", "norm_llr"}:
        score_prefix = "norm_llr_"
        scale_label = "normalized LLR"
    else:
        raise ValueError("scatter_scale must be 'llr' or 'normalized-llr'.")
    tf_list = [str(tf).strip() for tf in (tf_names or []) if str(tf).strip()]
    if not tf_list:
        tf_list = sorted([col.removeprefix(score_prefix) for col in plot_df.columns if col.startswith(score_prefix)])

    legend_fontsize = 12
    anchor_annotation_fontsize = 11

    def _draw_panel(
        ax: plt.Axes,
        *,
        panel_traj_df: pd.DataFrame,
        panel_baseline_df: pd.DataFrame,
        panel_elites_df: pd.DataFrame | None,
        x_col: str,
        y_col: str,
        x_label: str,
        y_label: str,
        title: str,
        panel_anchors: list[dict[str, object]] | None,
        panel_y_tf: str | None,
        show_legend: bool,
        show_x_label: bool = True,
        show_y_label: bool = True,
    ) -> tuple[dict[str, int], list[str]]:
        panel_traj_df[x_col] = _require_numeric(panel_traj_df, x_col, context="Trajectory")
        panel_traj_df[y_col] = _require_numeric(panel_traj_df, y_col, context="Trajectory")
        if not panel_baseline_df.empty:
            panel_baseline_df[x_col] = _require_numeric(panel_baseline_df, x_col, context="Baseline")
            panel_baseline_df[y_col] = _require_numeric(panel_baseline_df, y_col, context="Baseline")
        elite_rows, _, _ = _prepare_elite_points(
            panel_elites_df,
            trajectory_df=panel_traj_df,
            x_col=x_col,
            y_col=y_col,
            retain_elites=bool(retain_elites),
        )
        if not panel_baseline_df.empty:
            ax.scatter(
                panel_baseline_df[x_col].astype(float),
                panel_baseline_df[y_col].astype(float),
                s=22,
                c="#bdbdbd",
                alpha=0.28,
                edgecolors="none",
                zorder=3,
                label="Random baseline",
            )
        elite_stats = _overlay_selected_elites(
            ax,
            elite_rows,
            sampled_df=pd.DataFrame(columns=panel_traj_df.columns),
            x_col=x_col,
            y_col=y_col,
            collision_strategy="none_or_count",
            link_mode="exact_only",
            snap_exact_mapped_to_path=False,
        )
        if panel_anchors:
            anchor_x = [float(item["x"]) for item in panel_anchors]
            anchor_y = [float(item["y"]) for item in panel_anchors]
            ax.scatter(
                anchor_x,
                anchor_y,
                s=56,
                marker="o",
                facecolor="#d95f5f",
                edgecolor="#8f1f1f",
                linewidth=1.0,
                zorder=8,
                label="TF consensus anchors",
            )
            for item in panel_anchors:
                x = float(item["x"])
                y = float(item["y"])
                label = _clean_anchor_label(item.get("label") or item.get("tf") or "consensus anchor")
                tf_name = str(item.get("tf") or "").strip().lower()
                y_tf = str(panel_y_tf or "").strip().lower()
                if tf_name == y_tf:
                    xytext = (-8, -8)
                    ha = "right"
                else:
                    xytext = (8, -8)
                    ha = "left"
                ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=xytext,
                    textcoords="offset points",
                    fontsize=anchor_annotation_fontsize,
                    color="#3f3f3f",
                    ha=ha,
                )
        ax.set_xlabel(x_label if show_x_label else "")
        ax.set_ylabel(y_label if show_y_label else "")
        ax.set_title(title)
        apply_axes_style(ax, ygrid=True, xgrid=True, tick_labelsize=12, title_size=14, label_size=14)
        for grid_line in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
            grid_line.set_zorder(0)
        labels: list[str] = []
        if show_legend:
            labels = _legend_labels(ax, location="lower right", fontsize=legend_fontsize)
        return elite_stats, labels

    if mode == "all_pairs_grid":
        pairs = list(tf_pairs_grid or [])
        if not pairs:
            if len(tf_list) < 2:
                raise ValueError("all_pairs_grid score-space mode requires at least two TFs.")
            pairs = []
            for idx, tf_name in enumerate(tf_list):
                for tf_other in tf_list[idx + 1 :]:
                    pairs.append((tf_name, tf_other))
        if not pairs:
            raise ValueError("all_pairs_grid score-space mode requires at least one TF pair.")
        n_panels = len(pairs)
        ncols = min(3, n_panels)
        nrows = int(np.ceil(float(n_panels) / float(ncols)))
        grid_shared_axes = bool(scale == "normalized-llr")
        grid_label_mode = "edge_only"
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.6 * nrows), squeeze=False)
        all_axes = axes.flatten()
        legend_labels: list[str] = []
        panel_stats: dict[str, int] | None = None
        for idx, pair in enumerate(pairs):
            ax = all_axes[idx]
            row_idx = idx // ncols
            col_idx = idx % ncols
            pair_x_col, pair_y_col, _ = _resolve_scatter_columns(tf_pair=pair, scatter_scale=scatter_scale)
            panel_anchors = None
            if isinstance(consensus_anchors_by_pair, dict):
                pair_key = f"{pair[0]}|{pair[1]}"
                panel_anchors = consensus_anchors_by_pair.get(pair_key)
            stats, labels = _draw_panel(
                ax,
                panel_traj_df=plot_df.copy(),
                panel_baseline_df=baseline.copy(),
                panel_elites_df=elites_df.copy() if elites_df is not None else None,
                x_col=pair_x_col,
                y_col=pair_y_col,
                x_label=f"{_display_tf_name(pair[0])} best-window {scale_label}",
                y_label=f"{_display_tf_name(pair[1])} best-window {scale_label}",
                title=f"{_display_tf_name(pair[0])} vs {_display_tf_name(pair[1])}",
                panel_anchors=panel_anchors,
                panel_y_tf=str(pair[1]),
                show_legend=(idx == 0),
                show_x_label=(row_idx == nrows - 1),
                show_y_label=(col_idx == 0),
            )
            if idx == 0:
                legend_labels = labels
                panel_stats = stats
        for ax in all_axes[n_panels:]:
            ax.set_visible(False)
        if grid_shared_axes:
            drawn_axes = [all_axes[idx] for idx in range(n_panels)]
            x_limits = [ax.get_xlim() for ax in drawn_axes]
            y_limits = [ax.get_ylim() for ax in drawn_axes]
            x_min = min(limit[0] for limit in x_limits)
            x_max = max(limit[1] for limit in x_limits)
            y_min = min(limit[0] for limit in y_limits)
            y_max = max(limit[1] for limit in y_limits)
            for ax in drawn_axes:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
        fig.suptitle("Selected elites in TF score space", fontsize=14)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level, bbox_inches=None)
        plt.close(fig)
        first_stats = panel_stats or {
            "total": 0,
            "rendered_points": 0,
            "unique_coordinates": 0,
            "coordinate_collisions": 0,
            "collision_annotation_count": 0,
            "exact_mapped": 0,
            "path_link_count": 0,
            "snapped_to_path_count": 0,
        }
        return {
            "mode": "elite_score_space_context",
            "score_space_mode": mode,
            "panel_count": int(n_panels),
            "legend_labels": legend_labels,
            "chain_count": 0,
            "x_column": "multiple",
            "y_column": "multiple",
            "plotted_points_by_chain": {},
            "plotted_sweep_indices_by_chain": {},
            "best_update_points_by_chain": {},
            "retain_elites": bool(retain_elites),
            "elite_collision_strategy": "none_or_count",
            "elite_link_mode": "exact_only",
            "elite_points_plotted": int(first_stats["total"]),
            "elite_rendered_points": int(first_stats["rendered_points"]),
            "elite_unique_coordinates": int(first_stats["unique_coordinates"]),
            "elite_coordinate_collisions": int(first_stats["coordinate_collisions"]),
            "elite_collision_annotation_count": int(first_stats["collision_annotation_count"]),
            "elite_exact_mapped_points": int(first_stats["exact_mapped"]),
            "elite_path_link_count": int(first_stats["path_link_count"]),
            "elite_snapped_to_path_count": int(first_stats["snapped_to_path_count"]),
            "legend_fontsize": int(legend_fontsize),
            "anchor_annotation_fontsize": int(anchor_annotation_fontsize),
            "objective_caption": str(objective_caption or ""),
            "grid_shared_axes": bool(grid_shared_axes),
            "grid_label_mode": grid_label_mode,
        }

    if mode == "worst_vs_second_worst":
        if len(tf_pair) != 2:
            raise ValueError("tf_pair must contain exactly two TF names.")
        if scale != "llr":
            raise ValueError("worst_vs_second_worst score-space mode requires scatter_scale='llr'.")
        projected_plot_df = plot_df
        projected_baseline = baseline
        projected_elites = elites_df.copy() if elites_df is not None else None
        x_col = f"raw_llr_{tf_pair[0]}"
        y_col = f"raw_llr_{tf_pair[1]}"
        x_label = f"Worst TF ({_display_tf_name(tf_pair[0])}) best-window raw LLR"
        y_label = f"Second-worst TF ({_display_tf_name(tf_pair[1])}) best-window raw LLR"
    else:
        if len(tf_pair) != 2:
            raise ValueError("tf_pair must contain exactly two TF names.")
        x_col, y_col, _ = _resolve_scatter_columns(tf_pair=tf_pair, scatter_scale=scatter_scale)
        projected_plot_df = plot_df
        projected_baseline = baseline
        projected_elites = elites_df.copy() if elites_df is not None else None
        x_label = f"{_display_tf_name(tf_pair[0])} best-window {scale_label}"
        y_label = f"{_display_tf_name(tf_pair[1])} best-window {scale_label}"

    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    elite_stats, legend_labels = _draw_panel(
        ax,
        panel_traj_df=projected_plot_df,
        panel_baseline_df=projected_baseline,
        panel_elites_df=projected_elites,
        x_col=x_col,
        y_col=y_col,
        x_label=x_label,
        y_label=y_label,
        title="Selected elites in TF score space",
        panel_anchors=consensus_anchors,
        panel_y_tf=str(tf_pair[1]) if mode == "pair" else None,
        show_legend=True,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level, bbox_inches=None)
    plt.close(fig)
    return {
        "mode": "elite_score_space_context",
        "score_space_mode": mode,
        "panel_count": 1,
        "legend_labels": legend_labels,
        "chain_count": 0,
        "x_column": x_col,
        "y_column": y_col,
        "plotted_points_by_chain": {},
        "plotted_sweep_indices_by_chain": {},
        "best_update_points_by_chain": {},
        "retain_elites": bool(retain_elites),
        "elite_collision_strategy": "none_or_count",
        "elite_link_mode": "exact_only",
        "elite_points_plotted": int(elite_stats["total"]),
        "elite_rendered_points": int(elite_stats["rendered_points"]),
        "elite_unique_coordinates": int(elite_stats["unique_coordinates"]),
        "elite_coordinate_collisions": int(elite_stats["coordinate_collisions"]),
        "elite_collision_annotation_count": int(elite_stats["collision_annotation_count"]),
        "elite_exact_mapped_points": int(elite_stats["exact_mapped"]),
        "elite_path_link_count": int(elite_stats["path_link_count"]),
        "elite_snapped_to_path_count": int(elite_stats["snapped_to_path_count"]),
        "legend_fontsize": int(legend_fontsize),
        "anchor_annotation_fontsize": int(anchor_annotation_fontsize),
        "objective_caption": str(objective_caption or ""),
    }


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
    return _plot_chain_trajectory_sweep(
        trajectory_df=trajectory_df,
        y_column=y_column,
        y_mode=y_mode,
        objective_config=objective_config,
        cooling_config=cooling_config,
        tune_sweeps=tune_sweeps,
        objective_caption=objective_caption,
        out_path=out_path,
        dpi=dpi,
        png_compress_level=png_compress_level,
        stride=stride,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        chain_overlay=chain_overlay,
        summary_overlay=summary_overlay,
    )
