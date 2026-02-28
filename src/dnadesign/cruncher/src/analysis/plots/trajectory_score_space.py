"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/trajectory_score_space.py

Shared helper logic for trajectory score-space elite mapping and sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.plots.trajectory_common import (
    _best_update_indices,
    _stride_indices,
)


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


__all__ = [
    "_resolve_elite_match_column",
    "_prepare_elite_points",
    "_sample_scatter_backbone",
    "_sample_scatter_best_progression",
    "_build_chain_sweep_path_lookup",
    "_overlay_selected_elites",
]
