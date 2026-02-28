"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/trajectory_score_space_plot.py

Orchestrate trajectory score-space plotting modes and metadata assembly.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.plots._savefig import savefig
from dnadesign.cruncher.analysis.plots.trajectory_common import _prepare_chain_df
from dnadesign.cruncher.analysis.plots.trajectory_score_space_panel import (
    _display_tf_name,
    _render_score_space_panel,
    _resolve_grid_pairs,
    _resolve_score_scale,
    _resolve_tf_list,
)


def _resolve_scatter_columns(*, tf_pair: tuple[str, str], scatter_scale: str) -> tuple[str, str, str]:
    scale = str(scatter_scale).strip().lower()
    if scale in {"llr", "raw-llr", "raw_llr"}:
        return f"raw_llr_{tf_pair[0]}", f"raw_llr_{tf_pair[1]}", "llr"
    if scale in {"normalized-llr", "norm-llr", "norm_llr"}:
        return f"norm_llr_{tf_pair[0]}", f"norm_llr_{tf_pair[1]}", "normalized-llr"
    raise ValueError("scatter_scale must be 'llr' or 'normalized-llr'.")


def _build_score_space_metadata(
    *,
    score_space_mode: str,
    panel_count: int,
    legend_labels: list[str],
    x_column: str,
    y_column: str,
    retain_elites: bool,
    first_stats: dict[str, int],
    legend_fontsize: int,
    anchor_annotation_fontsize: int,
    objective_caption: str | None,
    grid_shared_axes: bool | None = None,
    grid_label_mode: str | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "mode": "elite_score_space_context",
        "score_space_mode": score_space_mode,
        "panel_count": int(panel_count),
        "legend_labels": legend_labels,
        "chain_count": 0,
        "x_column": x_column,
        "y_column": y_column,
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
    }
    if grid_shared_axes is not None:
        metadata["grid_shared_axes"] = bool(grid_shared_axes)
    if grid_label_mode is not None:
        metadata["grid_label_mode"] = str(grid_label_mode)
    return metadata


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
    score_prefix, scale_label = _resolve_score_scale(scatter_scale)
    scale = str(scatter_scale).strip().lower()
    tf_list = _resolve_tf_list(plot_df=plot_df, tf_names=tf_names, score_prefix=score_prefix)

    legend_fontsize = 12
    anchor_annotation_fontsize = 11

    if mode == "all_pairs_grid":
        pairs = _resolve_grid_pairs(tf_list=tf_list, tf_pairs_grid=tf_pairs_grid)
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
            stats, labels = _render_score_space_panel(
                ax=ax,
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
                retain_elites=bool(retain_elites),
                legend_fontsize=legend_fontsize,
                anchor_annotation_fontsize=anchor_annotation_fontsize,
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
        return _build_score_space_metadata(
            score_space_mode=mode,
            panel_count=n_panels,
            legend_labels=legend_labels,
            x_column="multiple",
            y_column="multiple",
            retain_elites=bool(retain_elites),
            first_stats=first_stats,
            legend_fontsize=legend_fontsize,
            anchor_annotation_fontsize=anchor_annotation_fontsize,
            objective_caption=objective_caption,
            grid_shared_axes=grid_shared_axes,
            grid_label_mode=grid_label_mode,
        )

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
    elite_stats, legend_labels = _render_score_space_panel(
        ax=ax,
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
        show_x_label=True,
        show_y_label=True,
        retain_elites=bool(retain_elites),
        legend_fontsize=legend_fontsize,
        anchor_annotation_fontsize=anchor_annotation_fontsize,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level, bbox_inches=None)
    plt.close(fig)
    return _build_score_space_metadata(
        score_space_mode=mode,
        panel_count=1,
        legend_labels=legend_labels,
        x_column=x_col,
        y_column=y_col,
        retain_elites=bool(retain_elites),
        first_stats=elite_stats,
        legend_fontsize=legend_fontsize,
        anchor_annotation_fontsize=anchor_annotation_fontsize,
        objective_caption=objective_caption,
    )


__all__ = [
    "_resolve_scatter_columns",
    "_build_score_space_metadata",
    "plot_elite_score_space_context",
]
