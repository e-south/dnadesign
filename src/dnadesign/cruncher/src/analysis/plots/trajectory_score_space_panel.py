"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/trajectory_score_space_panel.py

Shared panel rendering helpers for trajectory score-space plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from dnadesign.cruncher.analysis.plots._style import apply_axes_style
from dnadesign.cruncher.analysis.plots.trajectory_common import _require_numeric
from dnadesign.cruncher.analysis.plots.trajectory_score_space import (
    _overlay_selected_elites,
    _prepare_elite_points,
)


def _clean_anchor_label(label: object) -> str:
    text = str(label or "consensus anchor").strip()
    return text


def _display_tf_name(tf_name: str) -> str:
    tf_text = str(tf_name).strip()
    if not tf_text:
        return tf_text
    return tf_text[0].upper() + tf_text[1:]


def _resolve_score_scale(scatter_scale: str) -> tuple[str, str]:
    scale = str(scatter_scale).strip().lower()
    if scale in {"llr", "raw-llr", "raw_llr"}:
        return "raw_llr_", "raw LLR"
    if scale in {"normalized-llr", "norm-llr", "norm_llr"}:
        return "norm_llr_", "normalized LLR"
    raise ValueError("scatter_scale must be 'llr' or 'normalized-llr'.")


def _resolve_tf_list(*, plot_df: pd.DataFrame, tf_names: list[str] | None, score_prefix: str) -> list[str]:
    tf_list = [str(tf).strip() for tf in (tf_names or []) if str(tf).strip()]
    if tf_list:
        return tf_list
    return sorted([col.removeprefix(score_prefix) for col in plot_df.columns if col.startswith(score_prefix)])


def _resolve_grid_pairs(
    *,
    tf_list: list[str],
    tf_pairs_grid: list[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    pairs = list(tf_pairs_grid or [])
    if pairs:
        return pairs
    if len(tf_list) < 2:
        raise ValueError("all_pairs_grid score-space mode requires at least two TFs.")
    pairs = []
    for idx, tf_name in enumerate(tf_list):
        for tf_other in tf_list[idx + 1 :]:
            pairs.append((tf_name, tf_other))
    if not pairs:
        raise ValueError("all_pairs_grid score-space mode requires at least one TF pair.")
    return pairs


def _render_score_space_panel(
    *,
    ax: plt.Axes,
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
    show_x_label: bool,
    show_y_label: bool,
    retain_elites: bool,
    legend_fontsize: int,
    anchor_annotation_fontsize: int,
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
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=legend_fontsize, loc="lower right")
    return elite_stats, labels


__all__ = [
    "_clean_anchor_label",
    "_display_tf_name",
    "_resolve_score_scale",
    "_resolve_tf_list",
    "_resolve_grid_pairs",
    "_render_score_space_panel",
]
