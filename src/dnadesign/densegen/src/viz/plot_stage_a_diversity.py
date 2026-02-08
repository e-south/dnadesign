"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_stage_a_diversity.py

Stage-A core diversity figure builder.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import SubplotSpec

from ..utils.plot_style import format_regulator_label, stage_a_rcparams
from .plot_common import _apply_style, _style
from .plot_stage_a_common import _stage_a_regulator_colors, _stage_a_text_sizes


def _build_stage_a_diversity_figure(
    *,
    input_name: str,
    pool_df: pd.DataFrame,
    sampling: dict,
    style: dict,
    fig: mpl.figure.Figure | None = None,
    slot: SubplotSpec | None = None,
    title: str | None = None,
    show_column_titles: bool = True,
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes], list[mpl.axes.Axes]]:
    style = _style(style)
    style["seaborn_style"] = False
    rc = stage_a_rcparams(style)
    text_sizes = _stage_a_text_sizes(style)
    if "eligible_score_hist" not in sampling:
        raise ValueError(f"Stage-A sampling missing eligible score histogram for input '{input_name}'.")
    eligible_hist = sampling["eligible_score_hist"]
    if not isinstance(eligible_hist, list) or not eligible_hist:
        raise ValueError(f"Stage-A sampling missing eligible score histogram for input '{input_name}'.")
    regulators: list[str] = []
    for row in eligible_hist:
        if not isinstance(row, dict):
            raise ValueError(f"Stage-A sampling has invalid eligible score entry for input '{input_name}'.")
        if "regulator" not in row:
            raise ValueError(f"Stage-A sampling missing regulator labels for input '{input_name}'.")
        regulators.append(str(row["regulator"]))
    if not regulators:
        raise ValueError(f"Stage-A sampling missing regulator labels for input '{input_name}'.")
    if "regulator_id" in pool_df.columns:
        tf_col = "regulator_id"
    elif "tf" in pool_df.columns:
        tf_col = "tf"
    else:
        raise ValueError(f"Stage-A pool missing regulator_id or tf column for input '{input_name}'.")
    if "best_hit_score" not in pool_df.columns:
        raise ValueError(f"Stage-A pool missing best_hit_score for input '{input_name}'.")
    reg_colors = _stage_a_regulator_colors(regulators, style)
    n_regs = max(1, len(regulators))
    fig_width = float(style.get("figsize", (11.6, 3.8))[0])
    fig_height = max(2.75, 0.9 * n_regs + 0.7)
    subtitle_size = text_sizes["panel_title"] * 0.88
    title_pad = 9
    with mpl.rc_context(rc):
        if fig is None:
            fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
        if slot is None:
            panel = fig.add_gridspec(nrows=1, ncols=1)[0, 0]
        else:
            panel = slot
        body = panel.subgridspec(
            nrows=n_regs,
            ncols=3,
            width_ratios=[0.22, 1.15, 1.0],
            hspace=0.25,
            wspace=0.28,
        )
        axes_left: list[mpl.axes.Axes] = []
        axes_right: list[mpl.axes.Axes] = []
        label_axes: list[mpl.axes.Axes] = []
        for idx in range(n_regs):
            label_axes.append(fig.add_subplot(body[idx, 0]))
            axes_left.append(fig.add_subplot(body[idx, 1]))
            axes_right.append(fig.add_subplot(body[idx, 2]))

        diversity_by_reg: dict[str, dict] = {}
        row_by_reg: dict[str, dict] = {}
        for row in eligible_hist:
            reg = str(row["regulator"])
            if "diversity" not in row:
                raise ValueError(f"Stage-A diversity missing for input '{input_name}' ({reg}).")
            diversity = row["diversity"]
            if not isinstance(diversity, dict):
                raise ValueError(f"Stage-A diversity missing for input '{input_name}' ({reg}).")
            diversity_by_reg[reg] = diversity
            row_by_reg[reg] = row

        def _dist_from_counts(bins: list[float] | list[int], counts: list[int]) -> tuple[np.ndarray, np.ndarray]:
            if not bins or not counts:
                raise ValueError("Stage-A diversity missing nearest-neighbor bins or counts.")
            arr = np.asarray(counts, dtype=float)
            total = float(arr.sum())
            x = np.asarray(bins, dtype=float)
            if total <= 0:
                return x, np.zeros_like(x, dtype=float)
            y = arr / total
            return x, y

        metric_label = "Pairwise Hamming NN"
        top_color = "#cfe8dc"
        diversified_color = "#7fbf9b"
        label_size = float(style.get("label_size", float(style.get("font_size", 13.0))))
        axis_label_size = label_size
        tick_size = float(style.get("tick_size", label_size))
        for idx, reg in enumerate(regulators):
            hue = reg_colors.get(reg, "#4c78a8")
            row = row_by_reg[reg]
            diversity = diversity_by_reg[reg]
            label = format_regulator_label(reg)
            if "tfbs_core" in pool_df.columns:
                core_vals = pool_df.loc[pool_df[tf_col].astype(str) == reg, "tfbs_core"].dropna()
                if not core_vals.empty:
                    core_len = len(str(core_vals.iloc[0]))
                    if core_len > 0:
                        label = f"{label} (core {core_len} bp)"
            ax_label = label_axes[idx]
            ax_left = axes_left[idx]
            ax_right = axes_right[idx]
            ax_label.set_axis_off()
            ax_label.text(
                0.98,
                0.5,
                label,
                ha="right",
                va="center",
                fontsize=text_sizes["regulator_label"] * 0.95,
                color="#111111",
            )
            core_hamming = diversity.get("core_hamming")
            if not isinstance(core_hamming, dict):
                raise ValueError(f"Stage-A diversity missing core_hamming for '{input_name}' ({reg}).")
            nnd_unweighted = diversity.get("nnd_unweighted_k1")
            if not isinstance(nnd_unweighted, dict):
                raise ValueError(f"Stage-A diversity missing unweighted nnd_k1 for '{input_name}' ({reg}).")
            top_candidates = nnd_unweighted.get("top_candidates")
            diversified_candidates = nnd_unweighted.get("diversified_candidates")
            if not isinstance(top_candidates, dict) or not isinstance(diversified_candidates, dict):
                raise ValueError(
                    (f"Stage-A diversity missing unweighted nnd_k1 top/diversified for '{input_name}' ({reg}).")
                )
            bins = top_candidates.get("bins") or diversified_candidates.get("bins")
            if not isinstance(bins, list) or not bins:
                raise ValueError(f"Stage-A diversity missing nnd_k1 bins for '{input_name}' ({reg}).")
            top_counts = top_candidates.get("counts")
            diversified_counts = diversified_candidates.get("counts")
            if not isinstance(top_counts, list) or not isinstance(diversified_counts, list):
                raise ValueError(f"Stage-A diversity missing nnd_k1 counts for '{input_name}' ({reg}).")
            x_base, y_base = _dist_from_counts(bins, top_counts)
            x_act, y_act = _dist_from_counts(bins, diversified_counts)
            base_line = ax_left.bar(
                x_base,
                y_base,
                width=0.8,
                color=top_color,
                alpha=0.45,
                label="Top",
                zorder=2,
            )
            act_line = ax_left.bar(
                x_act,
                y_act,
                width=0.8,
                color=diversified_color,
                alpha=0.6,
                label="Diversified",
                zorder=3,
            )
            x_min = float(min(x_base.min(), x_act.min()))
            x_max = float(max(x_base.max(), x_act.max()))
            ax_left.set_xlim(x_min - 0.5, x_max + 0.5)
            ax_left.set_ylim(0.0, max(1.05 * float(np.max([y_base.max(), y_act.max()])), 0.4))
            ax_left.set_ylabel("")
            tick_vals = x_base
            if x_base.size > 1:
                tick_vals = x_base[::2]
            ax_left.set_xticks(tick_vals)
            ax_left.tick_params(labelbottom=True)
            ax_left.tick_params(axis="x", labelsize=tick_size)
            if idx == 0:
                ax_left.legend(
                    handles=[base_line, act_line],
                    loc="lower right",
                    frameon=False,
                    fontsize=text_sizes["annotation"] * 0.95,
                )
            selection_policy = str(row.get("selection_policy") or "").lower()
            pool_size_final = row.get("selection_pool_size_final")
            retained_count = row.get("retained")
            is_degenerate = (
                selection_policy == "mmr"
                and pool_size_final is not None
                and retained_count is not None
                and int(pool_size_final) <= int(retained_count)
            )
            reg_df = pool_df[pool_df[tf_col].astype(str) == reg].copy()
            if "selection_score_norm" not in reg_df.columns:
                raise ValueError(f"Stage-A pool missing selection_score_norm for input '{input_name}' ({reg}).")
            if "nearest_selected_distance_norm" not in reg_df.columns:
                raise ValueError(
                    f"Stage-A pool missing nearest_selected_distance_norm for input '{input_name}' ({reg})."
                )
            if "selection_rank" not in reg_df.columns:
                raise ValueError(f"Stage-A pool missing selection_rank for input '{input_name}' ({reg}).")
            score_norm = pd.to_numeric(reg_df["selection_score_norm"], errors="coerce")
            dist_norm = pd.to_numeric(reg_df["nearest_selected_distance_norm"], errors="coerce")
            ranks = pd.to_numeric(reg_df["selection_rank"], errors="coerce")
            mask = ranks.notna() & score_norm.notna()
            if selection_policy != "mmr":
                ax_right.text(
                    0.5,
                    0.5,
                    f"selection_policy={selection_policy}\nMMR metadata unavailable",
                    ha="center",
                    va="center",
                    fontsize=text_sizes["annotation"] * 0.8,
                    color="#111111",
                )
            elif not mask.any():
                ax_right.text(
                    0.5,
                    0.5,
                    "MMR metadata unavailable",
                    ha="center",
                    va="center",
                    fontsize=text_sizes["annotation"] * 0.85,
                    color="#111111",
                )
            else:
                dist_line = None
                score_line = None
                ordered = ranks[mask].to_numpy(dtype=float)
                sort_idx = np.argsort(ordered)
                ordered = ordered[sort_idx]
                score_vals = score_norm[mask].to_numpy(dtype=float)[sort_idx]
                dist_vals = dist_norm[mask].to_numpy(dtype=float)[sort_idx]
                dist_mask = np.isfinite(dist_vals)
                if dist_mask.any():
                    dist_line = ax_right.plot(
                        ordered[dist_mask],
                        dist_vals[dist_mask],
                        color=hue,
                        marker="o",
                        linewidth=1.2,
                        markersize=3.5,
                        alpha=0.85,
                        label="Nearest selected dist",
                        zorder=3,
                    )
                    y_max = float(np.nanmax(dist_vals[dist_mask]))
                    ax_right.set_ylim(0.0, y_max * 1.15 if y_max > 0 else 1.0)
                if ordered.size:
                    x_max = float(np.nanmax(ordered))
                    if not np.isfinite(x_max) or x_max <= 1.0:
                        ax_right.set_xlim(0.5, 1.5)
                    else:
                        ax_right.set_xlim(1.0, x_max)
                ax_score = ax_right.twinx()
                score_line = ax_score.plot(
                    ordered,
                    score_vals,
                    color="#555555",
                    linestyle="--",
                    linewidth=1.0,
                    marker="s",
                    markersize=2.4,
                    alpha=0.6,
                    label="Score vs max",
                )
                ax_score.set_ylim(0.0, 1.0)
                ax_score.tick_params(labelright=True, labelsize=tick_size, pad=0)
                _apply_style(ax_score, style)
                if "right" in ax_score.spines:
                    ax_score.spines["right"].set_visible(True)
                if dist_line and score_line:
                    ax_right.legend(
                        handles=[dist_line[0], score_line[0]],
                        loc="lower left",
                        frameon=False,
                        fontsize=text_sizes["annotation"] * 1.05,
                    )
            if is_degenerate:
                ax_right.text(
                    0.02,
                    0.94,
                    "MMR degenerate:\npool ≤ n_sites\n(diversified = top-score)",
                    ha="left",
                    va="top",
                    transform=ax_right.transAxes,
                    fontsize=text_sizes["annotation"] * 0.75,
                    color="#111111",
                )
            ax_right.set_ylabel("")
            ax_right.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            ax_right.tick_params(labelbottom=True)
            ax_right.tick_params(axis="both", labelsize=tick_size)
        if axes_left and show_column_titles:
            axes_left[0].set_title("NN distance distribution", fontsize=subtitle_size, pad=title_pad)
            axes_right[0].set_title("Selection trajectory", fontsize=subtitle_size, pad=title_pad)
        if axes_left:
            axes_left[-1].set_xlabel(metric_label, fontsize=label_size)
            axes_right[-1].set_xlabel("MMR selection step", fontsize=label_size)
            y_center = (axes_left[0].get_position().y1 + axes_left[-1].get_position().y0) / 2.0
            left_bbox = axes_left[0].get_position()
            right_bbox = axes_right[0].get_position()
            fig.text(
                left_bbox.x0 - 0.04,
                y_center,
                "Fraction",
                rotation="vertical",
                ha="right",
                va="center",
                fontsize=label_size,
                color="#111111",
            )
            fig.text(
                right_bbox.x0 - 0.04,
                y_center,
                "Nearest selected dist",
                rotation="vertical",
                ha="right",
                va="center",
                fontsize=label_size,
                color="#111111",
            )
            fig.text(
                right_bbox.x1 + 0.06,
                y_center,
                "Score vs max",
                rotation="vertical",
                ha="left",
                va="center",
                fontsize=label_size,
                color="#111111",
            )

        for ax in axes_left + axes_right:
            _apply_style(ax, style)
            if ax in axes_right and "right" in ax.spines:
                ax.spines["right"].set_visible(True)
        for ax in axes_left:
            ax.tick_params(axis="y", labelsize=axis_label_size)
    return fig, axes_left, axes_right
