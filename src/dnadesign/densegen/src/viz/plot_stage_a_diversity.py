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

from ..utils.plot_style import format_regulator_label, stage_a_rcparams
from .plot_common import _add_anchored_box, _apply_style, _style
from .plot_stage_a_common import _stage_a_regulator_colors, _stage_a_text_sizes


def _build_stage_a_diversity_figure(
    *,
    input_name: str,
    pool_df: pd.DataFrame,
    sampling: dict,
    style: dict,
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
    fig_width = float(style.get("figsize", (11, 4))[0])
    fig_height = max(4.0, 1.5 * n_regs + 1.1)
    subtitle_size = text_sizes["panel_title"] * 0.88
    title_pad = 12
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
        header_height = min(0.95, fig_height * 0.18)
        body_height = max(1.0, fig_height - header_height)
        outer = fig.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[header_height, body_height],
            hspace=0.05,
        )
        ax_header = fig.add_subplot(outer[0, 0])
        ax_header.set_axis_off()
        ax_header.set_label("header")
        ax_header.text(
            0.5,
            0.76,
            f"Stage-A core diversity (nearest-neighbor outcome + MMR contribution) -- {input_name}",
            ha="center",
            va="center",
            fontsize=text_sizes["fig_title"],
            color="#111111",
        )
        body = outer[1].subgridspec(
            nrows=n_regs,
            ncols=3,
            width_ratios=[0.22, 1.15, 1.0],
            hspace=0.32,
            wspace=0.35,
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

        metric_label = "Nearest-neighbor distance (k=1)"
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
                color="#222222",
            )
            core_hamming = diversity.get("core_hamming")
            if not isinstance(core_hamming, dict):
                raise ValueError(f"Stage-A diversity missing core_hamming for '{input_name}' ({reg}).")
            if "metric" not in core_hamming:
                raise ValueError(f"Stage-A diversity missing metric for '{input_name}' ({reg}).")
            metric = str(core_hamming["metric"]).strip()
            if not metric:
                raise ValueError(f"Stage-A diversity missing metric for '{input_name}' ({reg}).")
            if metric == "weighted_hamming_tolerant":
                metric_label = "Weighted Hamming nearest-neighbor distance (k=1)"
            nnd_block = core_hamming.get("nnd_k1")
            if not isinstance(nnd_block, dict):
                raise ValueError(f"Stage-A diversity missing nnd_k1 stats for '{input_name}' ({reg}).")
            top_candidates = nnd_block.get("top_candidates")
            diversified_candidates = nnd_block.get("diversified_candidates")
            if not isinstance(top_candidates, dict) or not isinstance(diversified_candidates, dict):
                raise ValueError(f"Stage-A diversity missing nnd_k1 top/diversified for '{input_name}' ({reg}).")
            bins = top_candidates.get("bins") or diversified_candidates.get("bins")
            if not isinstance(bins, list) or not bins:
                raise ValueError(f"Stage-A diversity missing nnd_k1 bins for '{input_name}' ({reg}).")
            top_counts = top_candidates.get("counts")
            diversified_counts = diversified_candidates.get("counts")
            if not isinstance(top_counts, list) or not isinstance(diversified_counts, list):
                raise ValueError(f"Stage-A diversity missing nnd_k1 counts for '{input_name}' ({reg}).")
            x_base, y_base = _dist_from_counts(bins, top_counts)
            x_act, y_act = _dist_from_counts(bins, diversified_counts)
            base_line = ax_left.plot(
                x_base,
                y_base,
                color="#777777",
                linewidth=1.3,
                label="Top Sequences",
                zorder=2,
            )[0]
            act_line = ax_left.plot(
                x_act,
                y_act,
                color=hue,
                linewidth=1.4,
                label="Diversified Sequences",
                zorder=3,
            )[0]
            ax_left.set_xlim(min(x_base.min(), x_act.min()), max(x_base.max(), x_act.max()))
            ax_left.set_ylim(0.0, max(1.05 * float(np.max([y_base.max(), y_act.max()])), 0.4))
            ax_left.set_ylabel("Fraction" if idx == 0 else "")
            ax_left.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            if idx == 0:
                ax_left.legend(
                    handles=[base_line, act_line],
                    loc="lower right",
                    frameon=False,
                    fontsize=text_sizes["annotation"] * 0.8,
                )
            note_lines: list[str] = []
            if "median" not in top_candidates or "median" not in diversified_candidates:
                raise ValueError(f"Stage-A diversity missing nnd_k1 median for '{input_name}' ({reg}).")
            base_med = float(top_candidates["median"])
            act_med = float(diversified_candidates["median"])
            note_lines.append(f"Δnnd (median) {act_med - base_med:+.2f}")
            objective_delta = diversity.get("objective_delta")
            if objective_delta is None:
                objective_top = diversity.get("objective_top_candidates")
                objective_diversified = diversity.get("objective_diversified_candidates")
                if objective_top is not None and objective_diversified is not None:
                    objective_delta = float(objective_diversified) - float(objective_top)
            if objective_delta is not None:
                note_lines.append(f"ΔJ {float(objective_delta):+.3f}")
            if "set_overlap_fraction" not in diversity:
                raise ValueError(f"Stage-A diversity missing overlap stats for '{input_name}' ({reg}).")
            overlap = float(diversity["set_overlap_fraction"])
            note_lines.append(f"overlap {overlap * 100:.1f}%")
            if note_lines:
                _add_anchored_box(
                    ax_left,
                    note_lines,
                    loc="upper left",
                    fontsize=text_sizes["annotation"] * 0.75,
                    alpha=0.9,
                    edgecolor="none",
                )
            selection_policy = str(row.get("selection_policy") or "").lower()
            reg_df = pool_df[pool_df[tf_col].astype(str) == reg].copy()
            if "selection_score_percentile" not in reg_df.columns:
                raise ValueError(f"Stage-A pool missing selection_score_percentile for input '{input_name}' ({reg}).")
            if "nearest_selected_distance_norm" not in reg_df.columns:
                raise ValueError(
                    f"Stage-A pool missing nearest_selected_distance_norm for input '{input_name}' ({reg})."
                )
            score_pct = pd.to_numeric(reg_df["selection_score_percentile"], errors="coerce")
            dist_norm = pd.to_numeric(reg_df["nearest_selected_distance_norm"], errors="coerce")
            ranks = (
                pd.to_numeric(reg_df["selection_rank"], errors="coerce") if "selection_rank" in reg_df.columns else None
            )
            mask = dist_norm.notna() & score_pct.notna()
            if ranks is not None:
                mask &= ranks > 1
            if selection_policy != "mmr":
                ax_right.text(
                    0.5,
                    0.5,
                    f"selection_policy={selection_policy}\nMMR metadata unavailable",
                    ha="center",
                    va="center",
                    fontsize=text_sizes["annotation"] * 0.8,
                    color="#666666",
                )
            elif not mask.any():
                ax_right.text(
                    0.5,
                    0.5,
                    "MMR metadata unavailable",
                    ha="center",
                    va="center",
                    fontsize=text_sizes["annotation"] * 0.85,
                    color="#666666",
                )
            else:
                score_norm = score_pct[mask].to_numpy(dtype=float)
                distances = dist_norm[mask].to_numpy(dtype=float)
                ax_right.scatter(
                    score_norm,
                    distances,
                    s=18,
                    color=hue,
                    alpha=0.75,
                    edgecolor="none",
                    zorder=3,
                )
                ax_right.set_xlim(0.0, 1.0)
                y_max = float(np.nanmax(distances)) if distances.size else 0.0
                ax_right.set_ylim(0.0, y_max * 1.15 if y_max > 0 else 1.0)
            ax_right.set_ylabel("Selection-time nearest distance (normalized)" if idx == 0 else "")
            ax_right.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            ax_left.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))
            ax_right.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))

        if axes_left:
            axes_left[0].set_title("Nearest-neighbor distance distribution", fontsize=subtitle_size, pad=title_pad)
            axes_right[0].set_title(
                "MMR contribution (score vs nearest distance)", fontsize=subtitle_size, pad=title_pad
            )
            axes_left[-1].set_xlabel(metric_label)
            axes_right[-1].set_xlabel("Score percentile (selection pool)")
            for ax in axes_left[:-1]:
                ax.tick_params(labelbottom=False)
            for ax in axes_right:
                ax.tick_params(labelbottom=True)

        for ax in axes_left + axes_right:
            _apply_style(ax, style)
    return fig, axes_left, axes_right
