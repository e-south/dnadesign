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
            f"Stage-A core diversity (pairwise outcome + MMR contribution) -- {input_name}",
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

        def _ecdf_from_counts(bins: list[float] | list[int], counts: list[int]) -> tuple[np.ndarray, np.ndarray]:
            if not bins or not counts:
                raise ValueError("Stage-A diversity missing pairwise bins or counts.")
            arr = np.asarray(counts, dtype=float)
            total = float(arr.sum())
            x = np.asarray(bins, dtype=float)
            if total <= 0:
                return x, np.zeros_like(x, dtype=float)
            y = np.cumsum(arr) / total
            return x, y

        metric_label = "Hamming distance (pairwise)"
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
                metric_label = "Weighted Hamming distance (pairwise)"
            pairwise = core_hamming.get("pairwise")
            if not isinstance(pairwise, dict):
                raise ValueError(f"Stage-A diversity missing pairwise stats for '{input_name}' ({reg}).")
            baseline = pairwise.get("baseline")
            actual = pairwise.get("actual")
            if not isinstance(baseline, dict) or not isinstance(actual, dict):
                raise ValueError(f"Stage-A diversity missing pairwise baseline/actual for '{input_name}' ({reg}).")
            bins = baseline.get("bins") or actual.get("bins")
            if not isinstance(bins, list) or not bins:
                raise ValueError(f"Stage-A diversity missing pairwise bins for '{input_name}' ({reg}).")
            base_counts = baseline.get("counts")
            actual_counts = actual.get("counts")
            if not isinstance(base_counts, list) or not isinstance(actual_counts, list):
                raise ValueError(f"Stage-A diversity missing pairwise counts for '{input_name}' ({reg}).")
            x_base, y_base = _ecdf_from_counts(bins, base_counts)
            x_act, y_act = _ecdf_from_counts(bins, actual_counts)
            base_line = ax_left.step(
                x_base,
                y_base,
                where="mid",
                color="#777777",
                linewidth=1.3,
                label="Top-score",
                zorder=2,
            )[0]
            act_line = ax_left.step(
                x_act,
                y_act,
                where="mid",
                color=hue,
                linewidth=1.4,
                label="MMR",
                zorder=3,
            )[0]
            ax_left.set_xlim(min(x_base.min(), x_act.min()), max(x_base.max(), x_act.max()))
            ax_left.set_ylim(0.0, 1.05)
            ax_left.set_ylabel("ECDF" if idx == 0 else "")
            ax_left.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            if idx == 0:
                ax_left.legend(
                    handles=[base_line, act_line],
                    loc="lower right",
                    frameon=False,
                    fontsize=text_sizes["annotation"] * 0.8,
                )
            note_lines: list[str] = []
            if "n_pairs" not in baseline or "n_pairs" not in actual:
                raise ValueError(f"Stage-A diversity missing pairwise n_pairs for '{input_name}' ({reg}).")
            base_pairs = int(baseline["n_pairs"])
            act_pairs = int(actual["n_pairs"])
            if base_pairs <= 0 or act_pairs <= 0:
                note_lines.append("pairwise n/a (n<2)")
            else:
                if "median" not in baseline or "median" not in actual:
                    raise ValueError(f"Stage-A diversity missing pairwise median for '{input_name}' ({reg}).")
                base_med = float(baseline["median"])
                act_med = float(actual["median"])
                note_lines.append(f"Δdiv (median) {act_med - base_med:+.2f}")
            objective_delta = diversity.get("objective_delta")
            if objective_delta is None:
                objective_base = diversity.get("objective_baseline")
                objective_actual = diversity.get("objective_actual")
                if objective_base is not None and objective_actual is not None:
                    objective_delta = float(objective_actual) - float(objective_base)
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
            pwm_max_score = row.get("pwm_max_score")
            if pwm_max_score is None:
                raise ValueError(f"Stage-A diversity missing pwm_max_score for '{input_name}' ({reg}).")
            reg_df = pool_df[pool_df[tf_col].astype(str) == reg].copy()
            if "nearest_selected_similarity" not in reg_df.columns:
                raise ValueError(f"Stage-A pool missing nearest_selected_similarity for input '{input_name}' ({reg}).")
            scores = pd.to_numeric(reg_df["best_hit_score"], errors="coerce")
            sims = pd.to_numeric(reg_df["nearest_selected_similarity"], errors="coerce")
            ranks = (
                pd.to_numeric(reg_df["selection_rank"], errors="coerce") if "selection_rank" in reg_df.columns else None
            )
            mask = sims.notna() & (sims > 0)
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
                score_norm = (scores[mask] / float(pwm_max_score)).to_numpy(dtype=float)
                distances = ((1.0 / sims[mask]) - 1.0).to_numpy(dtype=float)
                ax_right.scatter(
                    score_norm,
                    distances,
                    s=18,
                    color=hue,
                    alpha=0.75,
                    edgecolor="none",
                    zorder=3,
                )
                x_min = float(np.nanmin(score_norm))
                x_max = float(np.nanmax(score_norm))
                if not np.isfinite(x_min) or not np.isfinite(x_max):
                    x_min, x_max = 0.0, 1.0
                pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.1
                ax_right.set_xlim(x_min - pad, x_max + pad)
                y_max = float(np.nanmax(distances)) if distances.size else 0.0
                ax_right.set_ylim(0.0, y_max * 1.15 if y_max > 0 else 1.0)
            ax_right.set_ylabel("Selection-time nearest distance" if idx == 0 else "")
            ax_right.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            ax_left.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))
            ax_right.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))

        if axes_left:
            axes_left[0].set_title("Pairwise distance ECDF", fontsize=subtitle_size, pad=title_pad)
            axes_right[0].set_title(
                "MMR contribution (score vs nearest distance)", fontsize=subtitle_size, pad=title_pad
            )
            axes_left[-1].set_xlabel(metric_label)
            axes_right[-1].set_xlabel("Score / PWM max")
            for ax in axes_left[:-1]:
                ax.tick_params(labelbottom=False)
            for ax in axes_right:
                ax.tick_params(labelbottom=True)

        for ax in axes_left + axes_right:
            _apply_style(ax, style)
    return fig, axes_left, axes_right
