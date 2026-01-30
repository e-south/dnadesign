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
            f"Stage-A core diversity (tfbs_core only; Top-score vs MMR) -- {input_name}",
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

        def _fraction_from_counts(bins: list[float] | list[int], counts: list[int]) -> tuple[np.ndarray, np.ndarray]:
            if not bins or not counts:
                raise ValueError("Stage-A diversity missing pairwise bins or counts.")
            arr = np.asarray(counts, dtype=float)
            total = float(arr.sum())
            x = np.asarray(bins, dtype=float)
            if total <= 0:
                return x, np.zeros_like(x, dtype=float)
            y = arr / total
            return x, y

        metric_label = "Hamming distance (pairwise)"
        for idx, reg in enumerate(regulators):
            hue = reg_colors.get(reg, "#4c78a8")
            row = row_by_reg[reg]
            diversity = diversity_by_reg[reg]
            core_entropy = diversity.get("core_entropy")
            if not isinstance(core_entropy, dict):
                raise ValueError(f"Stage-A diversity missing core_entropy for '{input_name}' ({reg}).")
            base_entropy_block = core_entropy.get("baseline")
            actual_entropy_block = core_entropy.get("actual")
            if not isinstance(base_entropy_block, dict) or not isinstance(actual_entropy_block, dict):
                raise ValueError(f"Stage-A diversity missing entropy blocks for '{input_name}' ({reg}).")
            base_entropy = base_entropy_block.get("values")
            actual_entropy = actual_entropy_block.get("values")
            if not isinstance(base_entropy, list) or not isinstance(actual_entropy, list):
                raise ValueError(f"Stage-A diversity missing entropy values for '{input_name}' ({reg}).")
            if not actual_entropy:
                raise ValueError(f"Stage-A diversity missing entropy values for '{input_name}' ({reg}).")
            if len(base_entropy) < len(actual_entropy):
                raise ValueError(f"Stage-A diversity entropy lengths mismatch for '{input_name}' ({reg}).")
            core_len = len(actual_entropy)
            label = format_regulator_label(reg)
            if core_len:
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
                metric_label = "Weighted Hamming distance (pairwise, 1-IC)"
            pairwise = core_hamming.get("pairwise")
            if not isinstance(pairwise, dict):
                raise ValueError(f"Stage-A diversity missing pairwise stats for '{input_name}' ({reg}).")
            baseline = pairwise.get("baseline")
            actual = pairwise.get("actual")
            if not isinstance(baseline, dict) or not isinstance(actual, dict):
                raise ValueError(f"Stage-A diversity missing pairwise baseline/actual for '{input_name}' ({reg}).")
            upper_bound = pairwise.get("upper_bound")
            if upper_bound is not None and not isinstance(upper_bound, dict):
                raise ValueError(f"Stage-A diversity upper_bound must be a dict for '{input_name}' ({reg}).")
            bins = baseline.get("bins") or actual.get("bins")
            if not isinstance(bins, list) or not bins:
                raise ValueError(f"Stage-A diversity missing pairwise bins for '{input_name}' ({reg}).")
            base_counts = baseline.get("counts")
            actual_counts = actual.get("counts")
            if not isinstance(base_counts, list) or not isinstance(actual_counts, list):
                raise ValueError(f"Stage-A diversity missing pairwise counts for '{input_name}' ({reg}).")
            x_base, y_base = _fraction_from_counts(bins, base_counts)
            x_act, y_act = _fraction_from_counts(bins, actual_counts)
            bar_width = 0.8
            if x_act.size > 1:
                bar_width = float(np.median(np.diff(x_act))) * 0.85
            act_bar = ax_left.bar(
                x_act,
                y_act,
                width=bar_width,
                color=hue,
                alpha=0.35,
                label="MMR",
                zorder=2,
            )[0]
            base_line = ax_left.step(
                x_base,
                y_base,
                where="mid",
                color="#777777",
                linewidth=1.3,
                label="Top-score",
                zorder=3,
            )[0]
            ax_left.set_xlim(x_base.min() - bar_width, x_base.max() + bar_width)
            max_val = float(max(y_base.max(), y_act.max()))
            if max_val <= 0:
                max_val = 1.0
            ax_left.set_ylim(0, max_val * 1.15)
            ax_left.set_ylabel("Fraction of pairs" if idx == 0 else "")
            ax_left.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            if idx == 0:
                ax_left.legend(
                    handles=[base_line, act_bar],
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
                base_med = baseline["median"]
                act_med = actual["median"]
                note_lines.append(f"delta pairwise {float(act_med) - float(base_med):+.2f}")
            if "set_overlap_fraction" not in diversity or "set_overlap_swaps" not in diversity:
                raise ValueError(f"Stage-A diversity missing overlap stats for '{input_name}' ({reg}).")
            overlap = diversity["set_overlap_fraction"]
            swaps = diversity["set_overlap_swaps"]
            overlap_label = f"baseline overlap {float(overlap) * 100:.1f}%"
            overlap_label = f"{overlap_label} (swaps={int(swaps)})"
            note_lines.append(overlap_label)
            if "candidate_pool_size" not in diversity or "shortlist_target" not in diversity:
                raise ValueError(f"Stage-A diversity missing pool sizing for '{input_name}' ({reg}).")
            pool_size = diversity["candidate_pool_size"]
            shortlist_target = diversity["shortlist_target"]
            note_lines.append(f"k(pool/target) {pool_size}/{shortlist_target}")
            shortlist_factor = row["selection_shortlist_factor"]
            shortlist_min = row["selection_shortlist_min"]
            retained = row["retained"]
            if shortlist_factor is not None and shortlist_min is not None and retained is not None:
                retained_int = int(retained)
                if retained_int > 0:
                    note_lines.append(f"target=max({int(shortlist_min)}, {int(shortlist_factor)}×{retained_int})")
            if upper_bound is not None:
                if "median" not in upper_bound:
                    raise ValueError(f"Stage-A diversity upper_bound missing median for '{input_name}' ({reg}).")
                note_lines.append(f"max-div med {float(upper_bound['median']):.2f}")
            score_block = diversity.get("score_quantiles")
            if not isinstance(score_block, dict):
                raise ValueError(f"Stage-A diversity missing score quantiles for '{input_name}' ({reg}).")
            base = score_block.get("baseline")
            actual = score_block.get("actual")
            if not isinstance(base, dict) or not isinstance(actual, dict):
                raise ValueError(
                    f"Stage-A diversity missing baseline/actual score quantiles for '{input_name}' ({reg})."
                )
            if "p10" not in base or "p50" not in base or "p10" not in actual or "p50" not in actual:
                raise ValueError(f"Stage-A diversity missing p10/p50 score quantiles for '{input_name}' ({reg}).")
            p10_delta = float(actual["p10"]) - float(base["p10"])
            p50_delta = float(actual["p50"]) - float(base["p50"])
            note_lines.append(f"delta score p10/med {p10_delta:+.2f} / {p50_delta:+.2f}")
            base_global = score_block.get("baseline_global")
            if base_global is not None and not isinstance(base_global, dict):
                raise ValueError(
                    f"Stage-A diversity baseline_global quantiles must be a dict for '{input_name}' ({reg})."
                )
            if base_global is not None:
                if "p10" not in base_global or "p50" not in base_global:
                    raise ValueError(
                        f"Stage-A diversity missing global p10/p50 score quantiles for '{input_name}' ({reg})."
                    )
                g10_delta = float(actual["p10"]) - float(base_global["p10"])
                g50_delta = float(actual["p50"]) - float(base_global["p50"])
                note_lines.append(f"delta score global {g10_delta:+.2f} / {g50_delta:+.2f}")
            if note_lines:
                _add_anchored_box(
                    ax_left,
                    note_lines,
                    loc="upper left",
                    fontsize=text_sizes["annotation"] * 0.75,
                    alpha=0.9,
                    edgecolor="none",
                )
            positions = np.arange(1, len(actual_entropy) + 1)
            ax_right.bar(
                positions,
                actual_entropy,
                color=hue,
                alpha=0.65,
                width=0.8,
                label="actual",
            )
            base_vals = base_entropy[: len(actual_entropy)]
            ax_right.plot(
                positions,
                base_vals,
                color="#777777",
                linewidth=1.2,
                marker="o",
                markersize=2.5,
                label="baseline",
            )
            ax_right.set_ylabel("Entropy (bits)" if idx == 0 else "")
            ax_right.set_xlim(0.5, len(actual_entropy) + 0.5)
            ax_right.set_ylim(0.0, 2.0)
            if "pwm_consensus" not in row:
                raise ValueError(f"Stage-A diversity missing pwm_consensus for '{input_name}' ({reg}).")
            consensus = str(row["pwm_consensus"] or "")
            if consensus and len(consensus) >= len(actual_entropy):
                labels = [f"{pos}:{base}" for pos, base in enumerate(consensus[: len(actual_entropy)], start=1)]
                ax_right.set_xticks(positions)
                ax_right.set_xticklabels(labels, fontsize=text_sizes["annotation"] * 0.7)
            else:
                ax_right.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            if idx == 0:
                ax_right.legend(
                    loc="upper right",
                    frameon=False,
                    fontsize=text_sizes["annotation"] * 0.8,
                )
            base_sum = float(np.sum(base_entropy))
            act_sum = float(np.sum(actual_entropy))
            sum_line = f"sumH {base_sum:.1f} -> {act_sum:.1f}"
            _add_anchored_box(
                ax_right,
                [sum_line],
                loc="upper left",
                fontsize=text_sizes["annotation"] * 0.7,
                alpha=0.85,
                edgecolor="none",
            )
            ax_left.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))
            ax_right.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))

        if axes_left:
            axes_left[0].set_title("Core pairwise distance", fontsize=subtitle_size, pad=title_pad)
            axes_right[0].set_title("Core positional entropy", fontsize=subtitle_size, pad=title_pad)
            axes_left[-1].set_xlabel(metric_label)
            axes_right[-1].set_xlabel("Core position")
            for ax in axes_left[:-1]:
                ax.tick_params(labelbottom=False)
            for ax in axes_right:
                ax.tick_params(labelbottom=True)

        for ax in axes_left + axes_right:
            _apply_style(ax, style)
    return fig, axes_left, axes_right
