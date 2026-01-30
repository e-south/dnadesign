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
    eligible_hist = sampling.get("eligible_score_hist") or []
    if not eligible_hist:
        raise ValueError(f"Stage-A sampling missing eligible score histogram for input '{input_name}'.")
    regulators = [str(row.get("regulator") or "") for row in eligible_hist]
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
            f"Stage-A core diversity (tfbs_core only; baseline vs actual) -- {input_name}",
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

        diversity_by_reg = {str(row.get("regulator") or ""): row.get("diversity") for row in eligible_hist}

        def _ecdf_from_counts(bins: list[float] | list[int], counts: list[int]) -> tuple[np.ndarray, np.ndarray]:
            if not bins or not counts:
                raise ValueError("Stage-A diversity missing k-NN bins or counts.")
            arr = np.asarray(counts, dtype=float)
            total = float(arr.sum())
            if total <= 0:
                raise ValueError("Stage-A diversity k-NN counts are empty.")
            x = np.asarray(bins, dtype=float)
            y = np.cumsum(arr) / total
            return x, y

        for idx, reg in enumerate(regulators):
            hue = reg_colors.get(reg, "#4c78a8")
            diversity = diversity_by_reg.get(reg) if isinstance(diversity_by_reg.get(reg), dict) else None
            if not isinstance(diversity, dict):
                raise ValueError(f"Stage-A diversity missing for input '{input_name}' ({reg}).")
            core_len = None
            entropy_block = diversity.get("core_entropy")
            if isinstance(entropy_block, dict):
                actual_vals = entropy_block.get("actual", {}).get("values", [])
                if actual_vals:
                    core_len = len(actual_vals)
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
            nnd_k5 = core_hamming.get("nnd_k5") if isinstance(core_hamming.get("nnd_k5"), dict) else None
            nnd_k1 = core_hamming.get("nnd_k1") if isinstance(core_hamming.get("nnd_k1"), dict) else None
            plot_block = nnd_k5 or nnd_k1
            if not isinstance(plot_block, dict):
                raise ValueError(f"Stage-A diversity missing k-NN stats for '{input_name}' ({reg}).")
            baseline = plot_block.get("baseline") if isinstance(plot_block.get("baseline"), dict) else None
            actual = plot_block.get("actual") if isinstance(plot_block.get("actual"), dict) else None
            if not baseline or not actual:
                raise ValueError(f"Stage-A diversity missing k-NN baseline/actual for '{input_name}' ({reg}).")
            bins = None
            if isinstance(baseline.get("bins"), list):
                bins = baseline.get("bins")
            elif isinstance(actual.get("bins"), list):
                bins = actual.get("bins")
            if not bins:
                raise ValueError(f"Stage-A diversity missing k-NN bins for '{input_name}' ({reg}).")
            base_ecdf = _ecdf_from_counts(bins, baseline.get("counts", []))
            act_ecdf = _ecdf_from_counts(bins, actual.get("counts", []))
            x_base, y_base = base_ecdf
            x_act, y_act = act_ecdf
            base_line = ax_left.step(
                x_base,
                y_base,
                where="post",
                color="#777777",
                linewidth=1.2,
                label="baseline",
            )[0]
            act_line = ax_left.step(
                x_act,
                y_act,
                where="post",
                color=hue,
                linewidth=1.5,
                label="actual",
            )[0]
            ax_left.fill_between(x_act, y_act, step="post", color=hue, alpha=0.12)
            ax_left.set_xlim(0, max(x_act.max(), x_base.max()))
            ax_left.set_ylim(0, 1.0)
            ax_left.set_ylabel("Fraction <= d" if idx == 0 else "")
            ax_left.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            if idx == 0:
                ax_left.legend(
                    handles=[base_line, act_line],
                    loc="lower right",
                    frameon=False,
                    fontsize=text_sizes["annotation"] * 0.8,
                )
            note_lines: list[str] = []
            if nnd_k5 is None and nnd_k1 is not None:
                note_lines.append("k=1 (n<6)")
            baseline_med = baseline.get("median")
            actual_med = actual.get("median")
            if baseline_med is not None and actual_med is not None:
                note_lines.append(f"delta div {float(actual_med) - float(baseline_med):+.2f}")
            pairwise = core_hamming.get("pairwise") if isinstance(core_hamming.get("pairwise"), dict) else None
            if isinstance(pairwise, dict):
                base_pair = pairwise.get("baseline") if isinstance(pairwise.get("baseline"), dict) else None
                act_pair = pairwise.get("actual") if isinstance(pairwise.get("actual"), dict) else None
                base_med = base_pair.get("median") if base_pair is not None else None
                act_med = act_pair.get("median") if act_pair is not None else None
                if base_med is not None and act_med is not None:
                    note_lines.append(f"delta pairwise {float(act_med) - float(base_med):+.2f}")
            overlap = diversity.get("set_overlap_fraction")
            swaps = diversity.get("set_overlap_swaps")
            if overlap is not None:
                overlap_label = f"set overlap {float(overlap) * 100:.1f}%"
                if swaps is not None:
                    overlap_label = f"{overlap_label} (swaps={int(swaps)})"
                note_lines.append(overlap_label)
            pool_size = diversity.get("candidate_pool_size")
            shortlist_target = diversity.get("shortlist_target")
            if pool_size is not None or shortlist_target is not None:
                note_lines.append(
                    f"k(pool/target) {pool_size if pool_size is not None else '-'}"
                    f"/{shortlist_target if shortlist_target is not None else '-'}"
                )
            score_block = diversity.get("score_quantiles")
            if isinstance(score_block, dict):
                base = score_block.get("baseline") if isinstance(score_block.get("baseline"), dict) else None
                actual = score_block.get("actual") if isinstance(score_block.get("actual"), dict) else None
                base_global = (
                    score_block.get("baseline_global") if isinstance(score_block.get("baseline_global"), dict) else None
                )
                if base is not None and actual is not None:
                    p10_delta = None
                    p50_delta = None
                    if base.get("p10") is not None and actual.get("p10") is not None:
                        p10_delta = float(actual.get("p10")) - float(base.get("p10"))
                    if base.get("p50") is not None and actual.get("p50") is not None:
                        p50_delta = float(actual.get("p50")) - float(base.get("p50"))
                    if p10_delta is not None or p50_delta is not None:
                        p10_text = f"{p10_delta:+.2f}" if p10_delta is not None else "-"
                        p50_text = f"{p50_delta:+.2f}" if p50_delta is not None else "-"
                        note_lines.append(f"delta score p10/med {p10_text} / {p50_text}")
                if base_global is not None and actual is not None:
                    g10_delta = None
                    g50_delta = None
                    if base_global.get("p10") is not None and actual.get("p10") is not None:
                        g10_delta = float(actual.get("p10")) - float(base_global.get("p10"))
                    if base_global.get("p50") is not None and actual.get("p50") is not None:
                        g50_delta = float(actual.get("p50")) - float(base_global.get("p50"))
                    if g10_delta is not None or g50_delta is not None:
                        g10_text = f"{g10_delta:+.2f}" if g10_delta is not None else "-"
                        g50_text = f"{g50_delta:+.2f}" if g50_delta is not None else "-"
                        note_lines.append(f"delta score global {g10_text} / {g50_text}")
            if note_lines:
                _add_anchored_box(
                    ax_left,
                    note_lines,
                    loc="upper left",
                    fontsize=text_sizes["annotation"] * 0.75,
                    alpha=0.9,
                    edgecolor="none",
                )
            core_entropy = diversity.get("core_entropy")
            if not isinstance(core_entropy, dict):
                raise ValueError(f"Stage-A diversity missing core_entropy for '{input_name}' ({reg}).")
            base_entropy = core_entropy.get("baseline", {}).get("values", [])
            actual_entropy = core_entropy.get("actual", {}).get("values", [])
            if not actual_entropy:
                raise ValueError(f"Stage-A diversity missing entropy values for '{input_name}' ({reg}).")
            positions = np.arange(1, len(actual_entropy) + 1)
            ax_right.bar(
                positions,
                actual_entropy,
                color=hue,
                alpha=0.65,
                width=0.8,
                label="actual",
            )
            if base_entropy:
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
            ax_right.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            if idx == 0:
                ax_right.legend(
                    loc="upper right",
                    frameon=False,
                    fontsize=text_sizes["annotation"] * 0.8,
                )
            base_sum = float(np.sum(base_entropy)) if base_entropy else None
            act_sum = float(np.sum(actual_entropy))
            sum_line = f"sumH {base_sum:.1f} -> {act_sum:.1f}" if base_sum is not None else f"sumH {act_sum:.1f}"
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
            axes_left[0].set_title("Core k-NN distance", fontsize=subtitle_size, pad=title_pad)
            axes_right[0].set_title("Core positional entropy", fontsize=subtitle_size, pad=title_pad)
            axes_left[-1].set_xlabel("Hamming distance (k-NN)")
            axes_right[-1].set_xlabel("Core position")
            for ax in axes_left[:-1]:
                ax.tick_params(labelbottom=False)
            for ax in axes_right[:-1]:
                ax.tick_params(labelbottom=False)

        for ax in axes_left + axes_right:
            _apply_style(ax, style)
    return fig, axes_left, axes_right
