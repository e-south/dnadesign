"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_stage_a_yield.py

Stage-A yield and bias figure builder.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.plot_style import format_regulator_label, stage_a_rcparams
from .plot_common import _apply_style, _format_percent, _shared_x_cleanup, _style
from .plot_stage_a_common import _stage_a_regulator_colors, _stage_a_text_sizes


def _build_stage_a_yield_bias_figure(
    *,
    input_name: str,
    pool_df: pd.DataFrame,
    sampling: dict,
    style: dict,
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes], list[mpl.axes.Axes], mpl.axes.Axes]:
    style = _style(style)
    style["seaborn_style"] = False
    rc = stage_a_rcparams(style)
    text_sizes = _stage_a_text_sizes(style)
    if "eligible_score_hist" not in sampling:
        raise ValueError(f"Stage-A sampling missing eligible score histogram for input '{input_name}'.")
    eligible_hist = sampling["eligible_score_hist"]
    if not isinstance(eligible_hist, list) or not eligible_hist:
        raise ValueError(f"Stage-A sampling missing eligible score histogram for input '{input_name}'.")
    if "regulator_id" in pool_df.columns:
        tf_col = "regulator_id"
    elif "tf" in pool_df.columns:
        tf_col = "tf"
    else:
        raise ValueError(f"Stage-A pool missing regulator_id or tf column for input '{input_name}'.")
    regs: list[str] = []
    for row in eligible_hist:
        if "regulator" not in row:
            raise ValueError(f"Stage-A sampling missing regulator labels for input '{input_name}'.")
        regs.append(str(row["regulator"]))
    stage_counts = []
    diversity_by_reg: dict[str, dict] = {}
    consensus_by_reg: dict[str, str] = {}
    core_lengths: dict[str, int] = {}
    for row in eligible_hist:
        reg = str(row["regulator"])
        consensus = row.get("pwm_consensus_iupac")
        if not consensus:
            raise ValueError(f"Stage-A sampling missing pwm_consensus_iupac for '{input_name}' ({reg}).")
        consensus = str(consensus)
        consensus_by_reg[reg] = consensus
        core_lengths[reg] = len(consensus)
        diversity = row.get("diversity")
        if not isinstance(diversity, dict):
            raise ValueError(f"Stage-A sampling missing diversity for '{input_name}' ({reg}).")
        diversity_by_reg[reg] = diversity
        if "generated" not in row:
            raise ValueError(f"Stage-A sampling missing generated count for '{input_name}' ({reg}).")
        if "eligible_raw" not in row:
            raise ValueError(f"Stage-A sampling missing yield counters for '{input_name}' ({reg}).")
        if "eligible_unique" not in row or "retained" not in row:
            raise ValueError(f"Stage-A sampling missing retained counters for '{input_name}' ({reg}).")
        generated = row["generated"]
        eligible_raw = row["eligible_raw"]
        eligible_unique = row["eligible_unique"]
        retained = row["retained"]
        if any(val is None for val in (generated, eligible_raw, eligible_unique, retained)):
            raise ValueError(f"Stage-A sampling missing yield counters for '{input_name}' ({reg}).")
        if "selection_pool_source" not in row:
            raise ValueError(f"Stage-A sampling missing selection_pool_source for '{input_name}' ({reg}).")
        pool_source = row["selection_pool_source"]
        if pool_source == "shortlist_k":
            if "selection_shortlist_k" not in row:
                raise ValueError(f"Stage-A selection missing shortlist size for '{input_name}' ({reg}).")
            selection_pool = row["selection_shortlist_k"]
        elif pool_source == "tier_limit":
            if "selection_tier_limit" not in row:
                raise ValueError(f"Stage-A selection missing tier limit for '{input_name}' ({reg}).")
            selection_pool = row["selection_tier_limit"]
        elif pool_source == "eligible_unique":
            selection_pool = row["eligible_unique"]
        else:
            raise ValueError(f"Stage-A selection_pool_source invalid for '{input_name}' ({reg}).")
        if selection_pool is None:
            raise ValueError(f"Stage-A selection pool size missing for '{input_name}' ({reg}).")
        stage_counts.append([generated, eligible_raw, eligible_unique, selection_pool, retained])

    if not regs:
        raise ValueError(f"Stage-A sampling missing regulator labels for input '{input_name}'.")
    pool_regs = set(pool_df[tf_col].astype(str).tolist())
    for reg in regs:
        if reg not in pool_regs:
            raise ValueError(f"Stage-A pool missing regulator '{reg}' for input '{input_name}'.")

    fig_width = float(style.get("figsize", (11, 4.2))[0])
    base_height = float(style.get("figsize", (11, 4.2))[1])
    reg_order = [reg for reg in regs if reg]
    if not reg_order:
        raise ValueError(f"Stage-A sampling missing regulator labels for input '{input_name}'.")
    n_regs = max(1, len(reg_order))
    fig_height = max(4.8, base_height, 1.75 * n_regs + 0.8)
    reg_colors = _stage_a_regulator_colors(reg_order, style)
    stage_labels = ["Generated", "Eligible", "Unique core", "MMR pool", "Retained"]
    counts_by_reg = {reg: counts for reg, counts in zip(regs, stage_counts)}
    max_count = max((max(counts) for counts in stage_counts), default=0)
    subtitle_size = text_sizes["panel_title"] * 0.88
    tick_size = text_sizes["annotation"] * 0.65
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
            0.74,
            f"Stage-A yield & bias -- {input_name}",
            ha="center",
            va="center",
            fontsize=text_sizes["fig_title"],
            color="#111111",
        )
        body = outer[1].subgridspec(
            nrows=1,
            ncols=2,
            width_ratios=[1.0, 0.05],
            wspace=0.12,
        )
        main = body[0, 0].subgridspec(
            nrows=n_regs,
            ncols=2,
            width_ratios=[1.0, 1.0],
            hspace=0.32,
            wspace=0.52,
        )
        axes_left: list[mpl.axes.Axes] = []
        axes_right: list[mpl.axes.Axes] = []
        for idx in range(n_regs):
            share_left = axes_left[0] if axes_left else None
            share_right = axes_right[0] if axes_right else None
            axes_left.append(fig.add_subplot(main[idx, 0], sharex=share_left, sharey=share_left))
            axes_right.append(fig.add_subplot(main[idx, 1], sharey=share_right))
        cbar_ax = fig.add_subplot(body[0, 1])
        cbar_ax.set_axis_off()

        x_positions = np.arange(len(stage_labels))
        offset = max(1.0, max_count * 0.03) if max_count else 1.0
        y_limit = max_count * 1.25 + offset if max_count else 1.0
        for idx, reg in enumerate(reg_order):
            ax = axes_left[idx]
            if reg not in counts_by_reg:
                raise ValueError(f"Stage-A yield counts missing for '{input_name}' ({reg}).")
            counts = counts_by_reg[reg]
            hue = reg_colors[reg]
            ax.plot(x_positions, counts, color=hue, marker="o", linewidth=1.4, markersize=4)
            ax.set_ylim(0.0, y_limit)
            generated = counts[0] if counts else 0
            for step_idx, cur in enumerate(counts):
                frac = float(cur) / float(generated) if generated else 0.0
                label = f"{cur:,}\n{_format_percent(frac)}"
                ax.annotate(
                    label,
                    (x_positions[step_idx], cur),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=text_sizes["annotation"] * 0.75,
                    color="#222222",
                )
            label = format_regulator_label(reg)
            ax.set_ylabel("")
            ax.text(
                -0.18,
                0.64,
                label,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=text_sizes["regulator_label"] * 0.84,
                color="#222222",
                clip_on=False,
            )
            core_len = core_lengths.get(str(reg))
            if core_len:
                ax.text(
                    -0.18,
                    0.34,
                    f"(core {core_len} bp)",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=text_sizes["sublabel"] * 0.84,
                    color="#555555",
                    clip_on=False,
                )
            ax.tick_params(axis="y", pad=1, labelsize=tick_size)
            ax.tick_params(axis="x", labelsize=tick_size)
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
            ax.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, integer=True))
        axes_left[0].set_title("Stepwise sequence yield", fontsize=subtitle_size, pad=title_pad)
        axes_left[-1].set_xticks(x_positions)
        axes_left[-1].set_xticklabels(stage_labels)
        axes_left[-1].set_xlabel("Stage")
        for ax in axes_left:
            ax.set_xlim(-0.5, len(stage_labels) - 1 + 0.5)
        _shared_x_cleanup(axes_left)

        for idx, reg in enumerate(reg_order):
            ax = axes_right[idx]
            diversity = diversity_by_reg.get(reg)
            if not isinstance(diversity, dict):
                raise ValueError(f"Stage-A diversity missing for '{input_name}' ({reg}).")
            entropy_block = diversity.get("core_entropy")
            if not isinstance(entropy_block, dict):
                raise ValueError(f"Stage-A diversity missing core_entropy for '{input_name}' ({reg}).")
            top_block = entropy_block.get("top_candidates")
            if not isinstance(top_block, dict):
                raise ValueError(f"Stage-A diversity missing top entropy for '{input_name}' ({reg}).")
            top_values = top_block.get("values")
            if not isinstance(top_values, list) or not top_values:
                raise ValueError(f"Stage-A diversity missing top entropy values for '{input_name}' ({reg}).")
            diversified_block = entropy_block.get("diversified_candidates")
            if not isinstance(diversified_block, dict):
                raise ValueError(f"Stage-A diversity missing diversified entropy for '{input_name}' ({reg}).")
            values = diversified_block.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"Stage-A diversity missing entropy values for '{input_name}' ({reg}).")
            consensus = consensus_by_reg.get(reg)
            if consensus is None:
                raise ValueError(f"Stage-A sampling missing pwm_consensus for '{input_name}' ({reg}).")
            if len(consensus) != len(values):
                raise ValueError(
                    f"Stage-A diversity entropy length mismatch for '{input_name}' ({reg}). "
                    f"consensus={len(consensus)} entropy={len(values)}"
                )
            if len(consensus) != len(top_values):
                raise ValueError(
                    f"Stage-A diversity top entropy length mismatch for '{input_name}' ({reg}). "
                    f"consensus={len(consensus)} entropy={len(top_values)}"
                )
            entropy_vals = [float(v) for v in values]
            top_entropy_vals = [float(v) for v in top_values]
            positions = np.arange(1, len(entropy_vals) + 1)
            hue = reg_colors.get(reg, "#4c78a8")
            ax.bar(
                positions,
                entropy_vals,
                color=hue,
                alpha=0.35,
                edgecolor=hue,
                linewidth=0.8,
            )
            ax.plot(positions, entropy_vals, color=hue, linewidth=1.2, label="Diversified")
            ax.plot(
                positions,
                top_entropy_vals,
                color="#444444",
                linewidth=1.1,
                linestyle="--",
                label="Top score",
            )
            ax.set_xlim(0.5, len(entropy_vals) + 0.5)
            ax.set_ylim(0.0, 2.0)
            ax.set_xticks(positions)
            ax.set_xticklabels(list(consensus))
            ax.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
            ax.tick_params(axis="y", labelsize=tick_size, labelleft=True)
            ax.tick_params(axis="x", labelsize=tick_size)
            if idx == 0:
                ax.set_ylabel("Entropy (bits)")
                ax.legend(
                    loc="upper right",
                    frameon=False,
                    fontsize=text_sizes["annotation"] * 0.75,
                )
            if idx == len(reg_order) - 1:
                ax.set_xlabel("Core position")
        axes_right[0].set_title(
            "Core positional entropy (top vs diversified)",
            fontsize=subtitle_size,
            pad=title_pad,
        )
        for ax in axes_left + axes_right:
            _apply_style(ax, style)
            ax.tick_params(axis="both", labelsize=tick_size)
    return fig, axes_left, axes_right, cbar_ax
