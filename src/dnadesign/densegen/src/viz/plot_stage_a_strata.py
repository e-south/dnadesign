"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_stage_a_strata.py

Stage-A pool tier overview figure builder.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ..utils.plot_style import format_regulator_label, stage_a_rcparams
from .plot_common import _apply_style, _draw_tier_markers, _style
from .plot_stage_a_common import _pastelize_color, _stage_a_regulator_colors, _stage_a_text_sizes


def _build_stage_a_strata_overview_figure(
    *,
    input_name: str,
    pool_df: pd.DataFrame,
    sampling: dict,
    style: dict,
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes], mpl.axes.Axes]:
    style = _style(style)
    style["seaborn_style"] = False
    rc = stage_a_rcparams(style)
    text_sizes = _stage_a_text_sizes(style)
    if sampling.get("backend") != "fimo":
        raise ValueError(f"Stage-A strata overview requires FIMO sampling (input '{input_name}').")
    eligible_score_hist = sampling.get("eligible_score_hist") or []
    if not eligible_score_hist:
        raise ValueError(f"Stage-A sampling missing eligible score histogram for input '{input_name}'.")
    if "regulator_id" in pool_df.columns:
        tf_col = "regulator_id"
    elif "tf" in pool_df.columns:
        tf_col = "tf"
    else:
        raise ValueError(f"Stage-A pool missing regulator_id or tf column for input '{input_name}'.")
    if "tfbs_sequence" in pool_df.columns:
        tfbs_col = "tfbs_sequence"
    elif "tfbs" in pool_df.columns:
        tfbs_col = "tfbs"
    else:
        raise ValueError(f"Stage-A pool missing tfbs_sequence or tfbs column for input '{input_name}'.")
    if "best_hit_score" not in pool_df.columns:
        raise ValueError(f"Stage-A pool missing best_hit_score for input '{input_name}'.")

    regulators = [str(row.get("regulator") or "") for row in eligible_score_hist]
    if not regulators:
        raise ValueError(f"Stage-A sampling missing regulator labels for input '{input_name}'.")
    hist_by_reg: dict[str, tuple[list[float], list[int], float | None, float | None, float | None]] = {}
    global_scores: list[float] = []
    for row in eligible_score_hist:
        reg = str(row.get("regulator"))
        edges = [float(v) for v in (row.get("edges") or [])]
        counts = [int(v) for v in (row.get("counts") or [])]
        tier0_score = row.get("tier0_score")
        tier1_score = row.get("tier1_score")
        tier2_score = row.get("tier2_score")
        if not edges:
            raise ValueError(f"Stage-A eligible score histogram empty for input '{input_name}' ({reg}).")
        if len(counts) != len(edges) - 1:
            raise ValueError(f"Eligible score histogram length mismatch for '{input_name}' ({reg}).")
        for val in edges:
            global_scores.append(float(val))
        for val in (tier0_score, tier1_score, tier2_score):
            if val is not None:
                global_scores.append(float(val))
        hist_by_reg[reg] = (
            edges,
            counts,
            float(tier0_score) if tier0_score is not None else None,
            float(tier1_score) if tier1_score is not None else None,
            float(tier2_score) if tier2_score is not None else None,
        )

    base_colors = _stage_a_regulator_colors(regulators, style)
    color_by_reg = {reg: _pastelize_color(color, amount=0.35) for reg, color in base_colors.items()}

    n_regs = max(1, len(regulators))
    fig_width = float(style.get("figsize", (11, 4))[0])
    fig_height = max(3.8, 1.35 * n_regs + 1.2)
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
            0.86,
            f"Stage-A pool tiers -- {input_name}",
            ha="center",
            va="center",
            fontsize=text_sizes["fig_title"],
            color="#111111",
        )
        gs = outer[1].subgridspec(
            nrows=n_regs,
            ncols=2,
            width_ratios=[2.2, 1.1],
            hspace=0.28,
            wspace=0.28,
        )
        axes_left: list[mpl.axes.Axes] = []
        for idx in range(n_regs):
            ax = fig.add_subplot(gs[idx, 0], sharex=axes_left[0] if axes_left else None)
            axes_left.append(ax)
        ax_right = fig.add_subplot(gs[:, 1])

        retained_tiers = {}
        core_lengths: dict[str, int] = {}
        if "tfbs_core" in pool_df.columns:
            core_series = pool_df["tfbs_core"].astype(str)
            for reg, core in zip(pool_df[tf_col].astype(str).to_list(), core_series.to_list()):
                core_lengths.setdefault(reg, []).append(len(core))
            core_lengths = {reg: int(np.median(vals)) for reg, vals in core_lengths.items() if vals}
        if "tier" in pool_df.columns:
            tier_counts = pool_df.groupby([tf_col, "tier"], dropna=False).size().rename("count").reset_index()
            for _, row in tier_counts.iterrows():
                retained_tiers.setdefault(str(row[tf_col]), {})[int(row["tier"])] = int(row["count"])

        if not global_scores:
            raise ValueError(f"Stage-A eligible score histogram empty for input '{input_name}'.")
        global_min = float(min(global_scores))
        global_max = float(max(global_scores))
        pad = max(0.25, (global_max - global_min) * 0.03) if global_max > global_min else 0.25
        global_min -= pad
        global_max += pad
        for idx, reg in enumerate(regulators):
            ax = axes_left[idx]
            edges, counts, tier0_score, tier1_score, tier2_score = hist_by_reg.get(reg, ([], [], None, None, None))
            if not edges:
                raise ValueError(f"Stage-A eligible score histogram missing for '{input_name}' ({reg}).")
            counts_arr = np.asarray(counts, dtype=float)
            if counts_arr.size == 0:
                raise ValueError(f"Stage-A eligible score histogram empty for '{input_name}' ({reg}).")
            max_count = float(counts_arr.max()) if counts_arr.size else 0.0
            scale = max_count if max_count > 0 else 1.0
            density = counts_arr / scale
            centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2.0
            hue = color_by_reg.get(reg, "#4c78a8")
            ax.fill_between(centers, 0.0, density, color=hue, alpha=0.28)
            ax.plot(centers, density, color=hue, linewidth=1.2)
            retained_vals = pd.to_numeric(
                pool_df.loc[pool_df[tf_col].astype(str) == reg, "best_hit_score"],
                errors="coerce",
            ).dropna()
            if retained_vals.empty:
                raise ValueError(f"Stage-A retained scores missing for '{input_name}' ({reg}).")
            retained_counts, _ = np.histogram(retained_vals.to_numpy(dtype=float), bins=np.asarray(edges))
            retained_arr = np.asarray(retained_counts, dtype=float)
            if retained_arr.max() <= 0:
                raise ValueError(f"Stage-A retained score histogram empty for '{input_name}' ({reg}).")
            retained_density = retained_arr / scale
            ax.fill_between(centers, 0.0, retained_density, color=hue, alpha=0.5)
            retained = retained_tiers.get(reg, {})
            _draw_tier_markers(
                ax,
                [
                    ("Top 0.1% cutoff", tier0_score, str(retained.get(0, 0))),
                    ("Top 1% cutoff", tier1_score, str(retained.get(1, 0))),
                    ("Top 9% cutoff", tier2_score, str(retained.get(2, 0))),
                ],
                ymax_fraction=0.58,
                label_mode="box",
                loc="lower right",
                fontsize=text_sizes["annotation"] * 0.65,
            )
            ax.set_ylim(0, 1.05)
            retained_cutoff = float(retained_vals.min())
            y_min, y_max = ax.get_ylim()
            y_top = y_min + (y_max - y_min) * 0.58
            ax.axvline(
                retained_cutoff,
                ymin=0.0,
                ymax=0.58,
                linewidth=1.2,
                linestyle="-",
                color="#222222",
                alpha=0.95,
            )
            ax.scatter([retained_cutoff], [y_top], s=18, color="#222222", edgecolors="none", zorder=5)
            ax.annotate(
                "Retained cutoff",
                (retained_cutoff, y_top),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=text_sizes["annotation"] * 0.6,
                color="#222222",
            )
            ax.set_yticks([])
            label = format_regulator_label(reg)
            core_len = core_lengths.get(reg)
            ax.set_ylabel("")
            ax.text(
                -0.015,
                0.64,
                label,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=text_sizes["regulator_label"],
                color="#222222",
                clip_on=False,
            )
            if core_len:
                ax.text(
                    -0.015,
                    0.34,
                    f"(core {core_len} bp)",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=text_sizes["sublabel"],
                    color="#555555",
                    clip_on=False,
                )
            ax.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))
        for ax in axes_left:
            ax.set_xlim(global_min, global_max)

        if axes_left:
            axes_left[0].set_title(
                "Eligible unique cores: score distribution; retained subset highlighted",
                fontsize=text_sizes["annotation"],
                color="#444444",
                pad=12,
                loc="center",
            )
            axes_left[0].set_ylabel("Scaled density (peak=1)")
            axes_left[-1].set_xlabel("FIMO log-odds score")
            axes_left[-1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))

        lengths_by_reg: dict[str, list[int]] = {}
        for reg, seq in pool_df[[tf_col, tfbs_col]].itertuples(index=False):
            reg_label = str(reg)
            lengths_by_reg.setdefault(reg_label, []).append(len(str(seq)))
        for reg in regulators:
            if not lengths_by_reg.get(reg):
                raise ValueError(f"Stage-A pool missing TFBS lengths for '{input_name}' ({reg}).")
        all_lengths = [val for vals in lengths_by_reg.values() for val in vals]
        if not all_lengths:
            raise ValueError(f"Stage-A pool missing TFBS lengths for input '{input_name}'.")
        min_len = int(min(all_lengths))
        max_len = int(max(all_lengths))
        bins = np.arange(min_len - 0.5, max_len + 1.5, 1.0)
        for reg, lengths in lengths_by_reg.items():
            if not lengths:
                continue
            hue = color_by_reg.get(reg, "#4c78a8")
            ax_right.hist(
                lengths,
                bins=bins,
                density=False,
                alpha=0.25,
                color=hue,
                edgecolor=hue,
                linewidth=0.7,
            )
        span = max_len - min_len
        pad = max(5, int(round(span * 0.05))) if span > 0 else 5
        ax_right.set_xlim(min_len - pad, max_len + pad)
        ax_right.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=5))
        ax_right.set_xlabel("TFBS length (nt)")
        ax_right.set_ylabel("Count")
        ax_right.set_title(
            "Retained TFBS length counts",
            fontsize=text_sizes["annotation"],
            color="#444444",
            pad=12,
        )

        legend_handles = [
            Patch(
                facecolor=color_by_reg[reg],
                edgecolor=color_by_reg[reg],
                label=format_regulator_label(reg),
                alpha=0.35,
            )
            for reg in regulators
        ]
        if legend_handles:
            ax_right.legend(
                handles=legend_handles,
                loc="upper left",
                frameon=False,
                fontsize=text_sizes["annotation"] * 0.8,
            )

        for ax in axes_left + [ax_right]:
            _apply_style(ax, style)

    ax_left = axes_left[-1]
    ax_left.tick_params(axis="x", labelsize=text_sizes["annotation"] * 0.82)
    ax_right.tick_params(axis="x", labelsize=text_sizes["annotation"] * 0.8)
    ax_right.tick_params(axis="y", labelsize=text_sizes["annotation"] * 0.8)
    return fig, axes_left, ax_right
