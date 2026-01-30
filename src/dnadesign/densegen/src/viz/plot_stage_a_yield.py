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
from .plot_common import _add_anchored_box, _apply_style, _format_percent, _shared_x_cleanup, _style
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
    eligible_hist = sampling.get("eligible_score_hist") or []
    if not eligible_hist:
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

    regs = [str(row.get("regulator") or "") for row in eligible_hist]
    stage_counts = []
    duplication_factors: dict[str, float] = {}
    pool_headrooms: dict[str, float] = {}
    hit_overlap: dict[str, float] = {}
    core_lengths: dict[str, int] = {}
    if "tfbs_core" in pool_df.columns:
        core_series = pool_df["tfbs_core"].astype(str)
        for reg, core in zip(pool_df[tf_col].astype(str).to_list(), core_series.to_list()):
            core_lengths.setdefault(reg, []).append(len(core))
        core_lengths = {reg: int(np.median(vals)) for reg, vals in core_lengths.items() if vals}
    for row in eligible_hist:
        reg = str(row.get("regulator") or "")
        generated = row.get("generated")
        candidates_with_hit = row.get("candidates_with_hit")
        eligible_raw = row.get("eligible_raw")
        eligible_unique = row.get("eligible_unique")
        retained = row.get("retained")
        if (
            generated is None
            or candidates_with_hit is None
            or eligible_raw is None
            or eligible_unique is None
            or retained is None
        ):
            raise ValueError(f"Stage-A sampling missing yield counters for input '{input_name}' ({reg}).")
        selection_pool = row.get("selection_shortlist_k")
        if selection_pool is None:
            selection_pool = row.get("selection_tier_limit")
        if selection_pool is None and eligible_unique is not None:
            selection_pool = eligible_unique
        if eligible_unique is not None and int(eligible_unique) > 0 and eligible_raw is not None:
            duplication_factors[reg] = float(eligible_raw) / float(eligible_unique)
        if retained is not None and int(retained) > 0 and selection_pool is not None:
            pool_headrooms[reg] = float(selection_pool) / float(retained)
        audit = row.get("padding_audit")
        if isinstance(audit, dict):
            overlap = audit.get("best_hit_overlaps_intended_core_fraction")
            if overlap is not None:
                hit_overlap[reg] = float(overlap)
        stage_counts.append([generated, candidates_with_hit, eligible_raw, eligible_unique, selection_pool, retained])

    if not regs:
        raise ValueError(f"Stage-A sampling missing regulator labels for input '{input_name}'.")
    base_keys = pool_df[tf_col].astype(str).tolist()
    lengths = pool_df[tfbs_col].astype(str).str.len().to_numpy(dtype=float)
    scores = pd.to_numeric(pool_df["best_hit_score"], errors="coerce")
    if scores.isna().all():
        raise ValueError(f"Stage-A pool missing best_hit_score values for input '{input_name}'.")
    scores_arr = scores.to_numpy(dtype=float)
    gc_vals = []
    for seq in pool_df[tfbs_col].astype(str).to_list():
        seq = str(seq).upper()
        if not seq:
            gc_vals.append(0.0)
        else:
            gc_vals.append(float(seq.count("G") + seq.count("C")) / float(len(seq)))
    gc_arr = np.asarray(gc_vals, dtype=float)
    tf_vals = pool_df[tf_col].astype(str).to_numpy()

    def _stable_jitter(keys: list[str], width: float = 0.18) -> np.ndarray:
        import hashlib

        values = []
        for key in keys:
            digest = hashlib.md5(key.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
            values.append((bucket - 0.5) * width + 0.0)
        return np.asarray(values) if values else np.zeros((0,), dtype=float)

    jitter = _stable_jitter(base_keys)
    lengths_j = lengths + jitter

    fig_width = float(style.get("figsize", (11, 4.2))[0])
    base_height = float(style.get("figsize", (11, 4.2))[1])
    reg_order = [reg for reg in regs if reg]
    if not reg_order:
        raise ValueError(f"Stage-A sampling missing regulator labels for input '{input_name}'.")
    n_regs = max(1, len(reg_order))
    fig_height = max(4.8, base_height, 1.75 * n_regs + 0.8)
    reg_colors = _stage_a_regulator_colors(reg_order, style)
    stage_labels = ["Generated", "Has hit", "Eligible", "Unique core", "MMR pool", "Retained"]
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
            axes_right.append(fig.add_subplot(main[idx, 1], sharex=share_right, sharey=share_right))
        cbar_ax = fig.add_subplot(body[0, 1])

        x_positions = np.arange(len(stage_labels))
        offset = max(1.0, max_count * 0.03) if max_count else 1.0
        y_limit = max_count * 1.25 + offset if max_count else 1.0
        for idx, reg in enumerate(reg_order):
            ax = axes_left[idx]
            counts = counts_by_reg.get(reg)
            if not counts:
                raise ValueError(f"Stage-A yield counts missing for '{input_name}' ({reg}).")
            hue = reg_colors.get(reg, "#4c78a8")
            ax.plot(x_positions, counts, color=hue, marker="o", linewidth=1.4, markersize=4)
            ax.set_ylim(0.0, y_limit)
            for step_idx, cur in enumerate(counts):
                prev = counts[step_idx - 1] if step_idx > 0 else cur
                frac = float(cur) / float(prev) if prev else 0.0
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
            note_lines = []
            dup = duplication_factors.get(reg)
            if dup is not None:
                note_lines.append(f"dup pressure: eligible/unique={dup:.1f}x")
            headroom = pool_headrooms.get(reg)
            if headroom is not None:
                note_lines.append(f"MMR headroom: pool/retained={headroom:.1f}x")
            if note_lines:
                _add_anchored_box(
                    ax,
                    note_lines,
                    loc="upper right",
                    fontsize=text_sizes["annotation"] * 0.7,
                    alpha=0.85,
                    edgecolor="none",
                )

        axes_left[0].set_title("Stepwise sequence yield", fontsize=subtitle_size, pad=title_pad)
        axes_left[-1].set_xticks(x_positions)
        axes_left[-1].set_xticklabels(stage_labels)
        axes_left[-1].set_xlabel("Stage")
        for ax in axes_left:
            ax.set_xlim(-0.5, len(stage_labels) - 1 + 0.5)
        _shared_x_cleanup(axes_left)

        for idx, reg in enumerate(reg_order):
            ax = axes_right[idx]
            mask = tf_vals == reg
            if not np.any(mask):
                raise ValueError(f"Stage-A pool missing retained sites for '{input_name}' ({reg}).")
            ax.scatter(
                lengths_j[mask],
                scores_arr[mask],
                c=gc_arr[mask],
                cmap="viridis",
                alpha=0.65,
                s=12,
                marker="o",
                edgecolors="none",
            )
            ax.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=5))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
            note_lines = []
            if mask.any():
                corr = pd.Series(lengths[mask]).corr(pd.Series(scores_arr[mask]), method="spearman")
                if corr is not None and np.isfinite(corr):
                    note_lines.append(f"rho(score,len) {float(corr):+.2f}")
            overlap = hit_overlap.get(reg)
            if overlap is not None:
                note_lines.append(f"hit overlap {float(overlap) * 100:.0f}%")
            if note_lines:
                _add_anchored_box(
                    ax,
                    note_lines,
                    loc="upper right",
                    fontsize=text_sizes["annotation"] * 0.7,
                    alpha=0.85,
                    edgecolor="none",
                )
        axes_right[0].set_title(
            "Retained sites: score vs length (GC color)",
            fontsize=subtitle_size,
            pad=title_pad,
        )
        axes_right[-1].set_xlabel("TFBS length (nt)")
        for ax in axes_right:
            ax.set_ylabel("Best-hit score")
            ax.tick_params(axis="y", labelsize=tick_size, labelleft=True)
            ax.tick_params(axis="x", labelsize=tick_size)
        x_min = float(np.nanmin(lengths)) if len(lengths) else 0.0
        x_max = float(np.nanmax(lengths)) if len(lengths) else 0.0
        if x_max <= x_min:
            raise ValueError(f"Stage-A pool missing TFBS length range for input '{input_name}'.")
        for ax in axes_right:
            ax.set_xlim(x_min - 1.0, x_max + 1.0)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("GC fraction", fontsize=text_sizes["annotation"] * 0.85)
        cbar.ax.tick_params(labelsize=tick_size)
        _shared_x_cleanup(axes_right)
        for ax in axes_left + axes_right:
            _apply_style(ax, style)
            ax.tick_params(axis="both", labelsize=tick_size)
    return fig, axes_left, axes_right, cbar_ax
