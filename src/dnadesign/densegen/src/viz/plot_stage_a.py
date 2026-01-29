"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_stage_a.py

Stage-A summary plotting (tiers, yield/bias, and core diversity).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from ..core.artifacts.pool import TFBSPoolArtifact
from ..utils.plot_style import format_regulator_label, stage_a_rcparams
from .plot_common import (
    _add_anchored_box,
    _apply_style,
    _draw_tier_markers,
    _format_percent,
    _safe_filename,
    _shared_axis_cleanup,
    _shared_x_cleanup,
    _style,
)


def _pastelize_color(color: str, amount: float = 0.6) -> tuple[float, float, float, float]:
    base = to_rgba(color)
    return (
        base[0] + (1.0 - base[0]) * amount,
        base[1] + (1.0 - base[1]) * amount,
        base[2] + (1.0 - base[2]) * amount,
        base[3],
    )


# color utils
try:
    from matplotlib.colors import is_color_like as _mpl_is_color_like
except Exception:
    _mpl_is_color_like = None


def _is_color_like(x) -> bool:
    if _mpl_is_color_like is not None:
        try:
            return bool(_mpl_is_color_like(x))
        except Exception:
            pass
    try:
        to_rgba(x)
        return True
    except Exception:
        return False


def _palette(style: dict, n: int, *, no_repeat: bool = False):
    pal = style.get("palette", "okabe_ito")
    if isinstance(pal, str):
        key = pal.lower().replace("-", "_")
        if key in {"okabe_ito", "okabeito", "colorblind", "colorblind2", "colorblind_2"}:
            base = [
                "#000000",
                "#E69F00",
                "#56B4E9",
                "#009E73",
                "#F0E442",
                "#0072B2",
                "#D55E00",
                "#CC79A7",
            ]
            if n <= len(base):
                return base[:n]
            if no_repeat:
                raise ValueError(
                    f"Need {n} unique colors; okabe_ito has {len(base)}. Provide a longer palette or reduce categories."
                )
            return [base[i % len(base)] for i in range(n)]
        if _is_color_like(pal):  # single color
            if no_repeat and n > 1:
                raise ValueError(f"Single color '{pal}' cannot provide {n} unique colors.")
            return [pal] * n
        try:  # colormap name
            cmap = plt.get_cmap(pal)
            return [cmap(i / max(1, n - 1)) for i in range(n)]
        except Exception:
            raise ValueError(f"Unknown palette or colormap name: {pal!r}")
    if isinstance(pal, (list, tuple)):
        base = list(pal)
        if len(base) >= n:
            return base[:n]
        if no_repeat:
            raise ValueError(f"Need {n} unique colors; got {len(base)} in explicit list.")
        return [base[i % len(base)] for i in range(n)]
    raise ValueError(f"Invalid palette type: {type(pal).__name__}")


def _stage_a_text_sizes(style: dict) -> dict[str, float]:
    font_size = float(style.get("font_size", 12.0))
    label_size = float(style.get("label_size", font_size))
    panel_title = float(style.get("title_size", font_size * 1.15))
    fig_title = float(style.get("fig_title_size", panel_title * 1.15))
    regulator_label = float(style.get("regulator_label_size", label_size * 0.95))
    sublabel = float(style.get("sublabel_size", label_size * 0.8))
    annotation = float(style.get("annotation_size", label_size * 0.72))
    return {
        "fig_title": fig_title,
        "panel_title": panel_title,
        "regulator_label": regulator_label,
        "sublabel": sublabel,
        "annotation": annotation,
    }


def _stage_a_regulator_colors(regulators: list[str], style: dict) -> dict[str, str]:
    base = _palette(style, max(len(regulators), 6), no_repeat=False)
    special = {"lexa": "#0072B2", "cpxr": "#009E73"}
    color_by_reg: dict[str, str] = {}
    used: set[str] = set()
    for reg in regulators:
        lowered = str(reg).strip().lower()
        if lowered.startswith("lexa"):
            color_by_reg[reg] = special["lexa"]
            used.add(special["lexa"])
        elif lowered.startswith("cpxr"):
            color_by_reg[reg] = special["cpxr"]
            used.add(special["cpxr"])
    available = [color for color in base if color not in used]
    if not available:
        available = list(base)
    idx = 0
    for reg in regulators:
        if reg in color_by_reg:
            continue
        color_by_reg[reg] = available[idx % len(available)]
        idx += 1
    return color_by_reg


def _build_stage_a_strata_overview_figure(
    *,
    input_name: str,
    pool_df: pd.DataFrame,
    sampling: dict,
    style: dict,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes, mpl.axes.Axes]:
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
            f"Stage-A pool tiers — {input_name}",
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
                    ("0.1%", tier0_score, str(retained.get(0, 0))),
                    ("1%", tier1_score, str(retained.get(1, 0))),
                    ("9%", tier2_score, str(retained.get(2, 0))),
                ],
                ymax_fraction=0.58,
                label_mode="box",
                loc="lower right",
                fontsize=text_sizes["annotation"] * 0.65,
            )
            ax.set_yticks([])
            ax.set_ylim(0, 1.05)
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
                "Eligible score tiers with retained overlays",
                fontsize=text_sizes["annotation"],
                color="#444444",
                pad=12,
                loc="center",
            )
            axes_left[-1].set_xlabel("FIMO log-odds score")
            axes_left[-1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            _shared_axis_cleanup(axes_left)

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
    return fig, axes_left[0], ax_right


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
    core_lengths: dict[str, int] = {}
    if "tfbs_core" in pool_df.columns:
        core_series = pool_df["tfbs_core"].astype(str)
        for reg, core in zip(pool_df[tf_col].astype(str).to_list(), core_series.to_list()):
            core_lengths.setdefault(reg, []).append(len(core))
        core_lengths = {reg: int(np.median(vals)) for reg, vals in core_lengths.items() if vals}
    for row in eligible_hist:
        stage_counts.append(
            [
                row.get("generated"),
                row.get("candidates_with_hit"),
                row.get("eligible"),
                row.get("eligible_unique"),
                row.get("retained"),
            ]
        )

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
    stage_labels = ["Generated", "Hit", "Eligible", "Unique", "Retained"]
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
            f"Stage-A yield & bias — {input_name}",
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

        stage_labels = [label.capitalize() for label in stage_labels]
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
            f"Stage-A core diversity (tfbs_core only; baseline vs actual) — {input_name}",
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
            overlap = diversity.get("overlap_actual_fraction")
            swaps = diversity.get("overlap_actual_swaps")
            if overlap is not None:
                overlap_label = f"overlap {float(overlap) * 100:.1f}%"
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
            axes_left[-1].set_xlabel("Hamming distance (k=5 neighbor)")
            axes_right[-1].set_xlabel("Core position")
            for ax in axes_left[:-1]:
                ax.tick_params(labelbottom=False)
            for ax in axes_right[:-1]:
                ax.tick_params(labelbottom=False)

        for ax in axes_left + axes_right:
            _apply_style(ax, style)
    return fig, axes_left, axes_right


def plot_stage_a_summary(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    if pools is None or pool_manifest is None:
        raise ValueError("Stage-A summary requires pool manifests; run stage-a build-pool first.")
    raw_style = style or {}
    style = _style(raw_style)
    style["seaborn_style"] = False
    if "figsize" not in raw_style:
        style["figsize"] = (11, 4)
    paths: list[Path] = []
    for input_name, pool_df in pools.items():
        entry = pool_manifest.entry_for(input_name)
        sampling = entry.stage_a_sampling
        if sampling is None:
            raise ValueError(f"Stage-A sampling metadata missing for input '{input_name}'.")
        eligible_hist = sampling.get("eligible_score_hist") or []
        if not eligible_hist:
            raise ValueError(f"Stage-A sampling missing eligible score histogram for input '{input_name}'.")
        for row in eligible_hist:
            if row.get("diversity") is None:
                raise ValueError(
                    f"Stage-A diversity metrics missing for input '{input_name}' ({row.get('regulator')}). "
                    "Rebuild Stage-A pools."
                )
        fig, _, _ = _build_stage_a_strata_overview_figure(
            input_name=input_name,
            pool_df=pool_df,
            sampling=sampling,
            style=style,
        )
        fname = f"{out_path.stem}__{_safe_filename(input_name)}{out_path.suffix}"
        path = out_path.parent / fname
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig)
        paths.append(path)

        fig2, _, _, _ = _build_stage_a_yield_bias_figure(
            input_name=input_name,
            pool_df=pool_df,
            sampling=sampling,
            style=style,
        )
        fname = f"{out_path.stem}__{_safe_filename(input_name)}__yield_bias{out_path.suffix}"
        path2 = out_path.parent / fname
        fig2.savefig(path2, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig2)
        paths.append(path2)

        fig3, _, _ = _build_stage_a_diversity_figure(
            input_name=input_name,
            pool_df=pool_df,
            sampling=sampling,
            style=style,
        )
        fname = f"{out_path.stem}__{_safe_filename(input_name)}__diversity{out_path.suffix}"
        path3 = out_path.parent / fname
        fig3.savefig(path3, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig3)
        paths.append(path3)
    return paths
