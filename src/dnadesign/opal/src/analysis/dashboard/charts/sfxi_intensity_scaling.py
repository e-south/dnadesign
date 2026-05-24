"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/charts/sfxi_intensity_scaling.py

Intensity scaling diagnostics charts for SFXI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ....plots._mpl_utils import (
    COLORBLIND_PALETTE,
    apply_notebook_axes_style,
    apply_plot_style,
    pretty_label,
)
from .diagnostics_style import apply_diagnostics_title, diagnostics_figsize


def make_intensity_scaling_figure(
    sweep_df: pl.DataFrame,
    *,
    label_effect_raw: np.ndarray,
    pool_effect_raw: np.ndarray | None = None,
    title: str = "Intensity scaling diagnostics",
    subtitle: str | None = None,
):
    if sweep_df.is_empty():
        raise ValueError("Intensity scaling plot requires non-empty sweep data.")
    for col in ("setpoint_name", "denom_used", "clip_lo_fraction", "clip_hi_fraction"):
        if col not in sweep_df.columns:
            raise ValueError(f"Sweep data missing required column: {col}")

    labels = [str(v) for v in sweep_df.get_column("setpoint_name").to_list()]
    denom = np.asarray(sweep_df.get_column("denom_used").to_numpy(), dtype=float)
    clip_lo = np.asarray(sweep_df.get_column("clip_lo_fraction").to_numpy(), dtype=float)
    clip_hi = np.asarray(sweep_df.get_column("clip_hi_fraction").to_numpy(), dtype=float)

    label_raw = np.asarray(label_effect_raw, dtype=float).ravel()
    if label_raw.size == 0 or not np.all(np.isfinite(label_raw)):
        raise ValueError("label_effect_raw must be a non-empty finite array.")
    pool_raw = None
    if pool_effect_raw is not None:
        pool_raw = np.asarray(pool_effect_raw, dtype=float).ravel()
        if pool_raw.size == 0 or not np.all(np.isfinite(pool_raw)):
            raise ValueError("pool_effect_raw must be finite when provided.")

    apply_plot_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=diagnostics_figsize(width_scale=2.75, height_scale=0.86))
    ax_denom, ax_clip, ax_hist = axes
    for axis in (ax_denom, ax_clip, ax_hist):
        apply_notebook_axes_style(axis)

    x = np.arange(len(labels))
    ax_denom.bar(x, denom, color=COLORBLIND_PALETTE[0], alpha=0.85)
    ax_denom.set_title("Scaling denominator", fontsize=14)
    ax_denom.set_xticks(x)
    ax_denom.set_xticklabels(labels, rotation=45, ha="right")
    ax_denom.set_ylabel(pretty_label("denom_used"))
    ax_denom.tick_params(axis="x", labelsize=8)

    width = 0.35
    ax_clip.bar(x - width / 2, clip_lo, width=width, color=COLORBLIND_PALETTE[1], alpha=0.85, label="Lower")
    ax_clip.bar(x + width / 2, clip_hi, width=width, color=COLORBLIND_PALETTE[3], alpha=0.85, label="Upper")
    ax_clip.set_title("Clipping fraction", fontsize=14)
    ax_clip.set_xticks(x)
    ax_clip.set_xticklabels(labels, rotation=45, ha="right")
    ax_clip.set_ylabel("Fraction")
    ax_clip.legend(loc="upper right", frameon=False)
    ax_clip.tick_params(axis="x", labelsize=8)

    ax_hist.hist(label_raw, bins=20, color=COLORBLIND_PALETTE[2], alpha=0.72, label="Labels")
    if pool_raw is not None:
        ax_hist.hist(pool_raw, bins=20, color=COLORBLIND_PALETTE[4], alpha=0.55, label="Pool")
        ax_hist.legend(loc="upper right", frameon=False)
    ax_hist.set_title("Raw effect distribution", fontsize=14)
    ax_hist.set_xlabel(pretty_label("E_raw"))
    ax_hist.set_ylabel("Count")

    apply_diagnostics_title(fig, title=title, subtitle=subtitle, top=0.78)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.30, top=0.78, wspace=0.34)
    return fig
