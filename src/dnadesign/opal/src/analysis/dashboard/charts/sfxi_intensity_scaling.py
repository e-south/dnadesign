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

from ....plots._mpl_utils import apply_plot_style
from .diagnostics_style import diagnostics_figsize


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

    fig, axes = plt.subplots(
        1,
        3,
        figsize=diagnostics_figsize(width_scale=2.2, height_scale=0.9),
        constrained_layout=True,
    )
    ax_denom, ax_clip, ax_hist = axes

    x = np.arange(len(labels))
    ax_denom.bar(x, denom, color="#4C78A8", alpha=0.85)
    ax_denom.set_title("denom_used by setpoint")
    ax_denom.set_xticks(x)
    ax_denom.set_xticklabels(labels, rotation=45, ha="right")
    ax_denom.set_ylabel("denom_used")
    ax_denom.tick_params(axis="x", labelsize=8)

    width = 0.35
    ax_clip.bar(x - width / 2, clip_lo, width=width, color="#F58518", alpha=0.85, label="clip_lo")
    ax_clip.bar(x + width / 2, clip_hi, width=width, color="#E45756", alpha=0.85, label="clip_hi")
    ax_clip.set_title("clip fractions by setpoint")
    ax_clip.set_xticks(x)
    ax_clip.set_xticklabels(labels, rotation=45, ha="right")
    ax_clip.set_ylabel("fraction")
    ax_clip.legend(loc="best", fontsize=8)
    ax_clip.tick_params(axis="x", labelsize=8)

    ax_hist.hist(label_raw, bins=20, color="#54A24B", alpha=0.7, label="labels")
    if pool_raw is not None:
        ax_hist.hist(pool_raw, bins=20, color="#B279A2", alpha=0.5, label="pool")
        ax_hist.legend(loc="best", fontsize=8)
    ax_hist.set_title("E_raw distribution")
    ax_hist.set_xlabel("E_raw")
    ax_hist.set_ylabel("count")

    if subtitle:
        fig.suptitle(f"{title}\n{subtitle}")
    else:
        fig.suptitle(title)
    return fig
