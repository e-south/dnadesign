"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/charts/sfxi_setpoint_sweep.py

Setpoint sweep heatmap for SFXI diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

from ....plots._mpl_utils import (
    add_flush_colorbar,
    apply_notebook_axes_style,
    apply_plot_style,
    pretty_label,
    sequential_colormap,
)
from ...sfxi.setpoint_sweep import format_setpoint_label
from .diagnostics_style import DNAD_DIAGNOSTICS_PLOT_SIZE


def make_setpoint_sweep_figure(
    df: pl.DataFrame,
    *,
    metrics: Sequence[str],
    title: str = "Setpoint sweep",
    subtitle: str | None = None,
):
    if df.is_empty():
        raise ValueError("Setpoint sweep plot requires non-empty data.")
    if not metrics:
        raise ValueError("Setpoint sweep metrics must be non-empty.")
    for col in ("setpoint_name", *metrics):
        if col not in df.columns:
            raise ValueError(f"Setpoint sweep missing required column: {col}")

    if "setpoint_label" in df.columns:
        setpoint_labels = df.get_column("setpoint_label").to_list()
    elif "setpoint_vector" in df.columns:
        setpoint_labels = [format_setpoint_label(v) for v in df.get_column("setpoint_vector").to_list()]
    else:
        setpoint_labels = df.get_column("setpoint_name").to_list()

    values = np.zeros((len(metrics), len(setpoint_labels)), dtype=float)
    values[:] = np.nan
    for i, metric in enumerate(metrics):
        col_vals = df.get_column(metric).to_list()
        for j, val in enumerate(col_vals):
            if val is None or not np.isfinite(val):
                continue
            values[i, j] = float(val)

    apply_plot_style()
    import matplotlib.pyplot as plt

    cell = 0.38
    fig_w = max(float(DNAD_DIAGNOSTICS_PLOT_SIZE), len(setpoint_labels) * cell + 2.2)
    fig_h = max(3.4, len(metrics) * cell + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    mask = np.ma.masked_invalid(values)
    cmap = sequential_colormap("opal_seafoam")
    x_edges = np.arange(len(setpoint_labels) + 1)
    y_edges = np.arange(len(metrics) + 1)
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        mask,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        edgecolors="white",
        linewidth=0.8,
        shading="flat",
    )
    ax.set_xlim(0, len(setpoint_labels))
    ax.set_ylim(len(metrics), 0)
    ax.set_aspect("equal", adjustable="box")
    apply_notebook_axes_style(ax, grid=False, square=False)
    cbar = add_flush_colorbar(fig, ax, im)
    cbar.ax.set_title("Metric value", fontsize=11, pad=8)

    ax.set_xticks(np.arange(len(setpoint_labels)) + 0.5)
    ax.set_yticks(np.arange(len(metrics)) + 0.5)
    ax.set_xticklabels([str(s) for s in setpoint_labels], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([_metric_tick_label(m) for m in metrics])

    show_text = len(setpoint_labels) <= 16 and len(metrics) <= 6
    if show_text:
        for i in range(len(metrics)):
            for j in range(len(setpoint_labels)):
                val = values[i, j]
                if not np.isfinite(val):
                    label = "NA"
                else:
                    label = f"{val:.3f}"
                ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=7, color="black")

    rendered_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(rendered_title, pad=10, fontsize=15, linespacing=1.2)
    fig.subplots_adjust(left=0.22, right=0.82, bottom=0.30, top=0.82)
    return fig


def _metric_tick_label(metric: str) -> str:
    labels = {
        "logic_fidelity": r"Logic fidelity ($F_{\ell}$)",
        "effect_scaled": r"Scaled effect ($E_{\mathrm{scaled}}$)",
        "score": r"Score ($S$)",
    }
    return labels.get(str(metric), pretty_label(metric))
