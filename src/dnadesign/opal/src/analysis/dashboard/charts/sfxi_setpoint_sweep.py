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

from ....plots._mpl_utils import apply_plot_style
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

    n_cols = len(setpoint_labels)
    n_rows = len(metrics)
    cell = float(DNAD_DIAGNOSTICS_PLOT_SIZE) * 0.22
    width = max(float(DNAD_DIAGNOSTICS_PLOT_SIZE) * 1.2, cell * max(n_cols, 1))
    height = max(float(DNAD_DIAGNOSTICS_PLOT_SIZE) * 1.2, cell * max(n_rows, 1))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    mask = np.ma.masked_invalid(values)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#DDDDDD")
    im = ax.imshow(mask, aspect="equal", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(setpoint_labels)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([str(s) for s in setpoint_labels], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([str(m).replace("_", " ") for m in metrics], fontsize=9)

    show_text = len(setpoint_labels) <= 16 and len(metrics) <= 6
    if show_text:
        for i in range(len(metrics)):
            for j in range(len(setpoint_labels)):
                val = values[i, j]
                if not np.isfinite(val):
                    label = "NA"
                else:
                    label = f"{val:.3f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=7, color="black")

    if subtitle:
        ax.set_title(f"{title}\n{subtitle}")
    else:
        ax.set_title(title)
    return fig
