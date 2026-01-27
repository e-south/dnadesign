"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/charts/sfxi_setpoint_sweep.py

Setpoint sweep summary table/heatmap for SFXI diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

from ....plots._mpl_utils import apply_plot_style
from .diagnostics_style import diagnostics_table_figsize


def make_setpoint_sweep_figure(
    df: pl.DataFrame,
    *,
    metrics: Sequence[str],
    title: str = "Setpoint sweep",
    subtitle: str | None = None,
):
    if df.is_empty():
        raise ValueError("Setpoint sweep plot requires non-empty data.")
    for col in ("setpoint_name", *metrics):
        if col not in df.columns:
            raise ValueError(f"Setpoint sweep missing required column: {col}")

    setpoint_labels = df.get_column("setpoint_name").to_list()
    table_rows = []
    for metric in metrics:
        values = df.get_column(metric).to_list()
        row = [metric] + [f"{float(v):.3f}" if v is not None and np.isfinite(v) else "NA" for v in values]
        table_rows.append(row)

    apply_plot_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=diagnostics_table_figsize(n_cols=len(setpoint_labels) + 1, n_rows=len(metrics)),
        constrained_layout=True,
    )
    ax.axis("off")

    col_labels = ["metric"] + [str(s) for s in setpoint_labels]
    table = ax.table(cellText=table_rows, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    n_cols = len(col_labels)
    if n_cols <= 10:
        font_size = 9
    elif n_cols <= 16:
        font_size = 8
    else:
        font_size = 7
    table.set_fontsize(font_size)
    table.scale(1.0, 1.2)

    if subtitle:
        ax.set_title(f"{title}\n{subtitle}")
    else:
        ax.set_title(title)
    return fig
