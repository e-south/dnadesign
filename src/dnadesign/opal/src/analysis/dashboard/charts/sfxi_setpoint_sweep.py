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

from ....plots._mpl_utils import ensure_mpl_config_dir


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

    ensure_mpl_config_dir()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(setpoint_labels)), 2.4 + 0.4 * len(metrics)))
    ax.axis("off")

    col_labels = ["metric"] + [str(s) for s in setpoint_labels]
    table = ax.table(cellText=table_rows, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    if subtitle:
        ax.set_title(f"{title}\n{subtitle}")
    else:
        ax.set_title(title)
    fig.tight_layout()
    return fig
