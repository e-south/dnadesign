"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/charts/sfxi_uncertainty.py

Uncertainty diagnostics charts for SFXI dashboards.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ....plots._mpl_utils import annotate_plot_meta, apply_plot_style, scale_to_sizes, scatter_smart
from .diagnostics_style import diagnostics_figsize


def make_uncertainty_figure(
    df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
    size_col: str | None = None,
    title: str = "Uncertainty diagnostics",
    subtitle: str | None = None,
    alpha: float = 0.7,
    size_min: float = 14.0,
    size_max: float = 80.0,
    rasterize_at: int | None = None,
    cmap: str = "viridis",
):
    if df.is_empty():
        raise ValueError("Uncertainty plot requires non-empty data.")
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    x = df.select(pl.col(x_col).cast(pl.Float64, strict=False)).to_numpy().ravel()
    y = df.select(pl.col(y_col).cast(pl.Float64, strict=False)).to_numpy().ravel()
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Uncertainty plot requires finite x/y values.")

    sizes = np.full(x.shape, float(size_min), dtype=float)
    if size_col is not None and size_col in df.columns:
        size_vals = df.select(pl.col(size_col).cast(pl.Float64, strict=False)).to_numpy().ravel()
        sizes = scale_to_sizes(size_vals, s_min=size_min, s_max=size_max)

    c = None
    if hue_col is not None:
        if hue_col not in df.columns:
            raise ValueError(f"Missing hue column: {hue_col}")
        c = df.select(pl.col(hue_col).cast(pl.Float64, strict=False)).to_numpy().ravel()

    apply_plot_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=diagnostics_figsize(), constrained_layout=True)
    sc = scatter_smart(
        ax,
        x,
        y,
        s=sizes,
        alpha=alpha,
        c=c,
        cmap=cmap,
        rasterize_at=rasterize_at,
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if c is not None:
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(hue_col)

    annotate_plot_meta(
        ax,
        hue=hue_col,
        size_by=size_col,
        alpha=alpha,
        rasterized=bool(rasterize_at),
    )
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}")
    else:
        ax.set_title(title)
    return fig
