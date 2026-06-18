"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/dashboard/charts/sfxi_support_diagnostics.py

Support/extrapolation diagnostics charts for SFXI logic space.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ....plots._mpl_utils import (
    add_flush_colorbar,
    annotate_plot_meta,
    apply_notebook_axes_style,
    apply_plot_style,
    math_label,
    pretty_label,
    scale_to_sizes,
    scatter_smart,
)
from .diagnostics_style import diagnostics_figsize, finalize_single_panel_diagnostics


def make_support_diagnostics_figure(
    df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
    size_col: str | None = None,
    label_col: str | None = None,
    title: str = "Logic support diagnostics",
    subtitle: str | None = None,
    alpha: float = 0.7,
    size_min: float = 14.0,
    size_max: float = 80.0,
    rasterize_at: int | None = None,
    cmap: str = "viridis",
    show_meta: bool = False,
):
    if df.is_empty():
        raise ValueError("Support diagnostics plot requires non-empty data.")
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    x = df.select(pl.col(x_col).cast(pl.Float64, strict=False)).to_numpy().ravel()
    y = df.select(pl.col(y_col).cast(pl.Float64, strict=False)).to_numpy().ravel()
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Support diagnostics requires finite x/y values.")

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

    fig, ax = plt.subplots(figsize=diagnostics_figsize())
    apply_notebook_axes_style(ax, square=True)
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
    ax.set_xlabel(math_label("support_distance") if x_col == "dist_to_labeled_logic" else pretty_label(x_col))
    ax.set_ylabel(math_label(y_col))
    if c is not None:
        cb = add_flush_colorbar(fig, ax, sc, pad=0.045)
        cb.set_label(math_label(hue_col), labelpad=8)

    def _overlay(mask: np.ndarray, *, scale: float, marker: str, edge: str):
        if np.any(mask):
            ax.scatter(
                x[mask],
                y[mask],
                s=sizes[mask] * scale,
                marker=marker,
                facecolors="none",
                edgecolors=edge,
                linewidths=1.2,
                alpha=1.0,
            )

    if label_col and label_col in df.columns:
        mask = df.select(pl.col(label_col).fill_null(False)).to_numpy().ravel().astype(bool)
        _overlay(mask, scale=1.4, marker="o", edge="#000000")

    if show_meta:
        annotate_plot_meta(
            ax,
            hue=hue_col,
            size_by=size_col,
            alpha=alpha,
            rasterized=bool(rasterize_at),
        )
    finalize_single_panel_diagnostics(fig, title=title, subtitle=subtitle, right=0.75 if c is not None else 0.84)
    return fig
