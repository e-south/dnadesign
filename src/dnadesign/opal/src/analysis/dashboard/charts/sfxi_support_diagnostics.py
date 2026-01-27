"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/charts/sfxi_support_diagnostics.py

Support/extrapolation diagnostics charts for SFXI logic space.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ....plots._mpl_utils import annotate_plot_meta, ensure_mpl_config_dir, scale_to_sizes, scatter_smart


def make_support_diagnostics_figure(
    df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
    size_col: str | None = None,
    label_col: str | None = None,
    selected_col: str | None = None,
    top_k_col: str | None = None,
    title: str = "Logic support diagnostics",
    subtitle: str | None = None,
    alpha: float = 0.7,
    size_min: float = 14.0,
    size_max: float = 80.0,
    rasterize_at: int | None = None,
    cmap: str = "viridis",
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

    ensure_mpl_config_dir()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
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
    if selected_col and selected_col in df.columns:
        mask = df.select(pl.col(selected_col).fill_null(False)).to_numpy().ravel().astype(bool)
        _overlay(mask, scale=1.8, marker="*", edge="#000000")
    if top_k_col and top_k_col in df.columns:
        mask = df.select(pl.col(top_k_col).fill_null(False)).to_numpy().ravel().astype(bool)
        _overlay(mask, scale=1.6, marker="o", edge="#D55E00")

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
    fig.tight_layout()
    return fig
