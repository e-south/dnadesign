"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/charts/sfxi_factorial_effects.py

Matplotlib chart for SFXI factorial-effects diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

from ....plots._mpl_utils import annotate_plot_meta, ensure_mpl_config_dir, scale_to_sizes, scatter_smart
from ...sfxi.factorial_effects import compute_factorial_effects
from ...sfxi.state_order import STATE_ORDER, assert_state_order
from ..util import list_series_to_numpy


def make_factorial_effects_figure(
    df: pl.DataFrame,
    *,
    logic_col: str,
    size_col: str | None = None,
    label_col: str | None = None,
    selected_col: str | None = None,
    top_k_col: str | None = None,
    title: str = "Factorial effects map",
    subtitle: str | None = None,
    alpha: float = 0.7,
    size_min: float = 14.0,
    size_max: float = 80.0,
    rasterize_at: int | None = None,
    cmap: str = "coolwarm",
    state_order: Sequence[str] = STATE_ORDER,
):
    assert_state_order(state_order)
    if df.is_empty():
        raise ValueError("Factorial effects plot requires non-empty data.")
    if logic_col not in df.columns:
        raise ValueError(f"Missing logic vector column: {logic_col}")

    vec = list_series_to_numpy(df.get_column(logic_col), expected_len=None)
    if vec is None:
        raise ValueError("Invalid logic vectors: expected list-like numeric values.")
    if vec.shape[1] < 4:
        raise ValueError("Logic vectors must have length >= 4.")
    v = vec[:, 0:4]
    a_eff, b_eff, ab_eff = compute_factorial_effects(v, state_order=state_order)

    sizes = np.full(a_eff.shape, float(size_min), dtype=float)
    if size_col is not None and size_col in df.columns:
        size_vals = df.select(pl.col(size_col).cast(pl.Float64, strict=False)).to_numpy().ravel()
        sizes = scale_to_sizes(size_vals, s_min=size_min, s_max=size_max)

    ensure_mpl_config_dir()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    sc = scatter_smart(
        ax,
        a_eff,
        b_eff,
        s=sizes,
        alpha=alpha,
        c=ab_eff,
        cmap=cmap,
        rasterize_at=rasterize_at,
    )
    ax.set_xlabel("A effect")
    ax.set_ylabel("B effect")
    ax.axhline(0.0, color="#B0B0B0", linewidth=0.8)
    ax.axvline(0.0, color="#B0B0B0", linewidth=0.8)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("AB interaction")

    if label_col and label_col in df.columns:
        mask = df.select(pl.col(label_col).fill_null(False)).to_numpy().ravel().astype(bool)
        if np.any(mask):
            ax.scatter(
                a_eff[mask],
                b_eff[mask],
                s=sizes[mask] * 1.4,
                facecolors="none",
                edgecolors="#000000",
                linewidths=1.2,
                alpha=1.0,
            )
    if selected_col and selected_col in df.columns:
        mask = df.select(pl.col(selected_col).fill_null(False)).to_numpy().ravel().astype(bool)
        if np.any(mask):
            ax.scatter(
                a_eff[mask],
                b_eff[mask],
                s=sizes[mask] * 1.8,
                marker="*",
                edgecolors="#000000",
                linewidths=0.8,
                alpha=1.0,
            )
    if top_k_col and top_k_col in df.columns:
        mask = df.select(pl.col(top_k_col).fill_null(False)).to_numpy().ravel().astype(bool)
        if np.any(mask):
            ax.scatter(
                a_eff[mask],
                b_eff[mask],
                s=sizes[mask] * 1.6,
                facecolors="none",
                edgecolors="#D55E00",
                linewidths=1.4,
                alpha=1.0,
            )

    annotate_plot_meta(
        ax,
        hue="AB interaction",
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
