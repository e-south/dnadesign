"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/_mpl_utils.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def _apply_perf_rcparams() -> None:
    # Cheap wins for large point clouds
    plt.rcParams["agg.path.chunksize"] = int(
        os.getenv("OPAL_MPL_PATH_CHUNKSIZE", "10000")
    )
    plt.rcParams["path.simplify"] = True
    plt.rcParams["path.simplify_threshold"] = 0.0  # keep geometry intact


def scatter_smart(
    ax, x, y, *, s=16, alpha=0.85, rasterize_at=20000, edgecolors="none", **kw
):
    """
    Always deterministic; switches to rasterized draw above 'rasterize_at' points
    to prevent vector-graphics ballooning and crashes in backends like PDF/PS.

    No downsampling here (no fallbacks); just a drawing-mode choice.
    """
    _apply_perf_rcparams()
    x = np.asarray(x, dtype=np.float32)  # halves memory vs float64
    y = np.asarray(y, dtype=np.float32)
    rasterized = x.size >= int(rasterize_at)
    return ax.scatter(
        x,
        y,
        s=s,
        alpha=alpha,
        linewidths=0,
        edgecolors=edgecolors,
        rasterized=rasterized,
        **kw,
    )
