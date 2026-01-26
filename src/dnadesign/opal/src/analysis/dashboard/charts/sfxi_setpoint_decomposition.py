"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/charts/sfxi_setpoint_decomposition.py

Setpoint decomposition charts for residuals and intensity contributions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ....objectives import sfxi_math
from ....plots._mpl_utils import ensure_mpl_config_dir
from ...sfxi.state_order import STATE_ORDER, assert_state_order


def _to_grid(vec4: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec4, dtype=float).ravel()
    if arr.size != 4:
        raise ValueError("Expected a length-4 vector.")
    return np.array([[arr[0], arr[1]], [arr[2], arr[3]]], dtype=float)


def make_setpoint_decomposition_figure(
    v_hat: np.ndarray,
    y_star: np.ndarray,
    *,
    setpoint: np.ndarray,
    delta: float,
    title: str = "Setpoint decomposition",
    subtitle: str | None = None,
    state_order: Sequence[str] = STATE_ORDER,
):
    assert_state_order(state_order)
    v = np.asarray(v_hat, dtype=float).ravel()
    if v.size != 4:
        raise ValueError("v_hat must have length 4.")
    y = np.asarray(y_star, dtype=float).ravel()
    if y.size != 4:
        raise ValueError("y_star must have length 4.")
    p = np.asarray(setpoint, dtype=float).ravel()
    if p.size != 4:
        raise ValueError("setpoint must have length 4.")

    residuals = np.abs(v - p)
    weights = sfxi_math.weights_from_setpoint(p, eps=1e-12)
    if np.any(weights):
        y_lin = sfxi_math.recover_linear_intensity(y, delta=delta)
        contrib = weights * y_lin
        note = None
    else:
        contrib = np.zeros_like(residuals)
        note = "all-OFF setpoint ⇒ intensity ignored"

    ensure_mpl_config_dir()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    ax_res, ax_contrib = axes

    res_grid = _to_grid(residuals)
    contrib_grid = _to_grid(contrib)

    im0 = ax_res.imshow(res_grid, cmap="viridis")
    ax_res.set_title("Per-state residual |v_hat - p|")
    fig.colorbar(im0, ax=ax_res, fraction=0.046, pad=0.04)

    im1 = ax_contrib.imshow(contrib_grid, cmap="magma")
    ax_contrib.set_title("Per-state intensity contribution")
    fig.colorbar(im1, ax=ax_contrib, fraction=0.046, pad=0.04)

    if note:
        ax_contrib.text(
            0.5,
            -0.15,
            note,
            ha="center",
            va="top",
            transform=ax_contrib.transAxes,
            fontsize=9,
        )

    if subtitle:
        fig.suptitle(f"{title}\n{subtitle}")
    else:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
