"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/sfxi/intensity_scaling.py

Intensity scaling diagnostics for SFXI (denom, clipping, distributions).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...objectives import sfxi_math
from .state_order import STATE_ORDER, assert_state_order


@dataclass(frozen=True)
class IntensityScalingSummary:
    denom: float
    effect_raw: np.ndarray
    effect_scaled: np.ndarray
    clip_lo_fraction: float
    clip_hi_fraction: float
    intensity_disabled: bool


def summarize_intensity_scaling(
    y_star_labels: np.ndarray,
    setpoint: np.ndarray,
    *,
    delta: float,
    percentile: int,
    min_n: int,
    eps: float,
    state_order: Sequence[str] = STATE_ORDER,
) -> IntensityScalingSummary:
    assert_state_order(state_order)
    y_star = np.asarray(y_star_labels, dtype=float)
    if y_star.ndim == 1:
        y_star = y_star.reshape(1, -1)
    if y_star.shape[1] != 4:
        raise ValueError("y_star_labels must have shape (n, 4).")
    if not np.all(np.isfinite(y_star)):
        raise ValueError("y_star_labels must be finite.")

    p = np.asarray(setpoint, dtype=float).ravel()
    if p.size != 4:
        raise ValueError("setpoint must have length 4.")

    w = sfxi_math.weights_from_setpoint(p, eps=1e-12)
    intensity_disabled = not np.any(w)
    if intensity_disabled:
        effect_raw = np.zeros(y_star.shape[0], dtype=float)
        effect_scaled = np.ones(y_star.shape[0], dtype=float)
        return IntensityScalingSummary(
            denom=1.0,
            effect_raw=effect_raw,
            effect_scaled=effect_scaled,
            clip_lo_fraction=0.0,
            clip_hi_fraction=1.0 if effect_scaled.size else 0.0,
            intensity_disabled=True,
        )

    y_lin = sfxi_math.recover_linear_intensity(y_star, delta=delta)
    effect_raw = sfxi_math.effect_raw(y_lin, w)
    denom = sfxi_math.denom_from_pool(effect_raw, percentile=percentile, min_n=min_n, eps=eps)
    effect_scaled = sfxi_math.effect_scaled(effect_raw, denom)
    clip_lo_fraction = float(np.mean(effect_scaled <= 0.0 + 1e-12)) if effect_scaled.size else 0.0
    clip_hi_fraction = float(np.mean(effect_scaled >= 1.0 - 1e-12)) if effect_scaled.size else 0.0
    return IntensityScalingSummary(
        denom=float(denom),
        effect_raw=effect_raw,
        effect_scaled=effect_scaled,
        clip_lo_fraction=clip_lo_fraction,
        clip_hi_fraction=clip_hi_fraction,
        intensity_disabled=False,
    )
