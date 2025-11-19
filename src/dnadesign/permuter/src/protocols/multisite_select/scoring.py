"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/multisite_select/scoring.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _mad(a: np.ndarray) -> float:
    med = np.nanmedian(a)
    return float(np.nanmedian(np.abs(a - med)))


def robust_z(
    x: pd.Series | np.ndarray,
    *,
    gaussian_consistent: bool = False,
    winsor_mads: float | None = None,
) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    med = np.nanmedian(a)
    mad = _mad(a)
    if not np.isfinite(mad) or mad == 0.0:
        # Degenerate distribution (all near-identical) → define 0-centered zeros.
        z = np.zeros_like(a, dtype=float)
    else:
        scale = (1.4826 * mad) if gaussian_consistent else mad
        if scale == 0.0:
            z = np.zeros_like(a, dtype=float)
        else:
            z = (a - med) / scale
    if winsor_mads is not None and np.isfinite(winsor_mads) and winsor_mads > 0:
        lim = float(winsor_mads)
        z = np.clip(z, -lim, +lim, out=z)
    return z


def detect_k_drift(
    x: pd.Series,
    k: pd.Series,
    *,
    gaussian_consistent: bool = False,
) -> Tuple[bool, dict]:
    """
    Cheap, robust diagnostic for per‑k drift.
    Enable when the spread of per‑k medians exceeds 0.5 × global robust scale.
    """
    a = np.asarray(x, dtype=float)
    # work on raw medians; compare spread to global robust scale
    medians = (
        pd.DataFrame({"k": k.values, "x": a})
        .dropna()
        .groupby("k")["x"]
        .median()
        .sort_index()
    )
    if medians.empty:
        return False, {"reason": "no_k_groups"}
    spread = float(medians.max() - medians.min())
    mad = _mad(a)
    scale = (1.4826 * mad) if gaussian_consistent else mad
    if np.isfinite(scale) and scale > 0:
        trigger = spread > 0.5 * scale
    else:
        # Degenerate → conservative fallback
        trigger = spread > 0.5
    info = {
        "k_levels": int(medians.shape[0]),
        "median_min": float(medians.min()),
        "median_max": float(medians.max()),
        "span": spread,
        "global_robust_scale": float(scale),
    }
    return bool(trigger), info


def z_by_k(
    x: pd.Series,
    k: pd.Series,
    *,
    gaussian_consistent: bool = False,
    winsor_mads: float | None = None,
) -> np.ndarray:
    df = pd.DataFrame({"x": np.asarray(x, float), "k": k.values})
    z = np.empty(len(df), dtype=float)
    for kk, sub in df.groupby("k", sort=False):
        idx = sub.index.to_numpy()
        z[idx] = robust_z(
            sub["x"], gaussian_consistent=gaussian_consistent, winsor_mads=winsor_mads
        )
    return z
