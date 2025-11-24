"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/multisite_select/scoring.py

Robust score normalization and composite scoring for multi-site selection.

Implements:
  • robust median/MAD z-scaling for observed fitness (LLR) and epistasis,
  • Gaussian-consistent MAD scaling (optional),
  • optional symmetric winsorization in MAD units,
  • composite score = alpha · z_llr + β · z_epi.

No k-aware scaling is implemented here by design; the multi-site selector uses
global robust scaling across all mutation counts k.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

_GAUSSIAN_MAD = 1.4826  # factor to make MAD consistent with σ for a normal


def _mad(a: np.ndarray) -> float:
    med = np.nanmedian(a)
    return float(np.nanmedian(np.abs(a - med)))


@dataclass(frozen=True)
class ScalingSummary:
    median_llr: float
    mad_llr: float
    median_epi: float
    mad_epi: float


def robust_z(
    x: pd.Series | np.ndarray,
    *,
    gaussian_consistent: bool,
    winsor_mads: float | None = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Robust z-score using median/MAD.

    Returns (z, median, mad_raw). MAD is the *raw* MAD (before any 1.4826 factor).
    """
    arr = np.asarray(x, dtype=float)
    med = float(np.nanmedian(arr))
    mad = _mad(arr)
    if not np.isfinite(mad) or mad == 0.0:
        z = np.zeros_like(arr, dtype=float)
        return z, med, mad

    scale = mad * (_GAUSSIAN_MAD if gaussian_consistent else 1.0)
    if scale == 0.0 or not np.isfinite(scale):
        z = np.zeros_like(arr, dtype=float)
    else:
        z = (arr - med) / scale

    if winsor_mads is not None and np.isfinite(winsor_mads) and winsor_mads > 0:
        lim = float(winsor_mads)
        z = np.clip(z, -lim, +lim, out=z)

    return z, med, mad


def compute_scaled_scores(
    *,
    llr_obs: pd.Series | np.ndarray,
    delta: pd.Series | np.ndarray,
    gaussian_consistent: bool,
    winsor_mads: float | None,
    w_llr: float,
    w_epi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ScalingSummary]:
    """
    Compute robust z-scores z_llr, z_epi and composite score.

    score(v) = alpha · z_llr(v) + β · z_epi(v)

    Returns
    -------
    z_llr : np.ndarray
    z_epi : np.ndarray
    score : np.ndarray
    summary : ScalingSummary
    """
    z_llr, med_llr, mad_llr = robust_z(
        llr_obs, gaussian_consistent=gaussian_consistent, winsor_mads=winsor_mads
    )
    z_epi, med_epi, mad_epi = robust_z(
        delta, gaussian_consistent=gaussian_consistent, winsor_mads=winsor_mads
    )

    score = np.asarray(w_llr, float) * z_llr + np.asarray(w_epi, float) * z_epi

    summary = ScalingSummary(
        median_llr=float(med_llr),
        mad_llr=float(mad_llr),
        median_epi=float(med_epi),
        mad_epi=float(mad_epi),
    )
    return z_llr, z_epi, score, summary
