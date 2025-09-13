"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/objectives/sfxi_v1.py

Setpoint Fidelity x Intensity (SFXI) — vec8 aware objective.

Inputs:
  - y_pred: (N,8) [[v00,v10,v01,v11, y00*,y10*,y01*,y11*]]
  - params: dict with keys:
      setpoint_vector: list[float] length 4
      logic_exponent_beta: float (default 1.0)
      intensity_exponent_gamma: float (default 1.0)
      intensity_log2_offset_delta: float (default 0.0)
      scaling: {percentile:int, min_n:int, fallback_percentile:int, epsilon:float}
  - round_ctx: RoundContext (provides effect_pool_for_scaling, setpoint, percentile_cfg)

Returns:
  ObjectiveResult(score: np.ndarray, diagnostics: dict)

Diagnostics contain:
  - logic_fidelity: (N,)
  - effect_scaled: (N,)
  - denom_used: float
  - vhat: (N,4)
  - yhat_linear: (N,4)
  - flags: list[str] per row (optional; computed upstream if desired)

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple

import numpy as np

from ..registries.objectives import register_objective
from ..round_context import RoundContext
from ..utils import OpalError


class ObjectiveResult(NamedTuple):
    score: np.ndarray
    diagnostics: dict


def _logic_fidelity_l2_norm01(v: np.ndarray, s: np.ndarray) -> np.ndarray:
    # v in [0,1]^4, s in [0,1]^4
    dmax = np.maximum(s, 1.0 - s)
    D = float(np.sqrt((dmax**2).sum()))
    if D == 0.0:
        return np.ones(v.shape[0], dtype=float)
    d = np.linalg.norm(v - s.reshape(1, -1), axis=1) / D
    return np.clip(1.0 - d, 0.0, 1.0)


def _effect_scale_per_round(
    e_pred: np.ndarray,
    e_labels_pool: np.ndarray,
    p: int,
    min_n: int,
    fallback_p: int,
    eps: float,
) -> tuple[np.ndarray, float]:
    pool = e_labels_pool[np.isfinite(e_labels_pool)]
    if pool.size >= min_n:
        denom = float(np.percentile(pool, p))
    else:
        p_main = np.percentile(pool, p) if pool.size else 0.0
        p_fb = np.percentile(pool, fallback_p) if pool.size else 0.0
        denom = float(max(p_main, p_fb, eps))
    scaled = np.clip(e_pred / max(denom, eps), 0.0, 1.0)
    return scaled, denom


@register_objective("sfxi_v1")
def objective_sfxi_v1(
    *,
    y_pred: np.ndarray,  # shape (n,8)
    params: Dict[str, Any],
    round_ctx: RoundContext,
) -> ObjectiveResult:
    if y_pred.ndim != 2 or y_pred.shape[1] != 8:
        raise OpalError(
            f"sfxi_v1 expects predictions of shape (n,8), got {y_pred.shape}"
        )

    v = y_pred[:, :4]
    ystar = y_pred[:, 4:8]

    s = np.asarray(params.get("setpoint_vector", round_ctx.setpoint), dtype=float)
    beta = float(params.get("logic_exponent_beta", 1.0))
    gamma = float(params.get("intensity_exponent_gamma", 1.0))
    delta = float(params.get("intensity_log2_offset_delta", 0.0))

    # fidelity [0,1]
    F = _logic_fidelity_l2_norm01(v, s)

    # invert log2 to (non-negative) linear space per state, then setpoint-weighted sum
    y_lin = np.maximum(0.0, np.power(2.0, ystar) - delta)
    P = float(np.sum(s))
    w = np.zeros_like(s) if P <= 0 else (s / P)
    E_raw = (y_lin * w.reshape(1, -1)).sum(axis=1)

    # scale by per-round percentile over labeled pool
    sc = round_ctx.percentile_cfg
    E_scaled, denom = _effect_scale_per_round(
        E_raw,
        np.asarray(round_ctx.effect_pool_for_scaling, dtype=float),
        int(sc.get("p", sc.get("percentile", 95))),
        int(sc.get("min_n", 5)),
        int(sc.get("fallback_p", 75)),
        float(sc.get("eps", 1e-8)),
    )

    score = (np.power(F, beta)) * (np.power(E_scaled, gamma))

    diag = {
        "logic_fidelity": F,
        "effect_scaled": E_scaled,
        "denom_used": float(denom),
        "vhat": v,
        "yhat_linear": y_lin,
    }
    return ObjectiveResult(score=score, diagnostics=diag)
