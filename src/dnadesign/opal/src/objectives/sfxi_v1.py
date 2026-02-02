"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/objectives/sfxi_v1.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..core.round_context import PluginCtx, roundctx_contract
from ..registries.objectives import register_objective
from .sfxi_math import (
    STATE_ORDER,
    denom_from_pool,
    effect_raw_from_y_star,
    effect_scaled,
    logic_fidelity,
    parse_setpoint_vector,
    weights_from_setpoint,
)


class ObjectiveResult:
    def __init__(
        self,
        *,
        score: np.ndarray,
        scalar_uncertainty: Optional[np.ndarray],
        diagnostics: Dict[str, Any],
    ) -> None:
        self.score = score
        self.scalar_uncertainty = scalar_uncertainty
        self.diagnostics = diagnostics


def _compute_train_effect_pool(train_view, setpoint: np.ndarray, delta: float, *, min_n: int) -> Sequence[float]:
    """
    Use *current round* labels only for round-calibrated scaling.
    """

    def _dot_effect(y):
        y = np.asarray(y, dtype=float).ravel()
        if y.size < 8:
            return None
        raw, _ = effect_raw_from_y_star(
            y[4:8],
            setpoint,
            delta=delta,
            eps=1e-12,
            state_order=STATE_ORDER,
        )
        return float(raw[0])

    if not hasattr(train_view, "iter_labels_y_current_round"):
        raise ValueError("[sfxi_v1] train_view must expose iter_labels_y_current_round() for round-calibrated scaling.")

    pool_cur: list[float] = []
    for y in train_view.iter_labels_y_current_round():
        v = _dot_effect(y)
        if v is not None:
            pool_cur.append(v)

    if len(pool_cur) < int(min_n):
        raise ValueError(
            f"[sfxi_v1] Need at least min_n={int(min_n)} labels in current round to scale intensity; "
            f"got {len(pool_cur)}. Add labels or lower scaling.min_n."
        )
    return pool_cur


def _parse_scaling_cfg(raw: Any) -> Tuple[int, int, float]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("[sfxi_v1] scaling must be a mapping.")
    allowed = {"percentile", "min_n", "eps"}
    extra = sorted(set(raw.keys()) - allowed)
    if extra:
        raise ValueError(f"[sfxi_v1] Unknown scaling keys: {extra}. Allowed: {sorted(allowed)}")

    p = int(raw.get("percentile", 95))
    min_n = int(raw.get("min_n", 5))
    eps = float(raw.get("eps", 1e-8))

    if not (1 <= p <= 100):
        raise ValueError(f"[sfxi_v1] scaling.percentile must be in [1, 100]; got {p}.")
    if min_n <= 0:
        raise ValueError(f"[sfxi_v1] scaling.min_n must be >= 1; got {min_n}.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"[sfxi_v1] scaling.eps must be positive and finite; got {eps}.")
    return p, min_n, eps


@roundctx_contract(
    category="objective",
    requires=["core/labels_as_of_round"],
    produces=[
        "objective/<self>/denom_percentile",
        "objective/<self>/denom_value",
    ],
)
@register_objective("sfxi_v1")
def sfxi_v1(
    *,
    y_pred: np.ndarray,
    params: Dict[str, Any],
    ctx: Optional[PluginCtx] = None,
    train_view=None,
) -> ObjectiveResult:
    if y_pred.ndim == 3:
        y_pred, std_devs = y_pred
        var = np.exp2(std_devs)
    else:
        var = None
    # assert y_pred dims
    if not (isinstance(y_pred, np.ndarray) and y_pred.ndim == 2 and y_pred.shape[1] >= 8):
        raise ValueError(f"[sfxi_v1] Expected y_pred shape (n, 8+); got {getattr(y_pred, 'shape', None)}.")

    v_hat = np.clip(y_pred[:, 0:4].astype(float), 0.0, 1.0)
    y_star = y_pred[:, 4:8].astype(float)

    setpoint = parse_setpoint_vector(params)
    sum_setpoint = float(np.sum(setpoint))
    intensity_disabled = bool(not np.isfinite(sum_setpoint) or sum_setpoint <= 1e-12)
    w = weights_from_setpoint(setpoint)

    beta = float(params.get("logic_exponent_beta", 1.0))
    gamma = float(params.get("intensity_exponent_gamma", 1.0))
    delta = float(params.get("intensity_log2_offset_delta", 0.0))
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError(f"[sfxi_v1] logic_exponent_beta must be >= 0; got {beta}.")
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError(f"[sfxi_v1] intensity_exponent_gamma must be >= 0; got {gamma}.")
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError(f"[sfxi_v1] intensity_log2_offset_delta must be >= 0; got {delta}.")

    scaling_cfg = dict(params.get("scaling", {}) or {})
    p, min_n, eps = _parse_scaling_cfg(scaling_cfg)

    # ---- compute denom from training labels (TrainView) ----
    if intensity_disabled:
        effect_pool: Sequence[float] = []
        denom = 1.0
    else:
        if train_view is None:
            raise ValueError("[sfxi_v1] train_view is required")
        effect_pool = _compute_train_effect_pool(train_view, setpoint=setpoint, delta=delta, min_n=min_n)
        denom = denom_from_pool(effect_pool, percentile=p, min_n=min_n, eps=eps)

    # persist into RoundCtx (strict: must be declared in produces)
    if ctx is None:
        raise ValueError("[sfxi_v1] ctx (PluginCtx) is required")
    ctx.set("objective/<self>/denom_percentile", int(p))
    ctx.set("objective/<self>/denom_value", float(denom))

    # ---- score candidates ----
    F_logic = logic_fidelity(v_hat, setpoint)
    if intensity_disabled:
        E_raw = np.zeros(v_hat.shape[0], dtype=float)
        E_scaled = np.ones(v_hat.shape[0], dtype=float)
        score = np.power(F_logic, beta)
    else:
        E_raw, _ = effect_raw_from_y_star(
            y_star,
            setpoint,
            delta=delta,
            eps=1e-12,
            state_order=STATE_ORDER,
        )
        E_scaled = effect_scaled(E_raw, float(denom))
        score = np.power(F_logic, beta) * np.power(E_scaled, gamma)

    diagnostics: Dict[str, Any] = {
        "logic_fidelity": F_logic,
        "effect_raw": E_raw,
        "effect_scaled": E_scaled,
        "denom_used": float(denom),
        "clip_lo_mask": E_scaled <= 0.0 + 1e-12,
        "clip_hi_mask": E_scaled >= 1.0 - 1e-12,
        "weights": w,
        "setpoint": setpoint,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "scaling_cfg": dict(scaling_cfg),
        "train_effect_pool_size": int(len(effect_pool)),
        "intensity_disabled": bool(intensity_disabled),
        "all_off_setpoint": bool(intensity_disabled),
    }

    # ---- creating scalar uncertainty ----
    if var is not None:
        # Starting with effect intensity (because it is a lot easier)
        y_lin = (np.exp2(y_pred[:, 4:8])*np.log(2))**2
        ind_var = np.multiply(y_lin, var[:, 4:8])
        wt_var = np.multiply(ind_var, w)
        effect_var = np.sum(wt_var, axis = 1)
        effect_exp = np.sum(np.multiply(w, np.exp2(y_pred[:, 4:8])), axis=1)
        # Now on to logic fidelity
        c = 1/(_worst_corner_distance(setpoint)**2)
        ph2 = 4*(y_pred[:,0:4]**2)*var[:,0:4] + 4*(setpoint**2)*var[:,0:4] \
                + 2*(var[:,0:4]**2) - 8*y_pred[:,0:4]*setpoint*var[:,0:4]
        lf_var = c * np.sum(ph2, axis=1)
        lf_exp = c * np.sum((y_pred[:,0:4]**2 + var[:,0:4] - (2*y_pred[:,0:4] * setpoint ) + setpoint**2), axis=1)
        # Combine into scalar uncertainty
        scalar_uncertainty = effect_var * lf_var + effect_var * lf_exp**2 + lf_var * effect_exp**2
    else:
        scalar_uncertainty = None

    # Emit optional, named summary stats so the runner can log them generically
    try:
        clip_hi_frac = float(np.mean(diagnostics["clip_hi_mask"])) if "clip_hi_mask" in diagnostics else 0.0  # type: ignore[index]
        clip_lo_frac = float(np.mean(diagnostics["clip_lo_mask"])) if "clip_lo_mask" in diagnostics else 0.0  # type: ignore[index]
    except Exception:
        clip_hi_frac, clip_lo_frac = 0.0, 0.0
    diagnostics["summary_stats"] = {
        "score_min": float(np.nanmin(score)),
        "score_median": float(np.nanmedian(score)),
        "score_max": float(np.nanmax(score)),
        "clip_hi_fraction": clip_hi_frac,
        "clip_lo_fraction": clip_lo_frac,
        "train_effect_pool_size": int(len(effect_pool)),
        "denom_used": float(denom),
    }

    return ObjectiveResult(
        score=np.asarray(score, dtype=float).ravel(),
        scalar_uncertainty=scalar_uncertainty,
        diagnostics=diagnostics,
    )
