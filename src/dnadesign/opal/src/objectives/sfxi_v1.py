"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/objectives/sfxi_v1.py

Module Author(s): Elm Markert, Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..core.objective_result import ObjectiveResultV2
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
    worst_corner_distance,
)


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
        "objective/<self>/uncertainty_by_name",
    ],
)
@register_objective("sfxi_v1")
def sfxi_v1(
    *,
    y_pred: np.ndarray,
    params: Dict[str, Any],
    ctx: Optional[PluginCtx],
    train_view=None,
    y_pred_std=None,
) -> ObjectiveResultV2:
    # assert y_pred dims
    if not (isinstance(y_pred, np.ndarray) and y_pred.ndim == 2 and y_pred.shape[1] >= 8):
        raise ValueError(f"[sfxi_v1] Expected y_pred shape (n, 8+); got {getattr(y_pred, 'shape', None)}.")
    y_pred_std = np.asarray(y_pred_std, dtype=float) if y_pred_std is not None else None
    if y_pred_std is not None and y_pred_std.shape != y_pred.shape:
        raise ValueError(f"[sfxi_v1] y_pred_std shape mismatch: expected {y_pred.shape}, got {y_pred_std.shape}.")
    if y_pred_std is not None and not np.all(np.isfinite(y_pred_std)):
        raise ValueError("[sfxi_v1] y_pred_std must be finite.")
    if y_pred_std is not None and np.any(y_pred_std < 0.0):
        raise ValueError("[sfxi_v1] y_pred_std must be non-negative.")
    y_pred_var = (y_pred_std**2) if y_pred_std is not None else None

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

    uncertainty_method = str(params.get("uncertainty_method", "delta")).strip().lower()
    if uncertainty_method != "delta":
        raise ValueError("[sfxi_v1] uncertainty_method must be 'delta'.")

    # ---- scalar uncertainty (delta method over implemented score function) ----
    if y_pred_var is not None:
        n_rows = int(y_pred.shape[0])
        grad = np.zeros((n_rows, y_pred.shape[1]), dtype=float)

        logic_raw = y_pred[:, 0:4].astype(float)
        logic_clip_mask = ((logic_raw >= 0.0) & (logic_raw <= 1.0)).astype(float)
        dist = np.linalg.norm(v_hat - setpoint[None, :], axis=1)
        dist_safe = np.maximum(dist, 1e-12)
        D = float(worst_corner_distance(setpoint))
        dF_dv = np.zeros_like(v_hat, dtype=float)
        if np.isfinite(D) and D > 0.0:
            active_logic = (dist > 1e-12) & (F_logic > 0.0)
            dF_dv[active_logic, :] = -(v_hat[active_logic, :] - setpoint[None, :]) / (D * dist_safe[active_logic, None])

        logic_prefactor = np.zeros_like(F_logic, dtype=float)
        if beta > 0.0:
            logic_active = F_logic > 0.0
            logic_prefactor[logic_active] = beta * np.power(F_logic[logic_active], beta - 1.0)
        if intensity_disabled:
            logic_scale = logic_prefactor
        else:
            logic_scale = logic_prefactor * np.power(E_scaled, gamma)
        grad[:, 0:4] = logic_scale[:, None] * dF_dv * logic_clip_mask

        if not intensity_disabled:
            y_lin = np.power(2.0, y_star) - delta
            positive_intensity = y_lin > 0.0
            e_unclipped = E_raw / float(denom)
            e_active = (e_unclipped > 0.0) & (e_unclipped < 1.0)
            dEraw_dy = np.log(2.0) * np.power(2.0, y_star) * positive_intensity * w[None, :]
            dEscaled_dy = (dEraw_dy / float(denom)) * e_active[:, None]

            intensity_prefactor = np.zeros_like(E_scaled, dtype=float)
            if gamma > 0.0:
                active = e_active & (E_scaled > 0.0)
                intensity_prefactor[active] = gamma * np.power(E_scaled[active], gamma - 1.0)
            grad[:, 4:8] = (np.power(F_logic, beta) * intensity_prefactor)[:, None] * dEscaled_dy

        scalar_uncertainty_var = np.sum((grad**2) * y_pred_var, axis=1)
        if not np.all(np.isfinite(scalar_uncertainty_var)):
            raise ValueError("[sfxi_v1] computed scalar uncertainty variance contains non-finite values.")
        scalar_uncertainty_var = np.maximum(scalar_uncertainty_var, 0.0)
        scalar_uncertainty = np.sqrt(scalar_uncertainty_var)
        if not np.all(np.isfinite(scalar_uncertainty)):
            raise ValueError("[sfxi_v1] computed scalar uncertainty contains non-finite values.")
    else:
        scalar_uncertainty = None
    ctx.set(
        "objective/<self>/uncertainty_by_name",
        {"sfxi": scalar_uncertainty.tolist()} if scalar_uncertainty is not None else {},
    )

    # Emit optional, named summary stats so the runner can log them generically
    clip_hi_frac = float(np.mean(np.asarray(diagnostics["clip_hi_mask"], dtype=bool)))
    clip_lo_frac = float(np.mean(np.asarray(diagnostics["clip_lo_mask"], dtype=bool)))
    diagnostics["summary_stats"] = {
        "score_min": float(np.nanmin(score)),
        "score_median": float(np.nanmedian(score)),
        "score_max": float(np.nanmax(score)),
        "clip_hi_fraction": clip_hi_frac,
        "clip_lo_fraction": clip_lo_frac,
        "train_effect_pool_size": int(len(effect_pool)),
        "denom_used": float(denom),
        "denom_percentile": int(p),
        "uncertainty_method": uncertainty_method,
    }

    score_arr = np.asarray(score, dtype=float).ravel()
    return ObjectiveResultV2(
        scores_by_name={
            "sfxi": score_arr,
            "logic_fidelity": np.asarray(F_logic, dtype=float).ravel(),
            "effect_scaled": np.asarray(E_scaled, dtype=float).ravel(),
        },
        uncertainty_by_name={"sfxi": np.asarray(scalar_uncertainty, dtype=float).ravel()}
        if scalar_uncertainty is not None
        else {},
        diagnostics=diagnostics,
        modes_by_name={
            "sfxi": "maximize",
            "logic_fidelity": "maximize",
            "effect_scaled": "maximize",
        },
    )


sfxi_v1.__opal_score_channels__ = ("sfxi", "logic_fidelity", "effect_scaled")
sfxi_v1.__opal_uncertainty_channels__ = ("sfxi",)
sfxi_v1.__opal_score_modes__ = {
    "sfxi": "maximize",
    "logic_fidelity": "maximize",
    "effect_scaled": "maximize",
}
