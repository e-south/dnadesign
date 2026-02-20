"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/objectives/sfxi_v1.py

Module Author(s): Eric J. South, Elm Markert
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

_MAX_LOG2_FOR_SCORE = float(np.log2(np.finfo(float).max)) - 1.0
_MAX_LOG2_FOR_UNCERTAINTY = float(np.log2(np.sqrt(np.finfo(float).max) / np.log(2.0))) - 1.0


def _validate_intensity_log2_range(y_star: np.ndarray, *, upper: float, context: str) -> None:
    vals = np.asarray(y_star, dtype=float)
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    if np.any(vals > float(upper)):
        observed = float(np.max(vals))
        raise ValueError(
            f"sfxi_v1: y_pred intensity log2 values exceed stable {context} range "
            f"(max allowed {float(upper):.1f}, observed {observed:.1f})."
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
        raise ValueError("sfxi_v1: train_view must expose iter_labels_y_current_round() for round-calibrated scaling.")

    pool_cur: list[float] = []
    for y in train_view.iter_labels_y_current_round():
        v = _dot_effect(y)
        if v is not None:
            pool_cur.append(v)

    if len(pool_cur) < int(min_n):
        raise ValueError(
            f"sfxi_v1: need at least min_n={int(min_n)} labels in current round to scale intensity; "
            f"got {len(pool_cur)}. Add labels or lower scaling.min_n."
        )
    return pool_cur


def _parse_scaling_cfg(raw: Any) -> Tuple[int, int, float]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("sfxi_v1: scaling must be a mapping.")
    allowed = {"percentile", "min_n", "eps"}
    extra = sorted(set(raw.keys()) - allowed)
    if extra:
        raise ValueError(f"sfxi_v1: unknown scaling keys: {extra}. Allowed: {sorted(allowed)}")

    p = int(raw.get("percentile", 95))
    min_n = int(raw.get("min_n", 5))
    eps = float(raw.get("eps", 1e-8))

    if not (1 <= p <= 100):
        raise ValueError(f"sfxi_v1: scaling.percentile must be in [1, 100]; got {p}.")
    if min_n <= 0:
        raise ValueError(f"sfxi_v1: scaling.min_n must be >= 1; got {min_n}.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"sfxi_v1: scaling.eps must be positive and finite; got {eps}.")
    return p, min_n, eps


def _resolve_uncertainty_method(raw: Any, *, beta: float, gamma: float) -> str:
    method = None if raw is None else str(raw).strip().lower()
    if method is None:
        method = "analytical" if (beta == 1.0 and gamma == 1.0) else "delta"
    if method not in {"delta", "analytical"}:
        raise ValueError("sfxi_v1: uncertainty_method must be 'delta' or 'analytical'.")
    if method == "analytical" and not (beta == 1.0 and gamma == 1.0):
        raise ValueError(
            "sfxi_v1: analytical uncertainty requires logic_exponent_beta == 1 and intensity_exponent_gamma == 1."
        )
    return method


def _sample_index_pairs(mask: np.ndarray, *, max_items: int = 5) -> list[tuple[int, int]]:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    return [(int(r), int(c)) for r, c in coords[:max_items]]


def _validate_required_y_pred_std_strictly_positive(
    *,
    y_pred_std: np.ndarray,
    beta: float,
    gamma: float,
    intensity_disabled: bool,
    w: np.ndarray,
) -> None:
    required = np.zeros_like(y_pred_std, dtype=bool)
    if beta > 0.0:
        required[:, 0:4] = True
    if (not intensity_disabled) and gamma > 0.0:
        weighted_states = np.asarray(w, dtype=float) > 0.0
        required[:, 4:8] = weighted_states[None, :]

    violations = required & (np.asarray(y_pred_std, dtype=float) <= 0.0)
    if not np.any(violations):
        return

    count = int(np.sum(violations))
    sample = _sample_index_pairs(violations, max_items=5)
    min_value = float(np.min(np.asarray(y_pred_std, dtype=float)[violations]))
    raise ValueError(
        "sfxi_v1: y_pred_std entries that affect scalar uncertainty must be > 0; "
        f"found {count} violations; sample_indices={sample}; "
        f"min_offending_value={min_value:.6g}."
    )


def _validate_scalar_uncertainty_strictly_positive(*, scalar_uncertainty: np.ndarray, context: str) -> None:
    vals = np.asarray(scalar_uncertainty, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"sfxi_v1: {context} scalar uncertainty contains non-finite values.")
    bad = vals <= 0.0
    if not np.any(bad):
        return
    count = int(np.sum(bad))
    sample_rows = [int(i) for i in np.where(bad)[0][:5].tolist()]
    min_value = float(np.min(vals[bad]))
    raise ValueError(
        f"sfxi_v1: {context} scalar uncertainty must be > 0 for all candidates; "
        f"found {count} violations; sample_rows={sample_rows}; min_value={min_value:.6g}."
    )


def _power_derivative_prefactor(*, base: np.ndarray, exponent: float, exponent_name: str) -> np.ndarray:
    base_arr = np.asarray(base, dtype=float)
    exp = float(exponent)
    if exp == 0.0:
        return np.zeros_like(base_arr, dtype=float)
    if exp == 1.0:
        return np.ones_like(base_arr, dtype=float)
    if exp > 1.0:
        return exp * np.power(base_arr, exp - 1.0)
    if exp > 0.0:
        bad = base_arr <= 0.0
        if np.any(bad):
            sample_rows = [int(i) for i in np.where(bad)[0][:5].tolist()]
            min_value = float(np.min(base_arr[bad]))
            count = int(np.sum(bad))
            raise ValueError(
                f"sfxi_v1: {exponent_name} in (0, 1) requires positive base for "
                "delta-method derivative; "
                f"found {count} violations; sample_rows={sample_rows}; min_value={min_value:.6g}."
            )
        return exp * np.power(base_arr, exp - 1.0)
    raise ValueError(f"sfxi_v1: {exponent_name} must be >= 0; got {exp}.")


def _scalar_uncertainty_delta(
    *,
    y_pred: np.ndarray,
    y_pred_var: np.ndarray,
    v_hat: np.ndarray,
    y_star: np.ndarray,
    setpoint: np.ndarray,
    w: np.ndarray,
    beta: float,
    gamma: float,
    delta: float,
    denom: float,
    F_logic: np.ndarray,
    E_scaled: np.ndarray,
    intensity_disabled: bool,
) -> np.ndarray:
    n_rows = int(y_pred.shape[0])
    grad = np.zeros((n_rows, y_pred.shape[1]), dtype=float)

    dist = np.linalg.norm(v_hat - setpoint[None, :], axis=1)
    dist_safe = np.maximum(dist, 1e-12)
    D = float(worst_corner_distance(setpoint))
    dF_dv = np.zeros_like(v_hat, dtype=float)
    if np.isfinite(D) and D > 0.0:
        active_logic = dist > 1e-12
        dF_dv[active_logic, :] = -(v_hat[active_logic, :] - setpoint[None, :]) / (D * dist_safe[active_logic, None])

    logic_prefactor = _power_derivative_prefactor(
        base=F_logic,
        exponent=beta,
        exponent_name="logic_exponent_beta",
    )
    if intensity_disabled:
        logic_scale = logic_prefactor
    else:
        logic_scale = logic_prefactor * np.power(E_scaled, gamma)
    grad[:, 0:4] = logic_scale[:, None] * dF_dv

    if not intensity_disabled:
        y_lin = np.power(2.0, y_star) - delta
        positive_intensity = y_lin > 0.0
        dEraw_dy = np.log(2.0) * np.power(2.0, y_star) * positive_intensity * w[None, :]
        dEscaled_dy = dEraw_dy / float(denom)
        intensity_prefactor = _power_derivative_prefactor(
            base=E_scaled,
            exponent=gamma,
            exponent_name="intensity_exponent_gamma",
        )
        grad[:, 4:8] = (np.power(F_logic, beta) * intensity_prefactor)[:, None] * dEscaled_dy

    scalar_uncertainty_var = np.sum((grad**2) * y_pred_var, axis=1)
    if not np.all(np.isfinite(scalar_uncertainty_var)):
        raise ValueError("sfxi_v1: computed scalar uncertainty variance contains non-finite values.")
    with np.errstate(invalid="ignore"):
        scalar_uncertainty = np.sqrt(scalar_uncertainty_var)
    _validate_scalar_uncertainty_strictly_positive(
        scalar_uncertainty=scalar_uncertainty,
        context="delta",
    )
    return scalar_uncertainty


def _scalar_uncertainty_analytical(
    *,
    y_pred: np.ndarray,
    y_pred_var: np.ndarray,
    v_hat: np.ndarray,
    w: np.ndarray,
    denom: float,
    setpoint: np.ndarray,
    delta: float,
    intensity_disabled: bool,
) -> np.ndarray:
    # Analytical uncertainty formula updated to reflect lognormal distribution usage
    logic_mean = np.asarray(v_hat, dtype=float)
    logic_var = np.asarray(y_pred_var[:, 0:4], dtype=float)
    if intensity_disabled:
        effect_var = np.zeros(y_pred.shape[0], dtype=float)
        effect_exp = np.ones(y_pred.shape[0], dtype=float)
    else:
        effect_var_unwt = (np.exp2(y_pred_var[:, 4:8]) - 1) * np.exp2(
            y_pred_var[:, 4:8] + 2 * y_pred[:, 4:8]
        )
        effect_var_unsc = np.sum(np.multiply(effect_var_unwt, w**2), axis=1)
        effect_var = effect_var_unsc / (denom**2)
        effect_exp = (
            np.sum(
                np.multiply(
                    np.exp2(y_pred[:, 4:8] + y_pred_var[:, 4:8] / 2),
                    w,
                ),
                axis=1,
            )
            / denom
        )

    D = float(worst_corner_distance(setpoint))
    if not np.isfinite(D) or D <= 0.0:
        raise ValueError("sfxi_v1: invalid setpoint distance for analytical uncertainty.")
    c = 1.0 / (D**2)
    lf_var_unsc = (
        4.0 * (logic_mean**2) * logic_var
        + 4.0 * (setpoint[None, :] ** 2) * logic_var
        + 2.0 * (logic_var**2)
        - 8.0 * logic_mean * setpoint[None, :] * logic_var
    )
    lf_var = c * np.sum(lf_var_unsc, axis=1)
    lf_exp = c * np.sum(
        logic_mean**2 + logic_var - (2.0 * logic_mean * setpoint[None, :]) + setpoint[None, :] ** 2,
        axis=1,
    )
    scalar_uncertainty_var = effect_var * lf_var + effect_var * (lf_exp**2) + lf_var * (effect_exp**2)
    if not np.all(np.isfinite(scalar_uncertainty_var)):
        raise ValueError("sfxi_v1: computed scalar uncertainty variance contains non-finite values.")
    with np.errstate(invalid="ignore"):
        scalar_uncertainty = np.sqrt(scalar_uncertainty_var)
    _validate_scalar_uncertainty_strictly_positive(
        scalar_uncertainty=scalar_uncertainty,
        context="analytical",
    )
    return scalar_uncertainty


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
        raise ValueError(f"sfxi_v1: expected y_pred shape (n, 8+); got {getattr(y_pred, 'shape', None)}.")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("sfxi_v1: y_pred must be finite.")
    y_pred_std = np.asarray(y_pred_std, dtype=float) if y_pred_std is not None else None
    if y_pred_std is not None and y_pred_std.shape != y_pred.shape:
        raise ValueError(f"sfxi_v1: y_pred_std shape mismatch: expected {y_pred.shape}, got {y_pred_std.shape}.")
    if y_pred_std is not None and not np.all(np.isfinite(y_pred_std)):
        raise ValueError("sfxi_v1: y_pred_std must be finite.")
    if y_pred_std is not None and np.any(y_pred_std < 0.0):
        raise ValueError("sfxi_v1: y_pred_std must be non-negative.")
    with np.errstate(over="ignore", invalid="ignore"):
        y_pred_var = np.square(y_pred_std) if y_pred_std is not None else None
    if y_pred_var is not None and not np.all(np.isfinite(y_pred_var)):
        raise ValueError("sfxi_v1: variance contains non-finite values.")

    v_hat = np.clip(y_pred[:, 0:4].astype(float), 0.0, 1.0)
    y_star = y_pred[:, 4:8].astype(float)
    _validate_intensity_log2_range(y_star, upper=_MAX_LOG2_FOR_SCORE, context="score")
    if y_pred_var is not None:
        _validate_intensity_log2_range(y_star, upper=_MAX_LOG2_FOR_UNCERTAINTY, context="uncertainty")

    setpoint = parse_setpoint_vector(params)
    sum_setpoint = float(np.sum(setpoint))
    intensity_disabled = bool(not np.isfinite(sum_setpoint) or sum_setpoint <= 1e-12)
    w = weights_from_setpoint(setpoint)

    beta = float(params.get("logic_exponent_beta", 1.0))
    gamma = float(params.get("intensity_exponent_gamma", 1.0))
    delta = float(params.get("intensity_log2_offset_delta", 0.0))
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError(f"sfxi_v1: logic_exponent_beta must be >= 0; got {beta}.")
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError(f"sfxi_v1: intensity_exponent_gamma must be >= 0; got {gamma}.")
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError(f"sfxi_v1: intensity_log2_offset_delta must be >= 0; got {delta}.")
    if y_pred_std is not None:
        _validate_required_y_pred_std_strictly_positive(
            y_pred_std=y_pred_std,
            beta=beta,
            gamma=gamma,
            intensity_disabled=bool(intensity_disabled),
            w=w,
        )

    scaling_cfg = dict(params.get("scaling", {}) or {})
    p, min_n, eps = _parse_scaling_cfg(scaling_cfg)

    # ---- compute denom from training labels (TrainView) ----
    if intensity_disabled:
        effect_pool: Sequence[float] = []
        denom = 1.0
    else:
        if train_view is None:
            raise ValueError("sfxi_v1: train_view is required")
        effect_pool = _compute_train_effect_pool(train_view, setpoint=setpoint, delta=delta, min_n=min_n)
        denom = denom_from_pool(effect_pool, percentile=p, min_n=min_n, eps=eps)

    # persist into RoundCtx (strict: must be declared in produces)
    if ctx is None:
        raise ValueError("sfxi_v1: ctx (PluginCtx) is required")
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
    logic_clip_mask = ((y_pred[:, 0:4] >= 0.0) & (y_pred[:, 0:4] <= 1.0)).astype(float)

    diagnostics: Dict[str, Any] = {
        "logic_fidelity": F_logic,
        "effect_raw": E_raw,
        "effect_scaled": E_scaled,
        "denom_used": float(denom),
        "clip_lo_mask": E_scaled <= 0.0 + 1e-12,
        "clip_hi_mask": E_scaled >= 1.0 - 1e-12,
        "logic_clip_mask": logic_clip_mask,
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

    uncertainty_method: Optional[str] = None

    # ---- scalar uncertainty (selected method over implemented score function) ----
    if y_pred_var is not None:
        uncertainty_method = _resolve_uncertainty_method(params.get("uncertainty_method", None), beta=beta, gamma=gamma)
        if uncertainty_method == "delta":
            scalar_uncertainty = _scalar_uncertainty_delta(
                y_pred=y_pred,
                y_pred_var=y_pred_var,
                v_hat=v_hat,
                y_star=y_star,
                setpoint=setpoint,
                w=w,
                beta=beta,
                gamma=gamma,
                delta=delta,
                denom=float(denom),
                F_logic=F_logic,
                E_scaled=E_scaled,
                intensity_disabled=bool(intensity_disabled),
            )
        else:
            scalar_uncertainty = _scalar_uncertainty_analytical(
                y_pred=y_pred,
                y_pred_var=y_pred_var,
                v_hat=v_hat,
                w=w,
                denom = denom,
                setpoint=setpoint,
                delta=delta,
                intensity_disabled=bool(intensity_disabled),
            )
    else:
        scalar_uncertainty = None
    ctx.set(
        "objective/<self>/uncertainty_by_name",
        {"sfxi": scalar_uncertainty.tolist()} if scalar_uncertainty is not None else {},
    )

    # Emit optional, named summary stats so the runner can log them generically
    clip_hi_frac = float(np.mean(np.asarray(diagnostics["clip_hi_mask"], dtype=bool)))
    clip_lo_frac = float(np.mean(np.asarray(diagnostics["clip_lo_mask"], dtype=bool)))
    summary_stats: Dict[str, Any] = {
        "score_min": float(np.nanmin(score)),
        "score_median": float(np.nanmedian(score)),
        "score_max": float(np.nanmax(score)),
        "clip_hi_fraction": clip_hi_frac,
        "clip_lo_fraction": clip_lo_frac,
        "train_effect_pool_size": int(len(effect_pool)),
        "denom_used": float(denom),
        "denom_percentile": int(p),
    }
    if uncertainty_method is not None:
        summary_stats["uncertainty_method"] = uncertainty_method
    diagnostics["summary_stats"] = summary_stats

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
