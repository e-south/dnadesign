"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/objectives/sfxi_v1.py

SFXI objective (setpoint fidelity x intensity) for 8-vector targets.

- Splits Ŷ into v_hat[0:4] (logic ∈ [0,1]^4) and y_star[4:8] (log2 intensity).
- Computes logic fidelity vs setpoint with D(p) normalization.
- Recovers linear intensities with delta and computes E_raw via setpoint weights.
- Denominator is computed from TrainView labels and stored into RoundCtx as:
    objective/sfxi_v1/denom_percentile
    objective/sfxi_v1/denom_value
- Final score = (F_logic^beta) * (E_scaled^gamma).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..registries.objectives import register_objective
from ..round_context import PluginCtx, roundctx_contract


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


def _worst_corner_distance(p: np.ndarray) -> float:
    a = np.maximum(p * p, (1.0 - p) * (1.0 - p))
    return float(np.sqrt(np.sum(a)))


def _logic_fidelity(v_hat: np.ndarray, p: np.ndarray) -> np.ndarray:
    D = _worst_corner_distance(p)
    if not np.isfinite(D) or D <= 0.0:
        return np.ones(v_hat.shape[0], dtype=float)
    dist = np.linalg.norm(v_hat - p[None, :], axis=1)
    out = 1.0 - (dist / D)
    return np.clip(out, 0.0, 1.0)


def _weights_from_setpoint(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = float(np.sum(p))
    if not np.isfinite(P) or P <= eps:
        return np.zeros_like(p)
    return p / P


def _recover_linear_intensity(y_star: np.ndarray, delta: float) -> np.ndarray:
    return np.maximum(0.0, np.power(2.0, y_star) - float(delta))


def _compute_train_effect_pool(
    train_view, setpoint: np.ndarray, delta: float, *, min_n: int
) -> Sequence[float]:
    """
    Prefer *current round* labels if available; otherwise fall back to all labels ≤ as_of_round.
    """
    w = _weights_from_setpoint(setpoint)

    def _dot_effect(y):
        y = np.asarray(y, dtype=float).ravel()
        if y.size < 8:
            return None
        ylin = _recover_linear_intensity(y[4:8], delta)
        return float(np.dot(w, ylin))

    # Try current-round-only (duck-type)
    pool_cur: list[float] = []
    if hasattr(train_view, "iter_labels_y_current_round"):
        for y in train_view.iter_labels_y_current_round():
            v = _dot_effect(y)
            if v is not None:
                pool_cur.append(v)
        if len(pool_cur) >= int(min_n):
            return pool_cur

    # Fallback: all labels
    pool_all: list[float] = []
    for y in train_view.iter_labels_y():
        v = _dot_effect(y)
        if v is not None:
            pool_all.append(v)
    return pool_all


def _resolve_denom_from_pool(
    pool: Sequence[float], *, p: int, fallback_p: int, min_n: int, eps: float
) -> float:
    arr = np.asarray(pool, dtype=float)
    if arr.size >= int(min_n):
        v = float(np.percentile(arr, int(p)))
    elif arr.size > 0:
        v = float(np.percentile(arr, int(fallback_p)))
    else:
        v = float("nan")
    if not np.isfinite(v) or v <= 0.0:
        v = float(eps)
    return v


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
    # assert y_pred dims
    if not (
        isinstance(y_pred, np.ndarray) and y_pred.ndim == 2 and y_pred.shape[1] >= 8
    ):
        raise ValueError(
            f"[sfxi_v1] Expected y_pred shape (n, 8+); got {getattr(y_pred, 'shape', None)}."
        )

    v_hat = np.clip(y_pred[:, 0:4].astype(float), 0.0, 1.0)
    y_star = y_pred[:, 4:8].astype(float)

    setpoint = np.asarray(
        params.get("setpoint_vector", [0, 0, 0, 1]), dtype=float
    ).ravel()
    if setpoint.size != 4:
        raise ValueError("[sfxi_v1] setpoint_vector must have length 4.")
    setpoint = np.clip(setpoint, 0.0, 1.0)

    beta = float(params.get("logic_exponent_beta", 1.0))
    gamma = float(params.get("intensity_exponent_gamma", 1.0))
    delta = float(params.get("intensity_log2_offset_delta", 0.0))
    scaling_cfg = dict(params.get("scaling", {}) or {})
    p = int(scaling_cfg.get("percentile", 95))
    fallback_p = int(scaling_cfg.get("fallback_p", 75))
    min_n = int(scaling_cfg.get("min_n", 5))
    eps = float(scaling_cfg.get("eps", 1e-8))

    # ---- compute denom from training labels (TrainView) ----
    if train_view is None:
        raise ValueError("[sfxi_v1] train_view is required")
    effect_pool = _compute_train_effect_pool(
        train_view, setpoint=setpoint, delta=delta, min_n=min_n
    )
    denom = _resolve_denom_from_pool(
        effect_pool, p=p, fallback_p=fallback_p, min_n=min_n, eps=eps
    )

    # persist into RoundCtx (strict: must be declared in produces)
    if ctx is None:
        raise ValueError("[sfxi_v1] ctx (PluginCtx) is required")
    ctx.set("objective/<self>/denom_percentile", int(p))
    ctx.set("objective/<self>/denom_value", float(denom))

    # ---- score candidates ----
    y_lin = _recover_linear_intensity(y_star, delta=delta)
    w = _weights_from_setpoint(setpoint)
    E_raw = (y_lin * w[None, :]).sum(axis=1)
    E_scaled = np.clip(E_raw / float(denom), 0.0, 1.0)
    F_logic = _logic_fidelity(v_hat, setpoint)
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
    }

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
        scalar_uncertainty=None,
        diagnostics=diagnostics,
    )
