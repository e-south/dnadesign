"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/sfxi/uncertainty.py

Model-agnostic uncertainty contract and RF adapter for SFXI diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ...objectives import sfxi_math
from ...runtime.y_ops_inverse import apply_y_ops_inverse
from .ensemble import SupportsEnsemblePredictions


@dataclass(frozen=True)
class UncertaintyContext:
    setpoint: np.ndarray
    beta: float
    gamma: float
    delta: float
    denom: float | None
    y_ops: Sequence[Mapping[str, Any]]
    round_ctx: Any | None


@dataclass(frozen=True)
class UncertaintyResult:
    values: np.ndarray
    kind: str
    statistic: str
    semantics: str
    detail: dict[str, Any]


def supports_uncertainty(*, model: object | None) -> bool:
    if model is None:
        return False
    return isinstance(model, SupportsEnsemblePredictions)


def compute_uncertainty(
    model: SupportsEnsemblePredictions,
    X: np.ndarray,
    *,
    ctx: UncertaintyContext,
    batch_size: int,
) -> UncertaintyResult:
    if not supports_uncertainty(model=model):
        raise ValueError("Model does not support ensemble uncertainty predictions.")
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D numpy array.")
    if X_arr.shape[0] == 0:
        raise ValueError("X must contain at least one row.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    setpoint = np.asarray(ctx.setpoint, dtype=float).ravel()
    if setpoint.size != 4:
        raise ValueError("setpoint must have length 4.")
    w = sfxi_math.weights_from_setpoint(setpoint, eps=1e-12)
    intensity_disabled = not np.any(w)
    denom = ctx.denom
    if not intensity_disabled and (denom is None or not np.isfinite(denom) or denom <= 0.0):
        raise ValueError("denom is required and must be positive to score uncertainty when intensity is enabled.")

    if ctx.y_ops and ctx.round_ctx is None:
        raise ValueError("round_ctx is required to invert y-ops for uncertainty.")

    n_rows = int(X_arr.shape[0])
    mean = np.zeros(n_rows, dtype=float)
    m2 = np.zeros(n_rows, dtype=float)
    n_estimators = 0
    current_estimator = None
    seen: set[int] = set()

    for est_idx, row_start, row_end, y_pred in model.iter_ensemble_predictions(
        X_arr,
        batch_size=batch_size,
    ):
        if row_start < 0 or row_end > n_rows or row_end <= row_start:
            raise ValueError("Invalid row slice from ensemble predictions.")
        if not isinstance(est_idx, int) or est_idx < 0:
            raise ValueError("Estimator index must be a non-negative integer.")
        if current_estimator is None or est_idx != current_estimator:
            if est_idx in seen:
                raise ValueError("Ensemble predictions must be grouped by estimator.")
            seen.add(est_idx)
            current_estimator = est_idx
            n_estimators += 1
        k = float(n_estimators)

        y_hat = np.asarray(y_pred, dtype=float)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)
        if y_hat.shape[0] != (row_end - row_start):
            raise ValueError("Ensemble prediction batch has invalid row count.")
        if ctx.y_ops:
            y_hat = apply_y_ops_inverse(y_ops=ctx.y_ops, y=y_hat, ctx=ctx.round_ctx)
            if y_hat.ndim == 1:
                y_hat = y_hat.reshape(-1, 1)
            if not np.all(np.isfinite(y_hat)):
                raise ValueError("Non-finite values after y-ops inversion in uncertainty.")
        if y_hat.shape[1] < 4:
            raise ValueError("Predictions must have at least 4 outputs for logic.")

        v_hat = np.clip(y_hat[:, 0:4], 0.0, 1.0)
        F_logic = sfxi_math.logic_fidelity(v_hat, setpoint)
        if intensity_disabled:
            score = np.power(F_logic, float(ctx.beta))
        else:
            if y_hat.shape[1] < 8:
                raise ValueError("Predictions missing intensity components (need 8 outputs).")
            y_star = y_hat[:, 4:8]
            E_raw, _ = sfxi_math.effect_raw_from_y_star(
                y_star,
                setpoint,
                delta=float(ctx.delta),
                eps=1e-12,
                state_order=sfxi_math.STATE_ORDER,
            )
            E_scaled = sfxi_math.effect_scaled(E_raw, float(denom))
            score = np.power(F_logic, float(ctx.beta)) * np.power(E_scaled, float(ctx.gamma))
        if not np.all(np.isfinite(score)):
            raise ValueError("Non-finite scores encountered during uncertainty computation.")

        batch_slice = slice(row_start, row_end)
        delta = score - mean[batch_slice]
        mean[batch_slice] = mean[batch_slice] + delta / k
        m2[batch_slice] = m2[batch_slice] + delta * (score - mean[batch_slice])

    if n_estimators <= 0:
        raise ValueError("No ensemble estimators yielded predictions.")

    var = m2 / float(n_estimators)
    if np.any(var < -1e-12):
        raise ValueError("Negative variance encountered in uncertainty calculation.")
    var = np.clip(var, 0.0, None)
    values = np.sqrt(var)
    return UncertaintyResult(
        values=np.asarray(values, dtype=float).ravel(),
        kind="score",
        statistic="std",
        semantics="ensemble_spread",
        detail={
            "n_estimators": int(n_estimators),
            "ddof": 0,
            "batch_size": int(batch_size),
            "intensity_disabled": bool(intensity_disabled),
            "denom_fixed": bool(not intensity_disabled),
        },
    )
