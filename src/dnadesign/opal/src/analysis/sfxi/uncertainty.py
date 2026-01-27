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
from ..dashboard.y_ops import apply_y_ops_inverse


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
    components: str | None
    reduction: str | None
    detail: dict[str, Any]


def supports_uncertainty(*, model: object | None) -> bool:
    if model is None:
        return False
    if hasattr(model, "predict_per_tree"):
        return True
    if hasattr(model, "estimators_"):
        return True
    return False


def _predict_per_tree(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_per_tree"):
        preds = model.predict_per_tree(X)
        if preds is None:
            raise ValueError("predict_per_tree returned None.")
        return np.asarray(preds, dtype=float)
    ests = getattr(model, "estimators_", None)
    if not ests:
        raise ValueError("Model does not expose per-tree estimators.")
    preds = []
    for t in ests:
        y = t.predict(X)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        preds.append(y)
    return np.stack(preds, axis=0)


def _invert_y_ops(
    preds: np.ndarray,
    *,
    y_ops: Sequence[Mapping[str, Any]],
    round_ctx: Any | None,
) -> np.ndarray:
    if not y_ops:
        return preds
    if round_ctx is None:
        raise ValueError("round_ctx is required to invert y-ops for uncertainty.")
    out = []
    for t in range(preds.shape[0]):
        out.append(apply_y_ops_inverse(y_ops=y_ops, y=preds[t], ctx=round_ctx))
    return np.stack(out, axis=0)


def compute_uncertainty(
    model: object,
    X: np.ndarray,
    *,
    kind: str,
    ctx: UncertaintyContext,
    components: str = "all",
    reduction: str = "mean",
) -> UncertaintyResult:
    if not supports_uncertainty(model=model):
        raise ValueError("Model does not support uncertainty (no per-tree predictions).")
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D numpy array.")

    preds = _predict_per_tree(model, X_arr)
    if preds.ndim != 3:
        raise ValueError("Per-tree predictions must have shape (T, N, D).")

    preds = _invert_y_ops(preds, y_ops=ctx.y_ops, round_ctx=ctx.round_ctx)
    if preds.shape[2] < 4:
        raise ValueError("Per-tree predictions must have at least 4 outputs for logic.")

    kind_val = str(kind or "").strip().lower()
    if kind_val not in {"score", "y_hat"}:
        raise ValueError("kind must be 'score' or 'y_hat'.")

    if kind_val == "y_hat":
        if components not in {"all", "logic", "intensity"}:
            raise ValueError("components must be 'all', 'logic', or 'intensity'.")
        if components == "logic":
            idx = slice(0, 4)
        elif components == "intensity":
            if preds.shape[2] < 8:
                raise ValueError("Predictions missing intensity components (need 8 outputs).")
            idx = slice(4, 8)
        else:
            idx = slice(0, preds.shape[2])
        var = np.var(preds[:, :, idx], axis=0)
        if reduction == "mean":
            values = np.mean(var, axis=1)
        elif reduction == "max":
            values = np.max(var, axis=1)
        else:
            raise ValueError("reduction must be 'mean' or 'max'.")
        return UncertaintyResult(
            values=np.asarray(values, dtype=float).ravel(),
            kind="y_hat",
            components=components,
            reduction=reduction,
            detail={"n_trees": int(preds.shape[0])},
        )

    setpoint = np.asarray(ctx.setpoint, dtype=float).ravel()
    if setpoint.size != 4:
        raise ValueError("setpoint must have length 4.")
    w = sfxi_math.weights_from_setpoint(setpoint, eps=1e-12)
    intensity_disabled = not np.any(w)
    denom = ctx.denom
    if not intensity_disabled and (denom is None or not np.isfinite(denom) or denom <= 0.0):
        raise ValueError("denom is required and must be positive to score uncertainty when intensity is enabled.")

    scores = []
    for t in range(preds.shape[0]):
        y_hat = preds[t]
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
        scores.append(score)

    score_stack = np.stack(scores, axis=0)
    values = np.var(score_stack, axis=0)
    return UncertaintyResult(
        values=np.asarray(values, dtype=float).ravel(),
        kind="score",
        components=None,
        reduction="variance",
        detail={"n_trees": int(preds.shape[0])},
    )
