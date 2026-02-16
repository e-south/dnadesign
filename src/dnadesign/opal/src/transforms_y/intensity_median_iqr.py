"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/transforms_y/intensity_median_iqr.py

Module Author(s): Elm Markert, Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from ..registries.transforms_y import register_y_op


class _Params(BaseModel):
    min_labels: int = 5
    center: str = Field(default="median", pattern="^(median)$")
    scale: str = Field(default="iqr", pattern="^(iqr)$")
    eps: float = 1e-8


def _required_ctx_key(ctx, key: str):
    if ctx is None:
        raise ValueError("intensity_median_iqr requires ctx state.")
    sentinel = object()
    val = ctx.get(key, sentinel)
    if val is sentinel:
        raise ValueError(f"intensity_median_iqr missing required context key: {key}.")
    return val


def _required_center(ctx) -> np.ndarray:
    center = np.asarray(_required_ctx_key(ctx, "yops/intensity_median_iqr/center"), dtype=float).reshape(-1)
    if center.size != 4:
        raise ValueError("intensity_median_iqr center must have length 4.")
    if not np.all(np.isfinite(center)):
        raise ValueError("intensity_median_iqr center must be finite.")
    return center


def _required_scale(ctx) -> np.ndarray:
    scale = np.asarray(_required_ctx_key(ctx, "yops/intensity_median_iqr/scale"), dtype=float).reshape(-1)
    if scale.size != 4:
        raise ValueError("intensity_median_iqr scale must have length 4.")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("intensity_median_iqr scale must contain positive finite values.")
    return scale


def _required_eps(ctx) -> float:
    eps = float(_required_ctx_key(ctx, "yops/intensity_median_iqr/eps"))
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("intensity_median_iqr eps must be positive finite.")
    return eps


def _fit(Y: np.ndarray, params: _Params, ctx=None) -> None:
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, d = Y.shape
    if n < int(params.min_labels):
        # mark as no-op
        if ctx is not None:
            ctx.set("yops/intensity_median_iqr/enabled", False)
            ctx.set("yops/intensity_median_iqr/center", [0.0, 0.0, 0.0, 0.0])
            ctx.set("yops/intensity_median_iqr/scale", [1.0, 1.0, 1.0, 1.0])
            ctx.set("yops/intensity_median_iqr/eps", float(params.eps))
        return
    if d < 8:
        raise ValueError("intensity_median_iqr expects y-dim >= 8")

    block = Y[:, 4:8]
    med = np.median(block, axis=0)
    q75 = np.percentile(block, 75, axis=0)
    q25 = np.percentile(block, 25, axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr <= 0, params.eps, iqr)

    if ctx is not None:
        ctx.set("yops/intensity_median_iqr/enabled", True)
        ctx.set("yops/intensity_median_iqr/center", med.tolist())
        ctx.set("yops/intensity_median_iqr/scale", iqr.tolist())
        ctx.set("yops/intensity_median_iqr/eps", float(params.eps))


def _transform(Y: np.ndarray, params: _Params, ctx=None) -> np.ndarray:
    del params
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    enabled = bool(_required_ctx_key(ctx, "yops/intensity_median_iqr/enabled"))
    if not enabled:
        return Y
    if Y.shape[1] < 8:
        return Y
    med = _required_center(ctx)
    iqr = _required_scale(ctx)
    _ = _required_eps(ctx)

    out = Y.copy()
    out[:, 4:8] = (out[:, 4:8] - med[None, :]) / iqr[None, :]
    return out


def _inverse(Y: np.ndarray, params: _Params, ctx=None) -> np.ndarray:
    del params
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    enabled = bool(_required_ctx_key(ctx, "yops/intensity_median_iqr/enabled"))
    if not enabled:
        return Y
    if Y.shape[1] < 8:
        return Y
    med = _required_center(ctx)
    iqr = _required_scale(ctx)
    _ = _required_eps(ctx)

    out = Y.copy()
    out[:, 4:8] = out[:, 4:8] * iqr[None, :] + med[None, :]
    return out


def _inverse_std(Y: np.ndarray, params: _Params, ctx=None, *, y_pred_transformed=None) -> np.ndarray:
    del params, y_pred_transformed
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if not np.all(np.isfinite(Y)):
        raise ValueError("intensity_median_iqr inverse_std expects finite values.")
    if np.any(Y < 0.0):
        raise ValueError("intensity_median_iqr inverse_std expects non-negative standard deviations.")
    enabled = bool(_required_ctx_key(ctx, "yops/intensity_median_iqr/enabled"))
    if not enabled:
        return Y
    if Y.shape[1] < 8:
        return Y
    iqr = _required_scale(ctx)
    _ = _required_eps(ctx)

    out = Y.copy()
    out[:, 4:8] = out[:, 4:8] * iqr[None, :]
    return out


@register_y_op(
    "intensity_median_iqr",
    produces=[
        "yops/<self>/enabled",
        "yops/<self>/center",
        "yops/<self>/scale",
        "yops/<self>/eps",
    ],
)
def _entry():
    setattr(_inverse, "inverse_std", _inverse_std)
    return _fit, _transform, _inverse, _Params
