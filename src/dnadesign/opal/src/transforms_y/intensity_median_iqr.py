"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/transforms_y/intensity_median_iqr.py

Y-op: Robustly center/scale intensity targets (indices 4:8) by median/IQR.
- Fit phase computes per-dimension center/scale on training Y.
- Transform applies (Y[:,4:8] - center) / max(iqr, eps).
- Inverse restores objective-space: Y[:,4:8] * scale + center.
- Logic entries (0:4) are left unchanged.

Module Author(s): Eric J. South
Dunlop Lab
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
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if ctx is not None and not ctx.get("yops/intensity_median_iqr/enabled", True):
        return Y
    if Y.shape[1] < 8:
        return Y
    med = np.asarray((ctx.get("yops/intensity_median_iqr/center") or [0, 0, 0, 0]), dtype=float)
    iqr = np.asarray((ctx.get("yops/intensity_median_iqr/scale") or [1, 1, 1, 1]), dtype=float)
    eps = float(ctx.get("yops/intensity_median_iqr/eps") or params.eps)

    out = Y.copy()
    out[:, 4:8] = (out[:, 4:8] - med[None, :]) / np.maximum(iqr[None, :], eps)
    return out


def _inverse(Y: np.ndarray, params: _Params, ctx=None) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if ctx is not None and not ctx.get("yops/intensity_median_iqr/enabled", True):
        return Y
    if Y.shape[1] < 8:
        return Y
    med = np.asarray((ctx.get("yops/intensity_median_iqr/center") or [0, 0, 0, 0]), dtype=float)
    iqr = np.asarray((ctx.get("yops/intensity_median_iqr/scale") or [1, 1, 1, 1]), dtype=float)

    out = Y.copy()
    out[:, 4:8] = out[:, 4:8] * iqr[None, :] + med[None, :]
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
    return _fit, _transform, _inverse, _Params
