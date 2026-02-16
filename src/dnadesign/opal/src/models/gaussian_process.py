"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/models/gaussian_process.py

Implements the Gaussian Process model plugin and emits predictive uncertainty
to RoundCtx for objective/selection consumption.

Module Author(s): Elm Markert, Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from joblib import dump
from joblib import load as joblib_load
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    DotProduct,
    Kernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)

from ..core.round_context import roundctx_contract
from ..registries.models import register_model


@dataclass
class FitMetrics:
    kernel_lml: Optional[float] = None


def _kernel_bounds(raw: Dict[str, Any], *, field_name: str) -> tuple[float, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"[gaussian_process] {field_name} must be an object with 'lower' and 'upper'.")
    lo = float(raw.get("lower", 1e-5))
    hi = float(raw.get("upper", 1e5))
    if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0.0 and hi > lo):
        raise ValueError(f"[gaussian_process] {field_name} must satisfy 0 < lower < upper.")
    return lo, hi


def _build_kernel(kernel_cfg: Dict[str, Any] | Kernel | None):
    if kernel_cfg is None:
        return None
    if isinstance(kernel_cfg, Kernel):
        return kernel_cfg
    if not isinstance(kernel_cfg, dict):
        raise ValueError("[gaussian_process] kernel must be a mapping.")
    name = str(kernel_cfg.get("name", "rbf")).strip().lower()
    if name == "rbf":
        kernel = RBF(
            length_scale=float(kernel_cfg.get("length_scale", 1.0)),
            length_scale_bounds=_kernel_bounds(
                kernel_cfg.get("length_scale_bounds", {}),
                field_name="length_scale_bounds",
            ),
        )
    elif name == "matern":
        kernel = Matern(
            length_scale=float(kernel_cfg.get("length_scale", 1.0)),
            length_scale_bounds=_kernel_bounds(
                kernel_cfg.get("length_scale_bounds", {}),
                field_name="length_scale_bounds",
            ),
            nu=float(kernel_cfg.get("nu", 1.5)),
        )
    elif name == "rational_quadratic":
        kernel = RationalQuadratic(
            length_scale=float(kernel_cfg.get("length_scale", 1.0)),
            alpha=float(kernel_cfg.get("alpha", 1.0)),
            length_scale_bounds=_kernel_bounds(
                kernel_cfg.get("length_scale_bounds", {}),
                field_name="length_scale_bounds",
            ),
            alpha_bounds=_kernel_bounds(kernel_cfg.get("alpha_bounds", {}), field_name="alpha_bounds"),
        )
    elif name == "dot_product":
        kernel = DotProduct(
            sigma_0=float(kernel_cfg.get("sigma_0", 1.0)),
            sigma_0_bounds=_kernel_bounds(kernel_cfg.get("sigma_0_bounds", {}), field_name="sigma_0_bounds"),
        )
    else:
        raise ValueError(
            "[gaussian_process] kernel.name must be one of ['rbf', 'matern', 'rational_quadratic', 'dot_product']."
        )

    if bool(kernel_cfg.get("with_white_noise", False)):
        kernel = kernel + WhiteKernel(
            noise_level=float(kernel_cfg.get("noise_level", 1.0)),
            noise_level_bounds=_kernel_bounds(
                kernel_cfg.get("noise_level_bounds", {}),
                field_name="noise_level_bounds",
            ),
        )
    return kernel


def _json_safe_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in (params or {}).items():
        try:
            json.dumps(value)
            out[key] = value
        except Exception:
            out[key] = str(value)
    return out


@roundctx_contract(
    category="model",
    requires_by_stage={"fit": [], "predict": []},
    produces_by_stage={
        "fit": [
            "model/<self>/x_dim",
            "model/<self>/y_dim",
            "model/<self>/fit_metrics",
        ],
        "predict": [
            "model/<self>/std_devs",
        ],
    },
)
@register_model("gaussian_process")
class GaussianProcessModel:
    """
    GaussianProcess regressor plugin
    Parameters mirror sklearn GaussianProcessRegressor plus
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        raw_params = dict(params or {})
        self.params = _json_safe_params(raw_params)
        est_params = dict(raw_params)
        kernel_cfg = est_params.get("kernel")
        if kernel_cfg is not None:
            est_params["kernel"] = _build_kernel(kernel_cfg)
        self._est_params = est_params
        self._est: Optional[GaussianProcessRegressor] = None
        self._x_dim: Optional[int] = None

    # ---- plugin surface -----------------------------------------------------
    def get_params(self) -> Dict[str, Any]:
        return dict(self.params)

    def fit(self, X: np.ndarray, Y: np.ndarray, *, ctx=None):
        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise ValueError("[gaussian_process] X must be a 2D numpy array.")
        if not (isinstance(Y, np.ndarray) and Y.ndim == 2):
            raise ValueError("[gaussian_process] Y must be a 2D numpy array.")
        y_dim = int(Y.shape[1])

        est = GaussianProcessRegressor(**self._est_params)
        # sklearn expects 1D targets for single-output regression.
        # np.asarray avoids matrix subclasses keeping 2D shape after ravel().
        y_fit = np.asarray(Y).reshape(-1) if y_dim == 1 else Y
        est.fit(X, y_fit)
        self._est = est
        self._x_dim = int(X.shape[1])

        kernel_lml = est.log_marginal_likelihood()

        # Emit ctx metadata for auditability
        if ctx is not None:
            ctx.set("model/<self>/x_dim", int(X.shape[1]))
            ctx.set("model/<self>/y_dim", int(y_dim))
            ctx.set("model/<self>/fit_metrics", {"kernel_lml": kernel_lml})

        return FitMetrics(kernel_lml=kernel_lml)

    def predict(self, X: np.ndarray, std: bool = True, *, ctx=None) -> np.ndarray:
        if self._est is None:
            raise RuntimeError("[gaussian_process] predict() before fit().")

        if std:
            y, sd = self._est.predict(X, return_std=True)
        else:
            y = self._est.predict(X, return_std=False)
            sd = None

        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.ndim != 2:
            raise ValueError(f"[gaussian_process] predict output must be 1D or 2D; got shape={y.shape}.")

        if std:
            sd_arr = np.asarray(sd, dtype=float)
            if sd_arr.ndim == 1:
                sd_arr = sd_arr.reshape(-1, 1)
            if sd_arr.ndim != 2:
                raise ValueError(f"[gaussian_process] predict std output must be 1D or 2D; got shape={sd_arr.shape}.")
            if sd_arr.shape != y.shape:
                raise ValueError(
                    f"[gaussian_process] predict std shape mismatch: y shape={y.shape}, std shape={sd_arr.shape}."
                )
            if not np.all(np.isfinite(sd_arr)):
                raise ValueError("[gaussian_process] predict std output must be finite.")
            if np.any(sd_arr < 0.0):
                raise ValueError("[gaussian_process] predict std output must be non-negative.")

            if ctx is not None:
                sentinel = object()
                key = "model/<self>/std_devs"
                old_sd = ctx.get(key, sentinel)
                chunk = sd_arr.copy()
                if old_sd is sentinel:
                    ctx.set(key, [chunk])
                elif isinstance(old_sd, list):
                    ctx.set(key, old_sd + [chunk])
                else:
                    raise ValueError("[gaussian_process] invalid RoundCtx payload type for model/<self>/std_devs.")
        return y

    def save(self, path: str) -> None:
        if self._est is None:
            raise RuntimeError("[gaussian_process] save() before fit().")
        dump(self._est, path)
        # Parameters are recoverable from the estimator itself at load-time

    @classmethod
    def load(cls, path: str, params: Optional[Dict[str, Any]] = None) -> "GaussianProcessModel":
        est = joblib_load(path)
        raw = dict(params if params is not None else getattr(est, "get_params", lambda **_: {})(deep=False))
        m = cls(params=raw if params is not None else {})
        m._est = est
        m._est_params = dict(raw)
        m.params = _json_safe_params(dict(raw))
        x_dim = getattr(est, "n_features_in_", None)
        m._x_dim = int(x_dim) if x_dim is not None else None
        return m
