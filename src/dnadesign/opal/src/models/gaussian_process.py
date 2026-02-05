"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/models/gaussian_process.py

Module Author(s): Eric J. South, Elm Markert
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from joblib import dump
from joblib import load as joblib_load
from sklearn.gaussian_process import GaussianProcessRegressor

from ..core.round_context import roundctx_contract
from ..registries.models import register_model

@dataclass
class FitMetrics:
    kernal_lml: Optional[float] = None

@roundctx_contract(
    category="model",
    requires_by_stage={"fit":[], "predict":[]},
    produces_by_stage={
        "fit": ["model/<self>/x_dim",
        "model/<self>/y_dim",
        "model/<self>/fit_metrics",],
        "predict":["model/<self>/std_devs",]
    },
)

@register_model("gaussian_process")
class GaussianProcessModel:
    """
    GaussianProcess regressor plugin
    Parameters mirror sklearn GaussianProcessRegressor plus 
    """
    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = dict(params or {})
        # sklearn constructor params remain in self.params
        self._est: Optional[GaussianProcessRegressor] = None
        self._x_dim: Optional[int] = None

    # ---- plugin surface -----------------------------------------------------
    def get_params(self) -> Dict[str, Any]:
        # Prefer estimator params when available (e.g., after load)
        out: Dict[str, Any] = {}
        if self._est is not None:
            try:
                out.update(self._est.get_params(deep=False))
            except Exception:
                pass
        else:
            out.update(self.params)
        return out

    def fit(self, X: np.ndarray, Y: np.ndarray, *, ctx=None):
        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise ValueError("[guassian_process] X must be a 2D numpy array.")
        if not (isinstance(Y, np.ndarray) and Y.ndim == 2):
            raise ValueError("[gaussian_process] Y must be a 2D numpy array.")
        y_dim = int(Y.shape[1])

        est = GaussianProcessRegressor(**self.params)
        # sklearn expects 1D targets for single-output regression.
        # np.asarray avoids matrix subclasses keeping 2D shape after ravel().
        y_fit = np.asarray(Y).reshape(-1) if y_dim == 1 else Y
        est.fit(X, y_fit)
        self._est = est
        self._x_dim = int(X.shape[1])

        # sklearn always returns kernal_lml, so we will also return it as a fit metric
        kernal_lml = est.log_marginal_likelihood()

        # Emit ctx metadata for auditability
        if ctx is not None:
            ctx.set("model/<self>/x_dim", int(X.shape[1]))
            ctx.set("model/<self>/y_dim", int(y_dim))
            ctx.set("model/<self>/fit_metrics", {"kernal_lml": kernal_lml})

        return FitMetrics(kernal_lml=kernal_lml)

    def predict(self, X: np.ndarray, std = True, *, ctx=None) -> np.ndarray:
        if self._est is None:
            raise RuntimeError("[gaussian_process] predict() before fit().")
        y, sd = self._est.predict(X, return_std = std)
        y = np.asarray(y, dtype=float)
        sd = np.asarray(sd, dtype=float)
        if ctx is not None:
            ctx.set("model/<self>/std_devs", sd)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def save(self, path: str) -> None:
        if self._est is None:
            raise RuntimeError("[gaussian_process] save() before fit().")
        dump(self._est, path)
        # Parameters are recoverable from the estimator itself at load-time

    @classmethod
    def load(cls, path: str, params: Optional[Dict[str, Any]] = None) -> "GaussianProcessModel":
        est = joblib_load(path)
        # If params are provided, prefer them; otherwise use estimator params.
        resolved = params if params is not None else getattr(est, "get_params", lambda **_: {})()
        m = cls(params=resolved)
        m._est = est
        return m