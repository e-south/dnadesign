"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/models/random_forest.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from joblib import dump
from joblib import load as joblib_load
from sklearn.ensemble import RandomForestRegressor

from ..core.round_context import roundctx_contract
from ..registries.models import register_model


@dataclass
class FitMetrics:
    oob_r2: Optional[float] = None
    oob_mse: Optional[float] = None


@roundctx_contract(
    category="model",
    requires=[],
    produces=[
        "model/<self>/x_dim",
        "model/<self>/y_dim",
        "model/<self>/fit_metrics",
    ],
)
@register_model("random_forest")
class RandomForestModel:
    """
    RandomForest regressor plugin (multi-output safe).
    Parameters mirror sklearn RandomForestRegressor plus:
      - emit_feature_importance (bool; default False): emit per-round CSV artifact.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = dict(params or {})
        self.emit_feature_importance = bool(self.params.pop("emit_feature_importance", False))
        # sklearn constructor params remain in self.params
        self._est: Optional[RandomForestRegressor] = None
        self._x_dim: Optional[int] = None
        self._feature_importance: Optional[np.ndarray] = None

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
        # include our convenience flag in the public echo regardless
        out["emit_feature_importance"] = bool(self.emit_feature_importance)
        return out

    def fit(self, X: np.ndarray, Y: np.ndarray, *, ctx=None) -> FitMetrics:
        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise ValueError("[random_forest] X must be a 2D numpy array.")
        if not (isinstance(Y, np.ndarray) and Y.ndim == 2):
            raise ValueError("[random_forest] Y must be a 2D numpy array.")
        y_dim = int(Y.shape[1])

        est = RandomForestRegressor(**self.params)
        # sklearn expects 1D targets for single-output regression.
        # np.asarray avoids matrix subclasses keeping 2D shape after ravel().
        y_fit = np.asarray(Y).reshape(-1) if y_dim == 1 else Y
        est.fit(X, y_fit)
        self._est = est
        self._x_dim = int(X.shape[1])

        # OOB diagnostics if enabled
        oob_r2 = None
        oob_mse = None
        if bool(self.params.get("oob_score", False)):
            try:
                oob_r2 = float(getattr(est, "oob_score_", None))
            except Exception:
                oob_r2 = None

            try:
                oob_pred = np.asarray(getattr(est, "oob_prediction_", None), dtype=float)
                if oob_pred.size:
                    if y_dim == 1:
                        y_true = np.asarray(Y, dtype=float).reshape(-1)
                        oob_pred = oob_pred.reshape(-1)
                    else:
                        y_true = np.asarray(Y, dtype=float)
                    if oob_pred.shape == y_true.shape:
                        diff = oob_pred - y_true
                        oob_mse = float(np.nanmean(np.square(diff)))
            except Exception:
                oob_mse = None
        else:
            # sklearn only computes oob_score_ when enabled; keep None otherwise
            oob_r2 = None
            oob_mse = None

        # Optional: capture feature importance as an artifact
        if self.emit_feature_importance:
            try:
                imp = np.asarray(est.feature_importances_, dtype=float).ravel()
                if imp.size != self._x_dim:
                    # assertive: dimension mismatch indicates a bug in upstream transforms
                    raise ValueError(f"[random_forest] feature_importances_ length {imp.size} != X_dim {self._x_dim}")
                self._feature_importance = imp
            except Exception as e:
                # be explicit: if user asked for it, failure should surface
                raise RuntimeError(f"[random_forest] failed to compute feature importance: {e}")

        # Emit RoundCtx metadata for auditability
        if ctx is not None:
            ctx.set("model/<self>/x_dim", int(X.shape[1]))
            ctx.set("model/<self>/y_dim", int(y_dim))
            ctx.set("model/<self>/fit_metrics", {"oob_r2": oob_r2})

        return FitMetrics(oob_r2=oob_r2, oob_mse=oob_mse)

    def predict(self, X: np.ndarray, *, ctx=None) -> np.ndarray:
        if self._est is None:
            raise RuntimeError("[random_forest] predict() before fit().")
        y = np.asarray(self._est.predict(X), dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def iter_ensemble_predictions(
        self,
        X: np.ndarray,
        *,
        batch_size: int,
    ):
        """
        Stream per-estimator predictions as (estimator_index, row_start, row_end, y_pred_batch).
        """
        if self._est is None:
            raise RuntimeError("[random_forest] iter_ensemble_predictions() before fit().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("[random_forest] X must be a 2D numpy array.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("[random_forest] batch_size must be a positive integer.")
        ests = getattr(self._est, "estimators_", None)
        if not ests:
            raise ValueError("[random_forest] estimators_ missing for ensemble predictions.")
        n_rows = int(X_arr.shape[0])
        for idx, est in enumerate(ests):
            for start in range(0, n_rows, batch_size):
                end = min(start + batch_size, n_rows)
                y = np.asarray(est.predict(X_arr[start:end]), dtype=float)
                if y.ndim == 1:
                    y = y.reshape(-1, 1)
                if y.shape[0] != (end - start):
                    raise ValueError("[random_forest] estimator prediction batch has invalid row count.")
                if not np.all(np.isfinite(y)):
                    raise ValueError("[random_forest] estimator predictions contain non-finite values.")
                yield idx, start, end, y

    def feature_importances(self) -> Optional[np.ndarray]:
        """Return 1-D array of feature importances if available."""
        if self._est is None:
            return None
        try:
            imp = np.asarray(self._est.feature_importances_, dtype=float).ravel()
            return imp
        except Exception:
            return None

    def save(self, path: str) -> None:
        if self._est is None:
            raise RuntimeError("[random_forest] save() before fit().")
        dump(self._est, path)
        # Parameters are recoverable from the estimator itself at load-time

    @classmethod
    def load(cls, path: str, params: Optional[Dict[str, Any]] = None) -> "RandomForestModel":
        est = joblib_load(path)
        # If params are provided, prefer them; otherwise use estimator params.
        resolved = params if params is not None else getattr(est, "get_params", lambda **_: {})()
        m = cls(params=resolved)
        m._est = est
        return m

    @classmethod
    def from_artifact(cls, obj: Any) -> Optional["RandomForestModel"]:
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, RandomForestRegressor):
            params = getattr(obj, "get_params", lambda **_: {})()
            m = cls(params=params)
            m._est = obj
            m._x_dim = int(getattr(obj, "n_features_in_", 0) or 0) or None
            return m
        return None

    # ---- optional artifact hook --------------------------------------------
    def model_artifacts(self) -> Dict[str, pd.DataFrame]:
        """
        Return zero or more artifacts to be written by run_round in a model-agnostic way.
        Keys are artifact names; values are DataFrames to persist (CSV).
        """
        if self._feature_importance is None:
            return {}
        df = pd.DataFrame(
            {
                "feature_index": np.arange(self._x_dim, dtype=int),
                "importance": self._feature_importance.astype(float),
            }
        )
        # normalize to unit sum (defensive)
        s = float(df["importance"].sum())
        if s > 0:
            df["importance"] = df["importance"] / s
        return {"feature_importance": df}
