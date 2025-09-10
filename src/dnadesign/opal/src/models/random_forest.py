"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/models/random_forest.py

RandomForest wrapper with optional per-target robust scaling during fit.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ..registries.models import register_model


@dataclass(frozen=True)
class FitMetrics:
    oob_r2: Optional[float]
    oob_mse: Optional[float]


class _TargetScaler:
    """
    Robust per-target scaler (median, IQR/1.349). Safe on small-N; can be disabled.
    """

    def __init__(
        self,
        enable: bool = True,
        minimum_labels_required: int = 5,
        center_statistic: str = "median",
        scale_statistic: str = "iqr",
    ) -> None:
        self.enable = enable
        self.minimum_labels_required = int(minimum_labels_required)
        self.center_statistic = center_statistic
        self.scale_statistic = scale_statistic
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def _robust_center_scale(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        med = np.nanmedian(Y, axis=0)
        q75 = np.nanpercentile(Y, 75, axis=0)
        q25 = np.nanpercentile(Y, 25, axis=0)
        iqr = q75 - q25
        scale = np.where(iqr < 1e-12, 1.0, iqr / 1.349)
        return med, scale

    def fit(self, Y: np.ndarray) -> None:
        if not self.enable or Y.shape[0] < self.minimum_labels_required:
            self.center_, self.scale_ = None, None
            return
        c, s = self._robust_center_scale(Y)
        # Remove degenerate scaling
        s = np.where(np.isfinite(s) & (np.abs(s) > 1e-12), s, 1.0)
        self.center_, self.scale_ = c, s

    def transform(self, Y: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            return Y
        return (Y - self.center_.reshape(1, -1)) / self.scale_.reshape(1, -1)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            return Y
        return Y * self.scale_.reshape(1, -1) + self.center_.reshape(1, -1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable": self.enable,
            "minimum_labels_required": self.minimum_labels_required,
            "center_statistic": self.center_statistic,
            "scale_statistic": self.scale_statistic,
            "center_": None if self.center_ is None else self.center_.tolist(),
            "scale_": None if self.scale_ is None else self.scale_.tolist(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "_TargetScaler":
        ts = _TargetScaler(
            enable=d.get("enable", True),
            minimum_labels_required=d.get("minimum_labels_required", 5),
            center_statistic=d.get("center_statistic", "median"),
            scale_statistic=d.get("scale_statistic", "iqr"),
        )
        if d.get("center_") is not None:
            ts.center_ = np.asarray(d["center_"], dtype=float)
        if d.get("scale_") is not None:
            ts.scale_ = np.asarray(d["scale_"], dtype=float)
        return ts


class RandomForestModel:
    """
    Thin wrapper around sklearn RandomForestRegressor that:
      - handles robust per-target scaling at fit-time,
      - exposes per-tree predictions for uncertainty,
      - saves/loads via joblib with scaler state.
    """

    def __init__(self, rf: RandomForestRegressor, target_scaler: _TargetScaler):
        self.rf = rf
        self._scaler = target_scaler

    # ---------- Training ----------
    def fit(self, X: np.ndarray, Y: np.ndarray) -> FitMetrics:
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self._scaler.fit(Y)
        Y_scaled = self._scaler.transform(Y)

        # Ensure OOB enabled if requested in params
        self.rf.fit(X, Y_scaled)

        oob_r2 = getattr(self.rf, "oob_score_", None)
        oob_mse = None
        if hasattr(self.rf, "oob_prediction_"):
            oob_pred = self.rf.oob_prediction_
            if oob_pred is not None:
                # MSE on scaled space; single number averaged across outputs
                err = (oob_pred - Y_scaled) ** 2
                oob_mse = float(np.nanmean(err))
        return FitMetrics(
            oob_r2=float(oob_r2) if oob_r2 is not None else None, oob_mse=oob_mse
        )

    # ---------- Inference ----------
    def predict(self, X: np.ndarray) -> np.ndarray:
        yh_scaled = self.rf.predict(X)
        if yh_scaled.ndim == 1:
            yh_scaled = yh_scaled.reshape(-1, 1)
        yh = self._scaler.inverse_transform(yh_scaled)
        return yh

    def predict_per_tree(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (n_trees, n_samples, n_outputs) in original (inverse-transformed) units.
        """
        preds = []
        for est in self.rf.estimators_:
            y = est.predict(X)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            y = self._scaler.inverse_transform(y)
            preds.append(y)
        return np.asarray(preds, dtype=float)

    # ---------- Introspection ----------
    def feature_importances(self) -> Optional[np.ndarray]:
        return getattr(self.rf, "feature_importances_", None)

    def get_params(self) -> Dict[str, Any]:
        return self.rf.get_params(deep=True)

    # ---------- Persistence ----------
    def save(self, path: str | Path) -> None:
        obj = {"sk_model": self.rf, "target_scaler": self._scaler.to_dict()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)

    @staticmethod
    def load(path: str | Path) -> "RandomForestModel":
        obj = joblib.load(path)
        rf = obj["sk_model"]
        scaler = _TargetScaler.from_dict(obj.get("target_scaler", {}))
        return RandomForestModel(rf, scaler)


# ---------- Registry factory ----------


@register_model("random_forest")
def _factory(
    params: Dict[str, Any], target_scaler_cfg: Dict[str, Any]
) -> RandomForestModel:
    rf = RandomForestRegressor(**params)
    scaler = _TargetScaler(**target_scaler_cfg)
    return RandomForestModel(rf, scaler)
