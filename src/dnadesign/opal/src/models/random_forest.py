"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/models/random_forest.py

Random Forest regressor wrapper.

Thin adapter around sklearn.ensemble.RandomForestRegressor that:
- preserves init params for reproducibility,
- exposes fit/predict and extracts OOB metrics (R², MSE) when enabled,
- returns feature importances,
- supports joblib save/load with params bundled.

Defaults are chosen to mirror EVOLVEpro's top-layer configuration.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


@dataclass
class RFMetrics:
    oob_r2: float | None
    oob_mse: float | None


class RandomForestModel:
    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.model = RandomForestRegressor(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RFMetrics:
        self.model.fit(X, y)
        oob_r2 = None
        oob_mse = None
        if getattr(self.model, "oob_score", False):
            try:
                oob_r2 = float(self.model.oob_score_)
                if (
                    hasattr(self.model, "oob_prediction_")
                    and self.model.oob_prediction_ is not None
                ):
                    oob_mse = float(mean_squared_error(y, self.model.oob_prediction_))
            except Exception:
                # Keep metrics None on failure
                pass
        return RFMetrics(oob_r2=oob_r2, oob_mse=oob_mse)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importances(self) -> np.ndarray:
        return getattr(self.model, "feature_importances_", None)

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "params": self.params}, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestModel":
        obj = joblib.load(path)
        mdl = cls(**obj.get("params", {}))
        mdl.model = obj["model"]
        return mdl

    def get_params(self) -> Dict[str, Any]:
        return dict(self.params)
