"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/models/test_gaussian_process_model.py

Regression tests for GaussianProcess model prediction and load semantics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.opal.src.models.gaussian_process import GaussianProcessModel


class _Ctx:
    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value


def test_gaussian_process_predict_std_false_does_not_unpack() -> None:
    X = np.array([[0.0], [0.5], [1.0]], dtype=float)
    Y = np.array([[0.0], [0.5], [1.0]], dtype=float)
    model = GaussianProcessModel(params={"normalize_y": False, "alpha": 1e-6})
    model.fit(X, Y)

    yhat = model.predict(np.array([[0.25], [0.75]], dtype=float), std=False)
    assert yhat.shape == (2, 1)


def test_gaussian_process_predict_std_true_reshapes_sd_for_scalar() -> None:
    X = np.array([[0.0], [0.5], [1.0]], dtype=float)
    Y = np.array([[0.0], [0.5], [1.0]], dtype=float)
    model = GaussianProcessModel(params={"normalize_y": False, "alpha": 1e-6})
    model.fit(X, Y)
    ctx = _Ctx()

    yhat = model.predict(np.array([[0.25], [0.75]], dtype=float), std=True, ctx=ctx)
    assert yhat.shape == (2, 1)
    payload = ctx.get("model/<self>/std_devs")
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert np.asarray(payload[0], dtype=float).shape == (2, 1)


def test_gaussian_process_load_without_params_works(tmp_path) -> None:
    X = np.array([[0.0], [0.5], [1.0]], dtype=float)
    Y = np.array([[0.0], [0.5], [1.0]], dtype=float)
    model = GaussianProcessModel(
        params={
            "normalize_y": False,
            "alpha": 1e-6,
            "kernel": {"name": "matern", "length_scale": 0.5, "nu": 1.5, "with_white_noise": True},
        }
    )
    model.fit(X, Y)
    model_path = tmp_path / "gp.joblib"
    model.save(str(model_path))

    loaded = GaussianProcessModel.load(str(model_path), params=None)
    yhat = loaded.predict(np.array([[0.25], [0.75]], dtype=float), std=False)
    assert yhat.shape == (2, 1)
