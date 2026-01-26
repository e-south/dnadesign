"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_sfxi_uncertainty.py

Tests uncertainty contract and RF adapter behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.opal.src.analysis.sfxi.uncertainty import UncertaintyContext, compute_uncertainty
from dnadesign.opal.src.models.random_forest import RandomForestModel


def test_random_forest_uncertainty_shapes_and_nonnegative():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    Y = rng.normal(size=(8, 8))

    model = RandomForestModel(params={"n_estimators": 5, "random_state": 1})
    model.fit(X, Y)

    ctx = UncertaintyContext(
        setpoint=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        denom=None,
        y_ops=[],
        round_ctx=None,
    )

    res_vec = compute_uncertainty(model, X, kind="y_hat", ctx=ctx)
    assert res_vec.values.shape == (X.shape[0],)
    assert np.all(res_vec.values >= 0.0)

    res_score = compute_uncertainty(model, X, kind="score", ctx=ctx)
    assert res_score.values.shape == (X.shape[0],)
    assert np.all(res_score.values >= 0.0)
