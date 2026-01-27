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
import polars as pl

from dnadesign.opal.src.analysis.sfxi.uncertainty import UncertaintyContext, compute_uncertainty
from dnadesign.opal.src.models.random_forest import RandomForestModel
from dnadesign.opal.src.objectives import sfxi_math
from dnadesign.opal.src.plots.sfxi_uncertainty import _coerce_y_ops


def test_random_forest_uncertainty_matches_naive_std():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    Y = rng.normal(scale=0.1, size=(8, 8))

    model = RandomForestModel(params={"n_estimators": 5, "random_state": 1})
    model.fit(X, Y)

    setpoint = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
    ctx = UncertaintyContext(
        setpoint=setpoint,
        beta=1.1,
        gamma=0.9,
        delta=0.0,
        denom=1.0,
        y_ops=[],
        round_ctx=None,
    )

    result = compute_uncertainty(model, X, ctx=ctx, batch_size=3)
    assert result.values.shape == (X.shape[0],)
    assert np.all(result.values >= 0.0)
    assert result.statistic == "std"

    scores = []
    for est in model._est.estimators_:
        y_hat = np.asarray(est.predict(X), dtype=float)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)
        v_hat = np.clip(y_hat[:, 0:4], 0.0, 1.0)
        F_logic = sfxi_math.logic_fidelity(v_hat, setpoint)
        y_star = y_hat[:, 4:8]
        E_raw, _ = sfxi_math.effect_raw_from_y_star(
            y_star,
            setpoint,
            delta=0.0,
            eps=1e-12,
            state_order=sfxi_math.STATE_ORDER,
        )
        E_scaled = sfxi_math.effect_scaled(E_raw, 1.0)
        score = np.power(F_logic, 1.1) * np.power(E_scaled, 0.9)
        scores.append(score)
    score_stack = np.stack(scores, axis=0)
    naive_std = np.std(score_stack, axis=0, ddof=0)
    assert np.allclose(result.values, naive_std, rtol=1e-6, atol=1e-8)


def test_uncertainty_batch_size_invariance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(9, 4))
    Y = rng.normal(scale=0.1, size=(9, 8))

    model = RandomForestModel(params={"n_estimators": 4, "random_state": 2})
    model.fit(X, Y)

    ctx = UncertaintyContext(
        setpoint=np.array([1.0, 1.0, 0.0, 0.0], dtype=float),
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        denom=1.0,
        y_ops=[],
        round_ctx=None,
    )

    res_small = compute_uncertainty(model, X, ctx=ctx, batch_size=2)
    res_large = compute_uncertainty(model, X, ctx=ctx, batch_size=7)
    assert np.allclose(res_small.values, res_large.values, rtol=1e-6, atol=1e-8)


def test_uncertainty_requires_contract_and_context():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(4, 2))

    ctx = UncertaintyContext(
        setpoint=np.array([1.0, 0.0, 1.0, 0.0], dtype=float),
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        denom=None,
        y_ops=[],
        round_ctx=None,
    )

    class NoEnsemble:
        pass

    try:
        compute_uncertainty(NoEnsemble(), X, ctx=ctx, batch_size=2)
        assert False, "Expected ValueError for missing ensemble contract."
    except ValueError as exc:
        assert "ensemble" in str(exc).lower()

    ctx_need_denom = UncertaintyContext(
        setpoint=np.array([1.0, 1.0, 0.0, 0.0], dtype=float),
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        denom=None,
        y_ops=[],
        round_ctx=None,
    )
    model = RandomForestModel(params={"n_estimators": 2, "random_state": 3})
    model.fit(X, np.random.default_rng(3).normal(scale=0.1, size=(4, 8)))
    try:
        compute_uncertainty(model, X, ctx=ctx_need_denom, batch_size=2)
        assert False, "Expected ValueError for missing denom when intensity enabled."
    except ValueError as exc:
        assert "denom" in str(exc).lower()

    ctx_need_round = UncertaintyContext(
        setpoint=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        denom=None,
        y_ops=[{"name": "noop", "params": {}}],
        round_ctx=None,
    )
    try:
        compute_uncertainty(model, X, ctx=ctx_need_round, batch_size=2)
        assert False, "Expected ValueError for missing round_ctx when y_ops present."
    except ValueError as exc:
        assert "round_ctx" in str(exc).lower()


def test_coerce_y_ops_from_polars_series():
    series = pl.Series([[{"name": "intensity_median_iqr", "params": {"min_labels": 5}}]])
    assert _coerce_y_ops(series) == [{"name": "intensity_median_iqr", "params": {"min_labels": 5}}]
