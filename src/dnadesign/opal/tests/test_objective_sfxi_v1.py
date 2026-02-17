"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_objective_sfxi_v1.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.opal.src.core.round_context import PluginRegistryView, RoundCtx
from dnadesign.opal.src.objectives.sfxi_math import (
    STATE_ORDER,
    effect_raw_from_y_star,
    effect_scaled,
    logic_fidelity,
    parse_setpoint_vector,
)
from dnadesign.opal.src.objectives.sfxi_v1 import sfxi_v1


class _TrainView:
    def __init__(self, Y: np.ndarray, R: np.ndarray, as_of_round: int) -> None:
        self._Y = np.asarray(Y, dtype=float)
        self._R = np.asarray(R, dtype=int)
        self._as = int(as_of_round)

    def iter_labels_y_current_round(self):
        mask = self._R == self._as
        for i in np.where(mask)[0].tolist():
            yield self._Y[i, :]


def _ctx(as_of_round: int = 0) -> RoundCtx:
    reg = PluginRegistryView("rf", "sfxi_v1", "top_n", "identity", "sfxi_vec8_from_table_v1")
    return RoundCtx(core={"core/labels_as_of_round": int(as_of_round)}, registry=reg)


def test_sfxi_v1_scores_and_ctx_denom():
    # Two candidates with identical logic; different intensity in the setpoint state.
    y_pred = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # E_raw = 2 (2^1)
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # E_raw = 1 (2^0)
        ],
        dtype=float,
    )
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }

    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=None)

    # denom = 2, logic fidelity = 1, so scores should be [1.0, 0.5].
    assert np.allclose(res.scores_by_name["sfxi"], np.array([1.0, 0.5], dtype=float))
    assert int(res.diagnostics["summary_stats"]["denom_percentile"]) == 95

    snap = rctx.snapshot()
    assert snap["objective/sfxi_v1/denom_percentile"] == 95
    assert np.isclose(snap["objective/sfxi_v1/denom_value"], 2.0)


def test_sfxi_v1_requires_min_labels():
    y_pred = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "scaling": {"percentile": 95, "min_n": 2, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    with pytest.raises(ValueError, match="min_n"):
        sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=None)


def test_sfxi_v1_rejects_out_of_range_setpoint():
    y_pred = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    params = {"setpoint_vector": [0.0, -0.2, 0.0, 1.2]}

    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    with pytest.raises(ValueError, match="setpoint_vector"):
        sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=None)


def test_sfxi_v1_all_off_disables_intensity():
    y_pred = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0],
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
        ],
        dtype=float,
    )
    params = {
        "setpoint_vector": [0, 0, 0, 0],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 2.0,
        "scaling": {"percentile": 95, "min_n": 5, "eps": 1e-8},
    }

    train_Y = np.empty((0, 8), dtype=float)
    train_R = np.empty((0,), dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=None)

    assert np.allclose(res.scores_by_name["sfxi"], np.array([1.0, 0.0], dtype=float))
    assert res.diagnostics.get("intensity_disabled") is True
    assert np.allclose(res.diagnostics["effect_scaled"], np.ones(2, dtype=float))


def test_sfxi_v1_uncertainty_zero_when_std_zero():
    y_pred = np.array(
        [
            [0.2, 0.1, 0.0, 0.9, 0.4, 0.5, 0.3, 0.7],
            [0.1, 0.2, 0.1, 0.8, 0.5, 0.6, 0.4, 0.9],
        ],
        dtype=float,
    )
    y_pred_std = np.zeros_like(y_pred)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)
    unc = np.asarray(res.uncertainty_by_name["sfxi"], dtype=float)
    np.testing.assert_allclose(unc, np.zeros(y_pred.shape[0], dtype=float), rtol=0.0, atol=1e-12)


def test_sfxi_v1_uncertainty_all_off_setpoint_depends_on_logic():
    y_pred = np.array(
        [
            [0.2, 0.1, 0.2, 0.1, 2.0, 2.0, 2.0, 2.0],
            [0.8, 0.7, 0.9, 0.8, 2.0, 2.0, 2.0, 2.0],
        ],
        dtype=float,
    )
    y_pred_std = np.array(
        [
            [0.08, 0.06, 0.08, 0.06, 0.0, 0.0, 0.0, 0.0],
            [0.08, 0.06, 0.08, 0.06, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    params = {
        "setpoint_vector": [0, 0, 0, 0],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.empty((0, 8), dtype=float)
    train_R = np.empty((0,), dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)
    unc = np.asarray(res.uncertainty_by_name["sfxi"], dtype=float)
    assert np.any(unc > 0.0)


def test_sfxi_v1_uncertainty_delta_matches_monte_carlo_smoke():
    y_pred = np.array([[0.1, 0.2, 0.15, 0.85, 0.3, 0.5, 0.2, 0.9]], dtype=float)
    y_pred_std = np.array([[0.02, 0.03, 0.02, 0.02, 0.05, 0.04, 0.05, 0.03]], dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "intensity_log2_offset_delta": 0.0,
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)
    est_std = float(np.asarray(res.uncertainty_by_name["sfxi"], dtype=float)[0])

    denom = float(res.diagnostics["denom_used"])
    setpoint = parse_setpoint_vector(params)
    beta = float(params["logic_exponent_beta"])
    gamma = float(params["intensity_exponent_gamma"])
    delta = float(params["intensity_log2_offset_delta"])
    rng = np.random.default_rng(7)
    draws = 2000
    samples = rng.normal(loc=y_pred[0], scale=y_pred_std[0], size=(draws, y_pred.shape[1]))
    v_hat = np.clip(samples[:, 0:4], 0.0, 1.0)
    y_star = samples[:, 4:8]
    logic = logic_fidelity(v_hat, setpoint)
    effect_raw, _ = effect_raw_from_y_star(y_star, setpoint, delta=delta, eps=1e-12, state_order=STATE_ORDER)
    effect = effect_scaled(effect_raw, denom)
    scores = np.power(logic, beta) * np.power(effect, gamma)
    mc_std = float(np.std(scores, ddof=1))
    rel_err = abs(est_std - mc_std) / max(mc_std, 1e-12)
    assert rel_err < 0.30
