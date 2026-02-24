"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/objectives/test_objective_sfxi_v1.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.opal.src.core.round_context import PluginRegistryView, RoundCtx
from dnadesign.opal.src.objectives.sfxi_math import worst_corner_distance
from dnadesign.opal.src.objectives.sfxi_v1 import _scalar_uncertainty_analytical, sfxi_v1


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


def test_sfxi_v1_rejects_non_finite_y_pred():
    y_pred = np.array([[0.1, 0.2, 0.15, 0.85, 0.3, np.nan, 0.2, 0.9]], dtype=float)
    params = {"setpoint_vector": [0, 0, 0, 1], "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8}}
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    with pytest.raises(ValueError, match="y_pred must be finite"):
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
    with pytest.raises(ValueError, match=r"y_pred_std.*must be > 0"):
        sfxi_v1(
            y_pred=y_pred,
            params=params,
            ctx=octx,
            train_view=tv,
            y_pred_std=y_pred_std,
        )


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
        "uncertainty_method": "delta",
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


def test_sfxi_v1_uncertainty_allows_zero_std_for_zero_weight_intensity_states():
    y_pred = np.array([[0.2, 0.1, 0.2, 0.9, 0.3, 0.5, 0.2, 0.7]], dtype=float)
    y_pred_std = np.array([[0.08, 0.06, 0.08, 0.06, 0.0, 0.0, 0.0, 0.03]], dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "uncertainty_method": "delta",
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)
    unc = np.asarray(res.uncertainty_by_name["sfxi"], dtype=float)
    assert np.all(np.isfinite(unc))
    assert np.all(unc > 0.0)


def test_sfxi_v1_uncertainty_rejects_zero_std_for_weighted_intensity_state():
    y_pred = np.array([[0.2, 0.1, 0.2, 0.9, 0.3, 0.5, 0.2, 0.7]], dtype=float)
    y_pred_std = np.array([[0.08, 0.06, 0.08, 0.06, 0.03, 0.03, 0.03, 0.0]], dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "uncertainty_method": "delta",
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    with pytest.raises(ValueError, match=r"y_pred_std.*must be > 0"):
        sfxi_v1(
            y_pred=y_pred,
            params=params,
            ctx=octx,
            train_view=tv,
            y_pred_std=y_pred_std,
        )


def test_sfxi_v1_uncertainty_delta_regression_fixture():
    y_pred = np.array([[0.1, 0.2, 0.15, 0.85, 0.3, 0.5, 0.2, 0.6]], dtype=float)
    y_pred_std = np.array([[0.02, 0.03, 0.02, 0.02, 0.05, 0.04, 0.05, 0.03]], dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "intensity_log2_offset_delta": 0.0,
        "uncertainty_method": "delta",
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)
    est_std = float(np.asarray(res.uncertainty_by_name["sfxi"], dtype=float)[0])

    expected = 0.01871242242800085
    assert est_std == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_sfxi_v1_uncertainty_analytical_requires_beta_gamma_one():
    y_pred = np.array([[0.1, 0.2, 0.15, 0.85, 0.3, 0.5, 0.2, 0.9]], dtype=float)
    y_pred_std = np.array([[0.02, 0.03, 0.02, 0.02, 0.05, 0.04, 0.05, 0.03]], dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.1,
        "intensity_exponent_gamma": 1.0,
        "intensity_log2_offset_delta": 0.0,
        "uncertainty_method": "analytical",
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    with pytest.raises(ValueError, match="analytical.*requires"):
        sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)


def test_sfxi_v1_delta_rejects_fractional_beta_with_zero_logic_base():
    y_pred = np.array([[1.0, 1.0, 1.0, 0.0, 0.3, 0.4, 0.2, 0.8]], dtype=float)
    y_pred_std = np.full_like(y_pred, 0.05, dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 0.5,
        "intensity_exponent_gamma": 1.0,
        "intensity_log2_offset_delta": 0.0,
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    with pytest.raises(ValueError, match="logic_exponent_beta.*requires positive base"):
        sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)


def test_sfxi_v1_delta_rejects_fractional_gamma_with_zero_effect_base():
    y_pred = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    y_pred_std = np.full_like(y_pred, 0.05, dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 0.5,
        "intensity_log2_offset_delta": 1.0,
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    with pytest.raises(ValueError, match="intensity_exponent_gamma.*requires positive base"):
        sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)


def test_sfxi_v1_uncertainty_analytical_uses_log2_normal_moments():
    setpoint = np.array([0.3, 0.4, 0.7, 0.2], dtype=float)
    w = setpoint / float(np.sum(setpoint))
    y_pred = np.array([[0.2, 0.3, 0.4, 0.5, 0.9, 0.8, 1.1, 0.7]], dtype=float)
    y_pred_var = np.array([[0.03, 0.02, 0.04, 0.01, 0.36, 0.16, 0.25, 0.09]], dtype=float)
    denom = 1.25

    sigma = _scalar_uncertainty_analytical(
        y_pred=y_pred,
        y_pred_var=y_pred_var,
        v_hat=y_pred[:, 0:4],
        w=w,
        denom=denom,
        setpoint=setpoint,
        delta=0.0,
        intensity_disabled=False,
    )

    logic_mean = y_pred[:, 0:4]
    logic_var = y_pred_var[:, 0:4]
    D = float(worst_corner_distance(setpoint))
    c = 1.0 / (D**2)
    lf_var_unsc = (
        4.0 * (logic_mean**2) * logic_var
        + 4.0 * (setpoint[None, :] ** 2) * logic_var
        + 2.0 * (logic_var**2)
        - 8.0 * logic_mean * setpoint[None, :] * logic_var
    )
    lf_var = c * np.sum(lf_var_unsc, axis=1)
    lf_exp = c * np.sum(
        logic_mean**2 + logic_var - (2.0 * logic_mean * setpoint[None, :]) + setpoint[None, :] ** 2,
        axis=1,
    )

    ln2 = np.log(2.0)
    effect_var_unwt = (np.exp((ln2**2) * y_pred_var[:, 4:8]) - 1.0) * np.exp(
        2.0 * ln2 * y_pred[:, 4:8] + (ln2**2) * y_pred_var[:, 4:8]
    )
    effect_var = np.sum(effect_var_unwt * (w**2), axis=1) / (denom**2)
    effect_exp = np.sum(np.exp(ln2 * y_pred[:, 4:8] + 0.5 * (ln2**2) * y_pred_var[:, 4:8]) * w, axis=1) / denom

    expected_var = effect_var * lf_var + effect_var * (lf_exp**2) + lf_var * (effect_exp**2)
    expected_sigma = np.sqrt(expected_var)
    np.testing.assert_allclose(sigma, expected_sigma, rtol=1e-10, atol=1e-12)


def test_sfxi_v1_uncertainty_auto_defaults_to_analytical_when_beta_gamma_one():
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
    assert res.diagnostics["summary_stats"]["uncertainty_method"] == "analytical"


def test_sfxi_v1_uncertainty_auto_defaults_to_delta_when_exponents_not_one():
    y_pred = np.array([[0.1, 0.2, 0.15, 0.85, 0.3, 0.5, 0.2, 0.9]], dtype=float)
    y_pred_std = np.array([[0.02, 0.03, 0.02, 0.02, 0.05, 0.04, 0.05, 0.03]], dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.1,
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
    assert res.diagnostics["summary_stats"]["uncertainty_method"] == "delta"


def test_sfxi_v1_uncertainty_none_defaults_to_analytical_when_beta_gamma_one():
    y_pred = np.array([[0.1, 0.2, 0.15, 0.85, 0.3, 0.5, 0.2, 0.9]], dtype=float)
    y_pred_std = np.array([[0.02, 0.03, 0.02, 0.02, 0.05, 0.04, 0.05, 0.03]], dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "intensity_log2_offset_delta": 0.0,
        "uncertainty_method": None,
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)
    assert res.diagnostics["summary_stats"]["uncertainty_method"] == "analytical"


def test_sfxi_v1_uncertainty_rejects_unsupported_alias_string() -> None:
    y_pred = np.array([[0.1, 0.2, 0.15, 0.85, 0.3, 0.5, 0.2, 0.9]], dtype=float)
    y_pred_std = np.array([[0.02, 0.03, 0.02, 0.02, 0.05, 0.04, 0.05, 0.03]], dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "uncertainty_method": "auto",
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    with pytest.raises(ValueError, match="must be 'delta' or 'analytical'"):
        sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)


def test_sfxi_v1_uncertainty_analytical_all_off_is_finite_and_positive():
    y_pred = np.array(
        [
            [0.2, 0.1, 0.2, 0.1, 2.0, 2.0, 2.0, 2.0],
            [0.8, 0.7, 0.9, 0.8, 2.0, 2.0, 2.0, 2.0],
        ],
        dtype=float,
    )
    y_pred_std = np.array(
        [
            [0.08, 0.06, 0.08, 0.06, 0.02, 0.02, 0.02, 0.02],
            [0.08, 0.06, 0.08, 0.06, 0.02, 0.02, 0.02, 0.02],
        ],
        dtype=float,
    )
    params = {
        "setpoint_vector": [0, 0, 0, 0],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "uncertainty_method": "analytical",
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.empty((0, 8), dtype=float)
    train_R = np.empty((0,), dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)
    unc = np.asarray(res.uncertainty_by_name["sfxi"], dtype=float)
    assert np.all(np.isfinite(unc))
    assert np.all(unc > 0.0)


@pytest.mark.parametrize("method", ["delta", "analytical"])
def test_sfxi_v1_uncertainty_has_expected_shape_and_method_diagnostics(method: str) -> None:
    y_pred = np.array(
        [
            [0.1, 0.2, 0.15, 0.85, 0.3, 0.5, 0.2, 0.9],
            [0.4, 0.1, 0.3, 0.7, 0.25, 0.35, 0.45, 0.6],
        ],
        dtype=float,
    )
    y_pred_std = np.array(
        [
            [0.02, 0.03, 0.02, 0.02, 0.05, 0.04, 0.05, 0.03],
            [0.03, 0.02, 0.02, 0.03, 0.06, 0.04, 0.03, 0.05],
        ],
        dtype=float,
    )
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "intensity_log2_offset_delta": 0.0,
        "uncertainty_method": method,
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)
    unc = np.asarray(res.uncertainty_by_name["sfxi"], dtype=float).reshape(-1)
    assert unc.shape == (y_pred.shape[0],)
    assert np.all(np.isfinite(unc))
    assert np.all(unc > 0.0)
    assert res.diagnostics["summary_stats"]["uncertainty_method"] == method


def test_sfxi_v1_uncertainty_is_empty_without_std_input():
    y_pred = np.array(
        [
            [0.1, 0.2, 0.15, 0.85, 0.3, 0.5, 0.2, 0.9],
            [0.4, 0.1, 0.3, 0.7, 0.25, 0.35, 0.45, 0.6],
        ],
        dtype=float,
    )
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=None)
    assert res.uncertainty_by_name == {}


def test_sfxi_v1_uncertainty_analytical_c5666a7_regression_fixture():
    y_pred = np.array(
        [
            [0.1, 0.2, 0.15, 0.85, 0.3, 0.5, 0.2, 0.9],
            [0.4, 0.1, 0.3, 0.7, 0.25, 0.35, 0.45, 0.6],
        ],
        dtype=float,
    )
    y_pred_std = np.array(
        [
            [0.02, 0.03, 0.02, 0.02, 0.05, 0.04, 0.05, 0.03],
            [0.03, 0.02, 0.02, 0.03, 0.06, 0.04, 0.03, 0.05],
        ],
        dtype=float,
    )
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "intensity_log2_offset_delta": 0.0,
        "uncertainty_method": "analytical",
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)
    unc = np.asarray(res.uncertainty_by_name["sfxi"], dtype=float)
    expected = np.array([0.008229008782537631, 0.014461619845017327], dtype=float)
    np.testing.assert_allclose(unc, expected, rtol=1e-10, atol=1e-12)


def test_sfxi_v1_skips_uncertainty_method_validation_when_std_missing():
    y_pred = np.array([[0.1, 0.2, 0.15, 0.85, 0.3, 0.5, 0.2, 0.9]], dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.1,
        "intensity_exponent_gamma": 1.0,
        "intensity_log2_offset_delta": 0.0,
        "uncertainty_method": "analytical",
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=None)
    assert res.uncertainty_by_name == {}
    assert "uncertainty_method" not in res.diagnostics["summary_stats"]


def test_sfxi_v1_uncertainty_analytical_clipped_regime_matches_delta():
    y_pred = np.array([[2.5, -1.5, 1.7, 3.1, 8.0, 8.0, -8.0, 12.0]], dtype=float)
    y_pred_std = np.array([[0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1]], dtype=float)
    base_params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "intensity_log2_offset_delta": 0.0,
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx_delta = _ctx(as_of_round=0)
    octx_delta = rctx_delta.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    params_delta = dict(base_params)
    params_delta["uncertainty_method"] = "delta"
    res_delta = sfxi_v1(y_pred=y_pred, params=params_delta, ctx=octx_delta, train_view=tv, y_pred_std=y_pred_std)
    sigma_delta = float(np.asarray(res_delta.uncertainty_by_name["sfxi"], dtype=float)[0])

    rctx_analytical = _ctx(as_of_round=0)
    octx_analytical = rctx_analytical.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    params_analytical = dict(base_params)
    params_analytical["uncertainty_method"] = "analytical"
    res_analytical = sfxi_v1(
        y_pred=y_pred,
        params=params_analytical,
        ctx=octx_analytical,
        train_view=tv,
        y_pred_std=y_pred_std,
    )
    sigma_analytical = float(np.asarray(res_analytical.uncertainty_by_name["sfxi"], dtype=float)[0])

    assert np.isfinite(sigma_delta)
    assert np.isfinite(sigma_analytical)
    assert sigma_delta > 0.0
    assert sigma_analytical > 0.0


def test_sfxi_v1_uncertainty_delta_exact_logic_setpoint_fails_with_clear_error():
    y_pred = np.array([[0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]], dtype=float)
    y_pred_std = np.array([[0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 0],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "uncertainty_method": "delta",
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.empty((0, 8), dtype=float)
    train_R = np.empty((0,), dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    with pytest.raises(ValueError, match="delta uncertainty is undefined at exact logic setpoint"):
        sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)


def test_sfxi_v1_rejects_unstable_score_intensity_log2_range():
    y_pred = np.array([[0.1, 0.2, 0.15, 0.85, 2000.0, 2000.0, 2000.0, 2000.0]], dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    with pytest.raises(ValueError, match="stable score range"):
        sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=None)


@pytest.mark.parametrize("method", ["delta", "analytical"])
def test_sfxi_v1_rejects_unstable_uncertainty_intensity_log2_range(method: str):
    y_pred = np.array([[0.1, 0.2, 0.15, 0.85, 900.0, 900.0, 900.0, 900.0]], dtype=float)
    y_pred_std = np.array([[0.02, 0.03, 0.02, 0.02, 0.05, 0.04, 0.05, 0.03]], dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "intensity_log2_offset_delta": 0.0,
        "uncertainty_method": method,
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    with pytest.raises(ValueError, match="stable uncertainty range"):
        sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)


@pytest.mark.parametrize("method", ["delta", "analytical"])
def test_sfxi_v1_uncertainty_overflow_fails_fast(method: str):
    y_pred = np.array([[0.1, 0.2, 0.15, 0.85, 0.3, 0.5, 0.2, 0.9]], dtype=float)
    y_pred_std = np.full_like(y_pred, 1e200, dtype=float)
    params = {
        "setpoint_vector": [0, 0, 0, 1],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "intensity_log2_offset_delta": 0.0,
        "uncertainty_method": method,
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
    }
    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="variance contains non-finite values"):
            sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv, y_pred_std=y_pred_std)
