"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/objectives/test_sfxi_uncertainty_hardening.py

Hardening tests for SFXI objective uncertainty semantics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.opal.src.core.round_context import PluginRegistryView, RoundCtx
from dnadesign.opal.src.objectives.sfxi_v1 import sfxi_v1


class _TrainView:
    def __init__(self, y: np.ndarray) -> None:
        self._y = np.asarray(y, dtype=float)

    def iter_labels_y_current_round(self):
        yield from self._y


def test_sfxi_uncertainty_defaults_to_delta() -> None:
    result = _score_with_uncertainty(
        y_pred=np.asarray([[0.1, 0.2, 0.15, 0.85, 0.3, 0.5, 0.2, 0.9]]),
        y_pred_std=np.full((1, 8), 0.03),
        params={},
    )

    assert result.diagnostics["summary_stats"]["uncertainty_method"] == "delta"


def test_sfxi_uncertainty_rejects_removed_analytical_approximation() -> None:
    with pytest.raises(ValueError, match="analytical.*not supported"):
        _score_with_uncertainty(
            y_pred=np.asarray([[0.1, 0.2, 0.15, 0.85, 0.3, 0.5, 0.2, 0.9]]),
            y_pred_std=np.full((1, 8), 0.03),
            params={"uncertainty_method": "analytical"},
        )


def test_sfxi_delta_uncertainty_does_not_cross_saturated_effect_clip() -> None:
    result = _score_with_uncertainty(
        y_pred=np.asarray([[0.2, 0.2, 0.2, 0.8, 0.0, 0.0, 0.0, 10.0]]),
        y_pred_std=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03]]),
        params={
            "uncertainty_method": "delta",
            "logic_exponent_beta": 0.0,
            "intensity_exponent_gamma": 1.0,
        },
    )

    assert result.uncertainty_by_name["sfxi"][0] == 0.0


def test_sfxi_delta_uncertainty_matches_local_monte_carlo() -> None:
    y_pred = np.asarray([[0.2, 0.15, 0.1, 0.8, 0.0, 0.0, 0.0, -1.0]])
    y_pred_std = np.asarray([[0.005, 0.005, 0.005, 0.005, 0.0, 0.0, 0.0, 0.005]])
    result = _score_with_uncertainty(y_pred=y_pred, y_pred_std=y_pred_std, params={})
    delta_sigma = float(result.uncertainty_by_name["sfxi"][0])

    rng = np.random.default_rng(20260710)
    draws = rng.normal(loc=y_pred, scale=y_pred_std, size=(50_000, 8))
    logic = np.clip(draws[:, 0:4], 0.0, 1.0)
    setpoint = np.asarray([0.0, 0.0, 0.0, 1.0])
    logic_fidelity = np.clip(1.0 - np.linalg.norm(logic - setpoint, axis=1) / 2.0, 0.0, 1.0)
    effect_scaled = np.clip(np.power(2.0, draws[:, 7]), 0.0, 1.0)
    monte_carlo_sigma = float(np.std(logic_fidelity * effect_scaled, ddof=1))

    assert delta_sigma == pytest.approx(monte_carlo_sigma, rel=0.03)


def _score_with_uncertainty(
    *,
    y_pred: np.ndarray,
    y_pred_std: np.ndarray,
    params: dict[str, object],
):
    objective_params = {
        "setpoint_vector": [0.0, 0.0, 0.0, 1.0],
        "logic_exponent_beta": 1.0,
        "intensity_exponent_gamma": 1.0,
        "intensity_log2_offset_delta": 0.0,
        "scaling": {"percentile": 95, "min_n": 1, "eps": 1.0e-8},
        **params,
    }
    train = _TrainView(np.asarray([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]))
    registry = PluginRegistryView("rf", "sfxi_v1", "top_n", "identity", "sfxi_vec8_from_table_v1")
    round_ctx = RoundCtx(core={"core/labels_as_of_round": 0}, registry=registry)
    objective_ctx = round_ctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    return sfxi_v1(
        y_pred=y_pred,
        params=objective_params,
        ctx=objective_ctx,
        train_view=train,
        y_pred_std=y_pred_std,
    )
