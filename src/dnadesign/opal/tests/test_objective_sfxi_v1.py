"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_objective_sfxi_v1.py

Unit tests for the sfxi_v1 objective.

Module Author(s): Eric J. South (extended by Codex)
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.opal.src.core.round_context import PluginRegistryView, RoundCtx
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
    res = sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv)

    # denom = 2, logic fidelity = 1, so scores should be [1.0, 0.5].
    assert np.allclose(res.score, np.array([1.0, 0.5], dtype=float))

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
        sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv)


def test_sfxi_v1_rejects_out_of_range_setpoint():
    y_pred = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    params = {"setpoint_vector": [0.0, -0.2, 0.0, 1.2]}

    train_Y = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    train_R = np.array([0], dtype=int)
    tv = _TrainView(train_Y, train_R, as_of_round=0)

    rctx = _ctx(as_of_round=0)
    octx = rctx.for_plugin(category="objective", name="sfxi_v1", plugin=sfxi_v1)
    with pytest.raises(ValueError, match="setpoint_vector"):
        sfxi_v1(y_pred=y_pred, params=params, ctx=octx, train_view=tv)
