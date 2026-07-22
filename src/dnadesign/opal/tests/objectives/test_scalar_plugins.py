"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/objectives/test_scalar_plugins.py

Regression tests for scalar plugins OPAL objectives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.opal.src.config.plugin_schemas import validate_params
from dnadesign.opal.src.core.objective_result import validate_objective_result_v2
from dnadesign.opal.src.core.round_context import PluginRegistryView, RoundCtx
from dnadesign.opal.src.objectives.scalar_identity_v1 import scalar_identity_v1
from dnadesign.opal.src.objectives.spop_v1 import (
    SPOP_NUMERIC_SCOPE,
    SPOP_OBJECTIVE_NAME,
    SPOP_READER_METRIC_ID,
    SPOP_SCORE_CHANNEL,
    spop_v1,
)
from dnadesign.opal.src.registries.objectives import get_objective_declared_channels, list_objectives
from dnadesign.opal.src.registries.transforms_y import get_transform_y


def _tx_ctx(name: str):
    reg = PluginRegistryView("model", "objective", "selection", "transform_x", name)
    rctx = RoundCtx(core={"core/round_index": 0}, registry=reg)
    fn = get_transform_y(name)
    return fn, rctx.for_plugin(category="transform_y", name=name, plugin=fn)


def test_scalar_from_table_basic() -> None:
    df = pd.DataFrame({"sequence": ["AAA", "BBB"], "y": [0.1, 0.2]})
    fn, ctx = _tx_ctx("scalar_from_table_v1")
    out = fn(df, {"sequence_column": "sequence", "y_column": "y"}, ctx=ctx)
    assert list(out.columns) == ["sequence", "y"]
    assert out["y"].tolist() == [[0.1], [0.2]]


def test_scalar_from_table_includes_id() -> None:
    df = pd.DataFrame({"id": ["a", "b"], "sequence": ["AAA", "BBB"], "y": [0.1, 0.2]})
    fn, ctx = _tx_ctx("scalar_from_table_v1")
    out = fn(df, {"id_column": "id", "sequence_column": "sequence", "y_column": "y"}, ctx=ctx)
    assert list(out.columns) == ["id", "sequence", "y"]
    assert out["id"].tolist() == ["a", "b"]


def test_scalar_from_table_rejects_missing_columns() -> None:
    df = pd.DataFrame({"sequence": ["AAA"]})
    fn, ctx = _tx_ctx("scalar_from_table_v1")
    with pytest.raises(ValueError):
        fn(df, {"sequence_column": "sequence", "y_column": "y"}, ctx=ctx)


def test_scalar_from_table_rejects_non_finite() -> None:
    df = pd.DataFrame({"sequence": ["AAA"], "y": [np.nan]})
    fn, ctx = _tx_ctx("scalar_from_table_v1")
    with pytest.raises(ValueError):
        fn(df, {"sequence_column": "sequence", "y_column": "y"}, ctx=ctx)


def test_scalar_identity_objective_scores() -> None:
    y_pred = np.array([[0.1], [0.2]])
    res = scalar_identity_v1(y_pred=y_pred, params={}, ctx=None, train_view=None, y_pred_std=None)
    assert res.scores_by_name["scalar"].tolist() == [0.1, 0.2]
    assert "summary_stats" in res.diagnostics


def test_scalar_identity_objective_rejects_shape() -> None:
    with pytest.raises(ValueError):
        scalar_identity_v1(y_pred=np.array([0.1, 0.2]), params={}, ctx=None, train_view=None, y_pred_std=None)


def test_spop_objective_emits_first_class_score_channel() -> None:
    result = spop_v1(
        y_pred=np.array([[0.1], [0.8], [-0.2]], dtype=float),
        params={},
        ctx=None,
        train_view=None,
        y_pred_std=None,
    )

    checked = validate_objective_result_v2(result=result, objective_name=SPOP_OBJECTIVE_NAME, n_rows=3)
    assert checked.scores_by_name[SPOP_SCORE_CHANNEL].tolist() == [0.1, 0.8, -0.2]
    assert checked.modes_by_name == {SPOP_SCORE_CHANNEL: "maximize"}
    assert checked.diagnostics["metric_id"] == SPOP_READER_METRIC_ID
    assert checked.diagnostics["numeric_scope"] == SPOP_NUMERIC_SCOPE
    assert checked.diagnostics["negative_prediction_count"] == 1


def test_spop_objective_is_registered_with_declared_channel_contract() -> None:
    assert SPOP_OBJECTIVE_NAME in list_objectives()
    assert get_objective_declared_channels(SPOP_OBJECTIVE_NAME) == {
        "score": (SPOP_SCORE_CHANNEL,),
        "uncertainty": (),
        "score_modes": {SPOP_SCORE_CHANNEL: "maximize"},
    }
    assert validate_params("objective", SPOP_OBJECTIVE_NAME, {}) == {}


def test_spop_objective_rejects_non_scalar_or_parametrized_inputs() -> None:
    with pytest.raises(ValueError, match="params must be empty"):
        spop_v1(
            y_pred=np.array([[0.1]], dtype=float),
            params={"metric_id": SPOP_READER_METRIC_ID},
            ctx=None,
            train_view=None,
            y_pred_std=None,
        )
    with pytest.raises(ValueError, match="Expected y_pred with 1 column"):
        spop_v1(
            y_pred=np.array([[0.1, 0.2]], dtype=float),
            params={},
            ctx=None,
            train_view=None,
            y_pred_std=None,
        )
    with pytest.raises(ValueError, match="non-finite"):
        spop_v1(
            y_pred=np.array([[float("nan")]], dtype=float),
            params={},
            ctx=None,
            train_view=None,
            y_pred_std=None,
        )
