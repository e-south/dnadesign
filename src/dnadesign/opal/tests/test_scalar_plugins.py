"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_scalar_plugins.py

Unit tests for scalar Y ingest transform + objective.

Module Author(s): Eric J. South (extended by Codex)
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.opal.src.core.round_context import PluginRegistryView, RoundCtx
from dnadesign.opal.src.objectives.scalar_identity_v1 import scalar_identity_v1
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
    res = scalar_identity_v1(y_pred=y_pred, params={})
    assert res.score.tolist() == [0.1, 0.2]
    assert "summary_stats" in res.diagnostics


def test_scalar_identity_objective_rejects_shape() -> None:
    with pytest.raises(ValueError):
        scalar_identity_v1(y_pred=np.array([0.1, 0.2]), params={})
