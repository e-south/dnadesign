"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_roundctx_contracts.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
from pydantic import BaseModel

from dnadesign.opal.src.config.types import PluginRef
from dnadesign.opal.src.registries.objectives import get_objective, list_objectives, register_objective
from dnadesign.opal.src.registries.selections import get_selection, list_selections, register_selection
from dnadesign.opal.src.registries.transforms_y import list_y_ops, register_y_op, run_y_ops_pipeline
from dnadesign.opal.src.round_context import PluginRegistryView, RoundCtx, RoundCtxContractError, roundctx_contract


class _ObjResult:
    def __init__(self, score: np.ndarray, diagnostics: Dict[str, Any] | None = None) -> None:
        self.score = score
        self.scalar_uncertainty = None
        self.diagnostics = diagnostics or {}


def _ensure_objective_contracts() -> None:
    if "test_obj_contract" not in list_objectives():

        @roundctx_contract(
            category="objective",
            requires=["core/labels_as_of_round"],
            produces=["objective/<self>/foo"],
        )
        @register_objective("test_obj_contract")
        def _obj(*, y_pred, params, ctx=None, train_view=None):
            if ctx is not None:
                _ = ctx.get("core/labels_as_of_round")
                ctx.set("objective/<self>/foo", {"ok": True})
            return _ObjResult(score=np.ones(len(y_pred)), diagnostics={})

    if "test_obj_no_produce" not in list_objectives():

        @roundctx_contract(
            category="objective",
            requires=["core/labels_as_of_round"],
            produces=["objective/<self>/bar"],
        )
        @register_objective("test_obj_no_produce")
        def _obj_missing(*, y_pred, params, ctx=None, train_view=None):
            # Intentionally do not set required key
            return _ObjResult(score=np.ones(len(y_pred)), diagnostics={})


def _ensure_selection_contracts() -> None:
    if "test_sel_contract" not in list_selections():

        @roundctx_contract(
            category="selection",
            requires=["core/data/n_scored"],
            produces=["selection/<self>/note"],
        )
        @register_selection("test_sel_contract")
        def _sel(*, ids, scores, top_k, objective="maximize", tie_handling="competition_rank", ctx=None, **_):
            if ctx is not None:
                _ = ctx.get("core/data/n_scored")
                ctx.set("selection/<self>/note", "ok")
            return {"order_idx": np.arange(len(ids))}

    if "test_sel_no_produce" not in list_selections():

        @roundctx_contract(
            category="selection",
            requires=["core/data/n_scored"],
            produces=["selection/<self>/note"],
        )
        @register_selection("test_sel_no_produce")
        def _sel_missing(*, ids, scores, top_k, objective="maximize", tie_handling="competition_rank", ctx=None, **_):
            return {"order_idx": np.arange(len(ids))}


class _Params(BaseModel):
    pass


def _ensure_yops_contracts() -> None:
    if "test_yop_contract" not in list_y_ops():

        @register_y_op("test_yop_contract", requires=["yops/<self>/need"], produces=["yops/<self>/made"])
        def _yop():
            def fit(Y, params, ctx=None):
                if ctx is not None:
                    _ = ctx.get("yops/<self>/need")
                    ctx.set("yops/<self>/made", True)

            def transform(Y, params, ctx=None):
                return np.asarray(Y, dtype=float)

            def inverse(Y, params, ctx=None):
                return np.asarray(Y, dtype=float)

            return fit, transform, inverse, _Params

    if "test_yop_no_produce" not in list_y_ops():

        @register_y_op("test_yop_no_produce", requires=["yops/<self>/need"], produces=["yops/<self>/made"])
        def _yop_missing():
            def fit(Y, params, ctx=None):
                # Intentionally do not set required key
                return None

            def transform(Y, params, ctx=None):
                return np.asarray(Y, dtype=float)

            def inverse(Y, params, ctx=None):
                return np.asarray(Y, dtype=float)

            return fit, transform, inverse, _Params


def _rctx(core: Dict[str, Any]) -> RoundCtx:
    reg = PluginRegistryView("rf", "sfxi_v1", "top_n", "identity", "sfxi_vec8_from_table_v1")
    core_only = {k: v for k, v in core.items() if k.startswith("core/")}
    ctx = RoundCtx(core=core_only, registry=reg)
    for k, v in core.items():
        if not k.startswith("core/"):
            ctx.set(k, v)
    return ctx


def test_objective_contract_requires_and_audit():
    _ensure_objective_contracts()
    obj = get_objective("test_obj_contract")
    rctx = _rctx({"core/labels_as_of_round": 0})
    octx = rctx.for_plugin(category="objective", name="test_obj_contract", plugin=obj)
    obj(y_pred=np.zeros((2, 1)), params={}, ctx=octx, train_view=None)
    snap = rctx.snapshot()
    produced = snap.get("core/contracts/objective/test_obj_contract/produced", [])
    assert "objective/test_obj_contract/foo" in produced


def test_objective_contract_missing_requires():
    _ensure_objective_contracts()
    obj = get_objective("test_obj_contract")
    rctx = _rctx({})
    octx = rctx.for_plugin(category="objective", name="test_obj_contract", plugin=obj)
    with pytest.raises(RoundCtxContractError):
        obj(y_pred=np.zeros((1, 1)), params={}, ctx=octx, train_view=None)


def test_objective_contract_missing_produces():
    _ensure_objective_contracts()
    obj = get_objective("test_obj_no_produce")
    rctx = _rctx({"core/labels_as_of_round": 0})
    octx = rctx.for_plugin(category="objective", name="test_obj_no_produce", plugin=obj)
    with pytest.raises(RoundCtxContractError):
        obj(y_pred=np.zeros((1, 1)), params={}, ctx=octx, train_view=None)


def test_selection_contract_requires_and_audit():
    _ensure_selection_contracts()
    sel = get_selection("test_sel_contract", {})
    rctx = _rctx({"core/data/n_scored": 2})
    sctx = rctx.for_plugin(category="selection", name="test_sel_contract", plugin=sel)
    out = sel(ids=np.array(["a", "b"]), scores=np.array([1.0, 0.5]), top_k=1, ctx=sctx)
    assert "order_idx" in out
    snap = rctx.snapshot()
    produced = snap.get("core/contracts/selection/test_sel_contract/produced", [])
    assert "selection/test_sel_contract/note" in produced


def test_selection_contract_missing_produces():
    _ensure_selection_contracts()
    sel = get_selection("test_sel_no_produce", {})
    rctx = _rctx({"core/data/n_scored": 1})
    sctx = rctx.for_plugin(category="selection", name="test_sel_no_produce", plugin=sel)
    with pytest.raises(RoundCtxContractError):
        sel(ids=np.array(["a"]), scores=np.array([1.0]), top_k=1, ctx=sctx)


def test_yops_contract_requires_and_produces():
    _ensure_yops_contracts()
    rctx = _rctx({"yops/test_yop_contract/need": True})
    Y = np.array([[1.0], [2.0]])
    ops = [PluginRef("test_yop_contract", {})]
    out = run_y_ops_pipeline(stage="fit_transform", y_ops=ops, Y=Y, ctx=rctx)
    assert np.allclose(out, Y)
    snap = rctx.snapshot()
    produced = snap.get("core/contracts/yops/test_yop_contract/produced", [])
    assert "yops/test_yop_contract/made" in produced


def test_yops_contract_missing_requires():
    _ensure_yops_contracts()
    rctx = _rctx({})
    Y = np.array([[1.0]])
    ops = [PluginRef("test_yop_contract", {})]
    with pytest.raises(RoundCtxContractError):
        run_y_ops_pipeline(stage="fit_transform", y_ops=ops, Y=Y, ctx=rctx)


def test_yops_contract_missing_produces():
    _ensure_yops_contracts()
    rctx = _rctx({"yops/test_yop_no_produce/need": True})
    Y = np.array([[1.0]])
    ops = [PluginRef("test_yop_no_produce", {})]
    with pytest.raises(RoundCtxContractError):
        run_y_ops_pipeline(stage="fit_transform", y_ops=ops, Y=Y, ctx=rctx)
