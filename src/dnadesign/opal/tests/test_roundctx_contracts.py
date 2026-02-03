"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_roundctx_contracts.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from dnadesign.opal.src.config.types import PluginRef
from dnadesign.opal.src.core.round_context import (
    PluginRegistryView,
    RoundCtx,
    RoundCtxContractError,
    roundctx_contract,
)
from dnadesign.opal.src.registries.models import (
    get_model,
    list_models,
    register_model,
)
from dnadesign.opal.src.registries.objectives import (
    get_objective,
    list_objectives,
    register_objective,
)
from dnadesign.opal.src.registries.selection import (
    get_selection,
    list_selections,
    register_selection,
)
from dnadesign.opal.src.registries.transforms_x import (
    get_transform_x,
    list_transforms_x,
    register_transform_x,
)
from dnadesign.opal.src.registries.transforms_y import (
    get_transform_y,
    list_transforms_y,
    list_y_ops,
    register_transform_y,
    register_y_op,
    run_y_ops_pipeline,
)


class _ObjResult:
    def __init__(self, score: np.ndarray, diagnostics: Dict[str, Any] | None = None) -> None:
        self.score = score
        self.scalar_uncertainty = None
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class _StageContract:
    category: str
    requires: Tuple[str, ...]
    produces: Tuple[str, ...]
    requires_by_stage: Dict[str, Tuple[str, ...]]
    produces_by_stage: Dict[str, Tuple[str, ...]]


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
        def _sel(
            *,
            ids,
            scores,
            top_k,
            objective="maximize",
            tie_handling="competition_rank",
            ctx=None,
            **_,
        ):
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
        def _sel_missing(
            *,
            ids,
            scores,
            top_k,
            objective="maximize",
            tie_handling="competition_rank",
            ctx=None,
            **_,
        ):
            return {"order_idx": np.arange(len(ids))}


def _ensure_objective_stage_contract() -> None:
    if "test_obj_stage_contract" not in list_objectives():

        def _obj(*, y_pred, params, ctx=None, train_view=None):
            if ctx is not None:
                _ = ctx.get("core/labels_as_of_round")
                ctx.set("objective/<self>/stage_key", True)
            return _ObjResult(score=np.ones(len(y_pred)), diagnostics={})

        _obj.__opal_contract__ = _StageContract(
            category="objective",
            requires=tuple(),
            produces=tuple(),
            requires_by_stage={"objective": ("core/labels_as_of_round",)},
            produces_by_stage={"objective": ("objective/<self>/stage_key",)},
        )
        register_objective("test_obj_stage_contract")(_obj)


def _ensure_selection_stage_contract() -> None:
    if "test_sel_stage_contract" not in list_selections():

        def _sel(
            *,
            ids,
            scores,
            top_k,
            objective="maximize",
            tie_handling="competition_rank",
            ctx=None,
            **_,
        ):
            if ctx is not None:
                _ = ctx.get("core/data/n_scored")
                ctx.set("selection/<self>/stage_note", "ok")
            return {"order_idx": np.arange(len(ids))}

        _sel.__opal_contract__ = _StageContract(
            category="selection",
            requires=tuple(),
            produces=tuple(),
            requires_by_stage={"selection": ("core/data/n_scored",)},
            produces_by_stage={"selection": ("selection/<self>/stage_note",)},
        )
        register_selection("test_sel_stage_contract")(_sel)


def _ensure_transform_x_stage_contract() -> None:
    if "test_transform_x_stage" not in list_transforms_x():

        def _factory(params: Dict[str, Any]):
            def _tx(series, ctx=None):
                if ctx is not None:
                    _ = ctx.get("core/round_index")
                    ctx.set("transform_x/<self>/stage_seen", True)
                return np.zeros((len(series), 1), dtype=float)

            return _tx

        _factory.__opal_contract__ = _StageContract(
            category="transform_x",
            requires=tuple(),
            produces=tuple(),
            requires_by_stage={"transform_x": ("core/round_index",)},
            produces_by_stage={"transform_x": ("transform_x/<self>/stage_seen",)},
        )
        register_transform_x("test_transform_x_stage")(_factory)


def _ensure_transform_y_stage_contract() -> None:
    if "test_transform_y_stage" not in list_transforms_y():

        def _ty(df_tidy, params, ctx=None):
            if ctx is not None:
                _ = ctx.get("core/labels_as_of_round")
                ctx.set("transform_y/<self>/stage_seen", True)
            return df_tidy[["id", "y"]]

        _ty.__opal_contract__ = _StageContract(
            category="transform_y",
            requires=tuple(),
            produces=tuple(),
            requires_by_stage={"transform_y": ("core/labels_as_of_round",)},
            produces_by_stage={"transform_y": ("transform_y/<self>/stage_seen",)},
        )
        register_transform_y("test_transform_y_stage")(_ty)


def _ensure_model_stage_contract() -> None:
    if "test_model_stage" not in list_models():

        @register_model("test_model_stage")
        class _Model:
            def __init__(self, params: Dict[str, Any]) -> None:
                self._predict_calls = 0
                self._predict_calls_target = int(params.get("predict_calls_target", 2))

            def fit(self, X: np.ndarray, Y: np.ndarray, *, ctx=None):
                if ctx is not None:
                    ctx.set("model/<self>/x_dim", int(X.shape[1]))
                return None

            def predict(self, X: np.ndarray, *, ctx=None) -> np.ndarray:
                self._predict_calls += 1
                if ctx is not None and self._predict_calls >= self._predict_calls_target:
                    ctx.set("model/<self>/predict_summary", {"calls": int(self._predict_calls)})
                return np.zeros((X.shape[0], 1), dtype=float)

        _Model.__opal_contract__ = _StageContract(
            category="model",
            requires=tuple(),
            produces=tuple(),
            requires_by_stage={"fit": tuple(), "predict": tuple()},
            produces_by_stage={
                "fit": ("model/<self>/x_dim",),
                "predict": ("model/<self>/predict_summary",),
            },
        )


class _Params(BaseModel):
    pass


def _ensure_yops_contracts() -> None:
    if "test_yop_contract" not in list_y_ops():

        @register_y_op(
            "test_yop_contract",
            requires=["yops/<self>/need"],
            produces=["yops/<self>/made"],
        )
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

        @register_y_op(
            "test_yop_no_produce",
            requires=["yops/<self>/need"],
            produces=["yops/<self>/made"],
        )
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


def test_roundctx_contract_accepts_stage_maps():
    try:

        @roundctx_contract(
            category="model",
            requires_by_stage={"fit": ["core/round_index"]},
            produces_by_stage={"fit": ["model/<self>/x_dim"]},
        )
        def _dummy():
            return None

    except TypeError as exc:
        pytest.fail(f"roundctx_contract should accept stage maps: {exc}")


def test_pluginctx_stage_requires_and_produces():
    class _Dummy:
        pass

    dummy = _Dummy()
    dummy.__opal_contract__ = _StageContract(
        category="model",
        requires=tuple(),
        produces=tuple(),
        requires_by_stage={"fit": ("core/round_index",)},
        produces_by_stage={"fit": ("model/<self>/x_dim",)},
    )
    rctx = _rctx({"core/round_index": 0})
    mctx = rctx.for_plugin(category="model", name="dummy", plugin=dummy)
    try:
        mctx.precheck_requires(stage="fit")
    except TypeError as exc:
        pytest.fail(f"precheck_requires should accept stage: {exc}")
    except RoundCtxContractError as exc:
        pytest.fail(f"precheck_requires should use stage requires: {exc}")
    try:
        mctx.set("model/<self>/x_dim", 1)
    except RoundCtxContractError as exc:
        pytest.fail(f"stage produces should be settable: {exc}")
    try:
        mctx.postcheck_produces(stage="fit")
    except TypeError as exc:
        pytest.fail(f"postcheck_produces should accept stage: {exc}")
    except RoundCtxContractError as exc:
        pytest.fail(f"postcheck_produces should accept stage produces: {exc}")


def test_model_predict_stage_postcheck_deferred():
    _ensure_model_stage_contract()
    model = get_model("test_model_stage", {"predict_calls_target": 2})
    rctx = _rctx({"core/round_index": 0})
    mctx = rctx.for_plugin(category="model", name="test_model_stage", plugin=model)
    X = np.zeros((3, 2), dtype=float)
    Y = np.zeros((3, 1), dtype=float)
    model.fit(X, Y, ctx=mctx)
    try:
        model.predict(X[:1], ctx=mctx)
        try:
            mctx.postcheck_produces(stage="predict")
            pytest.fail("predict produces should not pass before final batch")
        except RoundCtxContractError:
            pass
        model.predict(X[1:], ctx=mctx)
    except RoundCtxContractError as exc:
        pytest.fail(f"predict should not enforce produces per batch: {exc}")
    try:
        mctx.postcheck_produces(stage="predict")
    except TypeError as exc:
        pytest.fail(f"postcheck_produces should accept stage: {exc}")
    except RoundCtxContractError as exc:
        pytest.fail(f"predict produces should pass at stage end: {exc}")


def test_objective_stage_contract_via_registry():
    _ensure_objective_stage_contract()
    obj = get_objective("test_obj_stage_contract")
    rctx = _rctx({"core/labels_as_of_round": 0})
    octx = rctx.for_plugin(category="objective", name="test_obj_stage_contract", plugin=obj)
    try:
        obj(y_pred=np.zeros((2, 1)), params={}, ctx=octx, train_view=None)
    except RoundCtxContractError as exc:
        pytest.fail(f"objective stage contracts should pass: {exc}")
    assert rctx.get("objective/test_obj_stage_contract/stage_key") is True


def test_selection_stage_contract_via_registry():
    _ensure_selection_stage_contract()
    sel = get_selection("test_sel_stage_contract", {})
    rctx = _rctx({"core/data/n_scored": 2})
    sctx = rctx.for_plugin(category="selection", name="test_sel_stage_contract", plugin=sel)
    try:
        sel(ids=np.array(["a", "b"]), scores=np.array([1.0, 0.5]), top_k=1, ctx=sctx)
    except RoundCtxContractError as exc:
        pytest.fail(f"selection stage contracts should pass: {exc}")
    assert rctx.get("selection/test_sel_stage_contract/stage_note") == "ok"


def test_transform_x_stage_contract_via_registry():
    _ensure_transform_x_stage_contract()
    tx = get_transform_x("test_transform_x_stage", {})
    rctx = _rctx({"core/round_index": 0})
    tctx = rctx.for_plugin(category="transform_x", name="test_transform_x_stage", plugin=tx)
    series = pd.Series(["a", "b"])
    out = tx(series, ctx=tctx)
    assert out.shape == (2, 1)
    assert rctx.get("transform_x/test_transform_x_stage/stage_seen") is True


def test_transform_y_stage_contract_via_registry():
    _ensure_transform_y_stage_contract()
    ty = get_transform_y("test_transform_y_stage")
    rctx = _rctx({"core/labels_as_of_round": 0})
    tctx = rctx.for_plugin(category="transform_y", name="test_transform_y_stage", plugin=ty)
    df_tidy = pd.DataFrame({"id": ["a", "b"], "y": [[0.1], [0.2]]})
    out = ty(df_tidy, {}, ctx=tctx)
    assert out["id"].tolist() == ["a", "b"]
    assert rctx.get("transform_y/test_transform_y_stage/stage_seen") is True
