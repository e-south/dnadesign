"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/runtime/test_selection_views.py

Runtime contracts for one-fit, multi-view campaign rounds.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dnadesign.opal.src.config.types import PluginRef, SelectionView
from dnadesign.opal.src.core.round_context import PluginRegistryView, RoundCtx
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.runtime.round.stages.objectives import evaluate_objectives
from dnadesign.opal.src.runtime.round.stages.selection import build_selection_batch, select_candidates


def _round_ctx() -> RoundCtx:
    registry = PluginRegistryView(
        model="random_forest",
        objective="selection_views",
        selection="selection_views",
        transform_x="identity",
        transform_y="scalar_from_table_v1",
    )
    return RoundCtx(
        core={"core/run_id": "r0-test", "core/round_index": 0, "core/labels_as_of_round": 0},
        registry=registry,
    )


def _view(view_id: str, *, require_exact_top_k: bool = False) -> SelectionView:
    return SelectionView(
        id=view_id,
        objective=PluginRef("scalar_identity_v1", {}),
        selection=PluginRef(
            "top_n",
            {
                "top_k": 1,
                "score_ref": "scalar",
                "objective_mode": "maximize",
                "tie_handling": "competition_rank",
                "require_exact_top_k": require_exact_top_k,
            },
        ),
    )


def test_selection_view_can_require_exact_cardinality_at_tied_boundary(tmp_path) -> None:
    cfg = SimpleNamespace(selection_views=[_view("target_a", require_exact_top_k=True)])
    inputs = SimpleNamespace(
        cfg=cfg,
        req=SimpleNamespace(as_of_round=0, verbose=False, k_override=None),
        rdir=tmp_path,
    )
    objectives = evaluate_objectives(
        inputs=inputs,
        rctx=_round_ctx(),
        Y_hat=np.asarray([[0.2], [0.2], [0.1]], dtype=float),
        y_pred_std=None,
        Y_train=np.asarray([[0.0]], dtype=float),
        R_train=np.asarray([0], dtype=int),
        id_order_pool=["a", "b", "c"],
    )

    with pytest.raises(
        OpalError,
        match="requires exactly 1 selected candidate, but competition_rank selected 2",
    ):
        select_candidates(
            inputs=inputs,
            rctx=_round_ctx(),
            id_order_pool=["a", "b", "c"],
            objectives=objectives,
        )


def test_objective_channels_are_namespaced_by_selection_view(tmp_path) -> None:
    cfg = SimpleNamespace(selection_views=[_view("target_a"), _view("target_b")])
    inputs = SimpleNamespace(
        cfg=cfg,
        req=SimpleNamespace(as_of_round=0, verbose=False),
        rdir=tmp_path,
    )

    result = evaluate_objectives(
        inputs=inputs,
        rctx=_round_ctx(),
        Y_hat=np.asarray([[0.1], [0.2]], dtype=float),
        y_pred_std=None,
        Y_train=np.asarray([[0.0]], dtype=float),
        R_train=np.asarray([0], dtype=int),
        id_order_pool=["a", "b"],
    )

    assert sorted(result.score_channels) == ["target_a/scalar", "target_b/scalar"]
    assert [item["selection_view_id"] for item in result.objective_defs] == ["target_a", "target_b"]
    assert [item["objective_name"] for item in result.objective_defs] == [
        "scalar_identity_v1",
        "scalar_identity_v1",
    ]


def test_each_selection_view_selects_from_its_namespaced_channel(tmp_path) -> None:
    views = [_view("target_a"), _view("target_b")]
    cfg = SimpleNamespace(selection_views=views)
    inputs = SimpleNamespace(
        cfg=cfg,
        req=SimpleNamespace(as_of_round=0, verbose=False, k_override=None),
        rdir=tmp_path,
    )
    rctx = _round_ctx()
    objectives = evaluate_objectives(
        inputs=inputs,
        rctx=rctx,
        Y_hat=np.asarray([[0.1], [0.2]], dtype=float),
        y_pred_std=None,
        Y_train=np.asarray([[0.0]], dtype=float),
        R_train=np.asarray([0], dtype=int),
        id_order_pool=["a", "b"],
    )

    selections = select_candidates(
        inputs=inputs,
        rctx=rctx,
        id_order_pool=["a", "b"],
        objectives=objectives,
    )

    assert sorted(selections) == ["target_a", "target_b"]
    for view_id, result in selections.items():
        assert result.selection_view_id == view_id
        assert result.score_ref == f"{view_id}/scalar"
        assert result.selected_effective == 1
        assert result.selected_bool.tolist() == [False, True]


def test_selection_batch_unions_memberships_without_silent_fill(tmp_path) -> None:
    views = [_view("target_a"), _view("target_b")]
    cfg = SimpleNamespace(selection_views=views)
    inputs = SimpleNamespace(
        cfg=cfg,
        req=SimpleNamespace(as_of_round=0, verbose=False, k_override=None),
        rdir=tmp_path,
    )
    rctx = _round_ctx()
    objectives = evaluate_objectives(
        inputs=inputs,
        rctx=rctx,
        Y_hat=np.asarray([[0.1], [0.2]], dtype=float),
        y_pred_std=None,
        Y_train=np.asarray([[0.0]], dtype=float),
        R_train=np.asarray([0], dtype=int),
        id_order_pool=["a", "b"],
    )
    selections = select_candidates(
        inputs=inputs,
        rctx=rctx,
        id_order_pool=["a", "b"],
        objectives=objectives,
    )
    candidates = pd.DataFrame({"id": ["a", "b"], "sequence": ["AAA", "BBB"]})

    batch = build_selection_batch(
        candidate_df=candidates,
        id_order_pool=["a", "b"],
        selections=selections,
        deduplicate_by="sequence",
        expected_unique_count=1,
    )

    assert batch.unique_count == 1
    assert batch.rows.iloc[0]["id"] == "b"
    assert batch.rows.iloc[0]["selection_view_ids"] == ["target_a", "target_b"]

    with pytest.raises(OpalError, match="expected 2 unique candidates, observed 1"):
        build_selection_batch(
            candidate_df=candidates,
            id_order_pool=["a", "b"],
            selections=selections,
            deduplicate_by="sequence",
            expected_unique_count=2,
        )


def test_selection_batch_preserves_objective_and_selector_scores(tmp_path) -> None:
    cfg = SimpleNamespace(selection_views=[_view("target_a")])
    inputs = SimpleNamespace(
        cfg=cfg,
        req=SimpleNamespace(as_of_round=0, verbose=False, k_override=None),
        rdir=tmp_path,
    )
    rctx = _round_ctx()
    objectives = evaluate_objectives(
        inputs=inputs,
        rctx=rctx,
        Y_hat=np.asarray([[0.1], [0.2]], dtype=float),
        y_pred_std=None,
        Y_train=np.asarray([[0.0]], dtype=float),
        R_train=np.asarray([0], dtype=int),
        id_order_pool=["a", "b"],
    )
    selections = select_candidates(
        inputs=inputs,
        rctx=rctx,
        id_order_pool=["a", "b"],
        objectives=objectives,
    )
    selections["target_a"] = replace(
        selections["target_a"],
        scores=np.asarray([0.4, 0.9], dtype=float),
    )

    batch = build_selection_batch(
        candidate_df=pd.DataFrame({"id": ["a", "b"], "sequence": ["AAA", "BBB"]}),
        id_order_pool=["a", "b"],
        selections=selections,
        deduplicate_by="sequence",
        expected_unique_count=1,
    )

    membership = batch.rows.iloc[0]["selection_memberships"][0]
    assert membership["score"] == pytest.approx(0.2)
    assert membership["selection_score"] == pytest.approx(0.9)
