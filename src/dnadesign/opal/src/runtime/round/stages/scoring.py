"""
Round scoring stage orchestration.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from ....core.round_context import RoundCtx
from ..contracts import RoundInputs, ScoreBundle
from .objectives import evaluate_objectives
from .prediction import fit_and_predict
from .selection import select_candidates


def stage_scoring(
    *,
    inputs: RoundInputs,
    rctx: RoundCtx,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    R_train: np.ndarray,
    tctx: Any,
    id_order_train: List[str],
    id_order_pool: List[str],
    y_dim: int,
) -> ScoreBundle:
    prediction = fit_and_predict(
        inputs=inputs,
        rctx=rctx,
        X_train=X_train,
        Y_train=Y_train,
        tctx=tctx,
        id_order_train=id_order_train,
        id_order_pool=id_order_pool,
        y_dim=y_dim,
    )
    objectives = evaluate_objectives(
        inputs=inputs,
        rctx=rctx,
        Y_hat=prediction.Y_hat,
        y_pred_std=prediction.y_pred_std,
        Y_train=Y_train,
        R_train=R_train,
        id_order_pool=id_order_pool,
    )
    selection = select_candidates(
        inputs=inputs,
        rctx=rctx,
        id_order_pool=id_order_pool,
        objectives=objectives,
    )

    return ScoreBundle(
        model=prediction.model,
        fit_metrics=prediction.fit_metrics,
        fit_duration=prediction.fit_duration,
        Y_hat=prediction.Y_hat,
        y_obj_scalar=selection.y_obj_scalar,
        diag=selection.diag,
        obj_summary_stats=selection.obj_summary_stats,
        obj_name=selection.obj_name,
        obj_params=selection.obj_params,
        obj_mode=selection.obj_mode,
        objective_defs=objectives.objective_defs,
        score_channels=objectives.score_channels,
        uncertainty_channels=objectives.uncertainty_channels,
        score_ref=selection.score_ref,
        uncertainty_ref=selection.uncertainty_ref,
        sel_name=selection.sel_name,
        sel_params=selection.sel_params,
        tie_handling=selection.tie_handling,
        mode=selection.mode,
        ranks_competition=selection.ranks_competition,
        selected_bool=selection.selected_bool,
        selected_effective=selection.selected_effective,
        top_k=selection.top_k,
        obj_sha=selection.obj_sha,
        scores=selection.scores,
        uq_scalar=selection.uq_scalar,
    )
