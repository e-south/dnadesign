"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/runtime/round/stages/scoring.py

Round scoring stage orchestration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from ....core.round_context import RoundCtx
from ..contracts import RoundInputs, ScoreBundle
from .objectives import evaluate_objectives
from .prediction import fit_and_predict
from .selection import build_selection_batch, select_candidates


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
    candidate_df,
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
    batch = build_selection_batch(
        candidate_df=candidate_df,
        id_order_pool=id_order_pool,
        selections=selection,
        deduplicate_by=inputs.cfg.selection_batch.deduplicate_by,
        expected_unique_count=inputs.cfg.selection_batch.expected_unique_count,
    )
    objective_meta_sha = next(iter(selection.values())).obj_sha

    return ScoreBundle(
        model=prediction.model,
        fit_metrics=prediction.fit_metrics,
        fit_duration=prediction.fit_duration,
        Y_hat=prediction.Y_hat,
        objective_defs=objectives.objective_defs,
        score_channels=objectives.score_channels,
        uncertainty_channels=objectives.uncertainty_channels,
        selections=selection,
        selection_batch=batch,
        objective_meta_sha=objective_meta_sha,
    )
