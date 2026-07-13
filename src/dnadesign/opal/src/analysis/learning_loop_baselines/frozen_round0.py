"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/learning_loop_baselines/frozen_round0.py

OPAL-owned frozen round-0 scoring primitives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from ...config import load_config
from ...core.utils import OpalError
from ...registries.models import get_model
from ...registries.transforms_x import get_transform_x
from ...registries.transforms_y import run_y_ops_pipeline
from ...runtime.round.context import build_round_ctx
from ...runtime.round.stages.objectives import evaluate_objectives
from ...runtime.round.stages.prediction import inverse_yops_outputs
from ...runtime.round.stages.selection import resolve_channel_ref
from ...runtime.round_plan import plan_round
from ...storage.label_sources import label_source_from_config
from ...storage.store_factory import records_store_from_config


def frozen_round0_scores(
    config_path: str | Path,
    *,
    selection_view_id: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Score a campaign candidate pool from the model trained only on round-0 labels."""

    cfg = load_config(Path(config_path))
    view = _selection_view(cfg, selection_view_id=selection_view_id)
    _validate_supported_selection(view)
    store = records_store_from_config(cfg)
    df = store.load_runtime_frame(include_x=False)
    label_source = label_source_from_config(cfg, store)
    label_source.validate(df)
    plan = plan_round(store, df, cfg, 0, label_source=label_source)
    if plan.training_df.empty:
        raise ValueError(f"Frozen replay found no round-0 labels for {config_path}")

    train_ids = plan.training_df["id"].astype(str).tolist()
    candidate_ids = plan.candidate_df["id"].astype(str).tolist()
    if not candidate_ids:
        raise ValueError(f"Frozen replay candidate pool is empty for {config_path}")
    if set(train_ids) & set(candidate_ids):
        raise ValueError(f"Frozen replay ranking contains initial labeled IDs for {config_path}")

    y_train = np.stack(plan.training_df["y"].map(lambda value: np.asarray(value, dtype=float)).to_list(), axis=0)
    r_train = plan.training_df["r"].astype(int).to_numpy()
    y_dim = int(y_train.shape[1])
    run_id, _, rctx = build_round_ctx(cfg=cfg, as_of_round=0, y_dim=y_dim, n_train=len(train_ids))
    tx = get_transform_x(cfg.data.transforms_x.name, cfg.data.transforms_x.params)
    tctx = rctx.for_plugin(category="transform_x", name=cfg.data.transforms_x.name, plugin=tx)
    x_train, id_order_train = store.transform_matrix_from_records(train_ids, ctx=tctx)
    if id_order_train != train_ids:
        raise ValueError(f"Frozen replay training ID order changed for {config_path}")

    model = get_model(cfg.model.name, cfg.model.params)
    mctx = rctx.for_plugin(category="model", name=cfg.model.name, plugin=model)
    y_train_fit = run_y_ops_pipeline(stage="fit_transform", y_ops=cfg.training.y_ops or [], Y=y_train, ctx=rctx)
    model.fit(x_train, y_train_fit, ctx=mctx)
    y_hat_fit, _ = _predict_candidate_pool(
        store=store,
        model=model,
        model_ctx=mctx,
        transform_ctx=tctx,
        candidate_ids=candidate_ids,
        selection_view_id=selection_view_id,
        batch_size=int(cfg.scoring.score_batch_size),
    )
    y_hat, _ = inverse_yops_outputs(rctx=rctx, y_ops_cfg=cfg.training.y_ops or [], y_pred=y_hat_fit, y_pred_std=None)
    score = _objective_scores(
        cfg=cfg,
        rctx=rctx,
        y_hat=y_hat,
        y_train=y_train,
        r_train=r_train,
        candidate_ids=candidate_ids,
    )
    scores = pd.DataFrame(
        {
            "id": candidate_ids,
            "score": score,
            "run_id": run_id,
            "campaign_config_path": str(config_path),
        }
    )
    if scores["id"].duplicated().any():
        raise ValueError(f"Frozen replay score table contains duplicate candidate IDs for {config_path}")
    return scores, train_ids


def _selection_view(cfg: Any, *, selection_view_id: str) -> Any:
    matches = [view for view in cfg.selection_views if view.id == selection_view_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate selection view {selection_view_id!r}")
    return matches[0]


def _validate_supported_selection(view: Any) -> None:
    selection = view.selection
    if str(selection.name) != "top_n":
        raise ValueError(f"Frozen replay currently supports top_n selection only; got {selection.name!r}")
    params = dict(selection.params)
    if str(params.get("objective_mode") or "maximize") != "maximize":
        raise ValueError("Frozen replay currently supports maximize top_n objectives only")
    if not bool(params.get("exclude_already_labeled", True)):
        raise ValueError("Frozen replay requires selection.exclude_already_labeled=true")


def _predict_candidate_pool(
    *,
    store: Any,
    model: Any,
    model_ctx: Any,
    transform_ctx: Any,
    candidate_ids: list[str],
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    chunks: list[np.ndarray] = []
    predicted_ids: list[str] = []
    for batch_x, batch_ids in store.iter_transform_matrix_batches(
        candidate_ids,
        ctx=transform_ctx,
        batch_size=batch_size,
    ):
        chunks.append(model.predict(batch_x, ctx=model_ctx))
        predicted_ids.extend(batch_ids)
    y_hat = np.vstack(chunks) if chunks else np.zeros((0, 1), dtype=float)
    return _align_predictions(y_hat=y_hat, predicted_ids=predicted_ids, requested_ids=candidate_ids)


def _align_predictions(
    *,
    y_hat: np.ndarray,
    predicted_ids: list[str],
    requested_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    predicted = [str(value) for value in predicted_ids]
    requested = [str(value) for value in requested_ids]
    if np.asarray(y_hat).shape[0] != len(predicted):
        raise OpalError(
            f"Frozen replay prediction row count mismatch: {np.asarray(y_hat).shape[0]} vs {len(predicted)}"
        )
    if len(predicted) != len(set(predicted)):
        raise OpalError("Frozen replay prediction stream produced duplicate ids")
    if set(predicted) != set(requested):
        missing = sorted(set(requested) - set(predicted))
        extra = sorted(set(predicted) - set(requested))
        raise OpalError(f"Frozen replay prediction id mismatch: missing={missing[:5]} extra={extra[:5]}")
    if predicted == requested:
        return np.asarray(y_hat, dtype=float), requested
    positions = {row_id: index for index, row_id in enumerate(predicted)}
    aligned = np.asarray(y_hat, dtype=float)[[positions[row_id] for row_id in requested]]
    return aligned, requested


def _objective_scores(
    *,
    cfg: Any,
    rctx: Any,
    y_hat: np.ndarray,
    y_train: np.ndarray,
    r_train: np.ndarray,
    candidate_ids: list[str],
    selection_view_id: str,
) -> np.ndarray:
    req = SimpleNamespace(as_of_round=0, verbose=False)
    inputs = SimpleNamespace(cfg=cfg, req=req)
    objectives = evaluate_objectives(
        inputs=inputs,
        rctx=rctx,
        Y_hat=y_hat,
        y_pred_std=None,
        Y_train=y_train,
        R_train=r_train,
        id_order_pool=candidate_ids,
    )
    view = _selection_view(cfg, selection_view_id=selection_view_id)
    score_ref = f"{selection_view_id}/{str(view.selection.params.get('score_ref') or '')}"
    score = resolve_channel_ref(score_ref, objectives.score_channels, label="score_ref")
    if not np.all(np.isfinite(score)):
        raise ValueError("Frozen replay objective scores contain non-finite values")
    return np.asarray(score, dtype=float)
