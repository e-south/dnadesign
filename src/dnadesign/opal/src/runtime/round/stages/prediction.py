"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/runtime/round/stages/prediction.py

Model fitting, batched prediction, and Y-op inversion for an OPAL round.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from ....core.progress import NullProgress
from ....core.round_context import RoundCtx
from ....core.utils import OpalError, now_iso
from ....registries.models import get_model
from ....registries.transforms_y import run_y_ops_pipeline
from ....storage.artifacts import append_round_log_event
from ..contracts import RoundInputs
from .telemetry import log
from .y_ops import coalesce_uncertainty_chunks, inverse_yops_outputs


@dataclass(frozen=True)
class PredictionBundle:
    model: Any
    fit_metrics: Any
    fit_duration: float
    Y_hat: np.ndarray
    y_pred_std: np.ndarray | None


def _predict_batch_total(estimated_batches: int, batch_index: int) -> int:
    return max(int(estimated_batches), int(batch_index))


def _align_predictions_to_requested_order(
    *,
    y_hat: np.ndarray,
    y_pred_std: np.ndarray | None,
    predicted_ids: List[str],
    requested_ids: List[str],
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    predicted = [str(row_id) for row_id in predicted_ids]
    requested = [str(row_id) for row_id in requested_ids]
    if np.asarray(y_hat).shape[0] != len(predicted):
        raise OpalError(
            f"Streaming prediction row count mismatch; expected {len(predicted)}, got {np.asarray(y_hat).shape[0]}."
        )
    if len(predicted) != len(set(predicted)):
        raise OpalError("Streaming prediction produced duplicate ids; aborting before writing selection artifacts.")
    if len(requested) != len(set(requested)):
        raise OpalError("Streaming prediction requested duplicate ids; aborting before writing selection artifacts.")
    predicted_set = set(predicted)
    requested_set = set(requested)
    if predicted_set != requested_set:
        missing = sorted(requested_set - predicted_set)
        extra = sorted(predicted_set - requested_set)
        detail = []
        if missing:
            detail.append(f"missing={missing[:10]}")
        if extra:
            detail.append(f"extra={extra[:10]}")
        raise OpalError(
            "Streaming prediction id coverage mismatch; aborting before writing selection artifacts"
            + (f" ({'; '.join(detail)})." if detail else ".")
        )
    if predicted == requested:
        return y_hat, y_pred_std, False

    position = {row_id: index for index, row_id in enumerate(predicted)}
    reorder_index = np.asarray([position[row_id] for row_id in requested], dtype=int)
    aligned_y_hat = np.asarray(y_hat)[reorder_index]
    aligned_std = None
    if y_pred_std is not None:
        std = np.asarray(y_pred_std)
        if std.shape[0] != len(predicted):
            raise OpalError(
                f"Streaming prediction uncertainty row count mismatch; expected {len(predicted)}, got {std.shape[0]}."
            )
        aligned_std = std[reorder_index]
    return aligned_y_hat, aligned_std, True


def fit_and_predict(
    *,
    inputs: RoundInputs,
    rctx: RoundCtx,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    tctx: Any,
    id_order_train: List[str],
    id_order_pool: List[str],
    y_dim: int,
) -> PredictionBundle:
    cfg = inputs.cfg
    req = inputs.req
    rdir = inputs.rdir
    round_index = int(req.as_of_round)
    run_id = str(rctx.get("core/run_id", default=""))

    yops_cfg = getattr(cfg.training, "y_ops", []) or []
    log(req.verbose, f"[y-ops] applying {len(yops_cfg)} op(s) to training labels: {([p.name for p in yops_cfg] or [])}")
    append_round_log_event(
        rdir / "logs" / "round.log.jsonl",
        {
            "ts": now_iso(),
            "round": round_index,
            "run_id": run_id,
            "stage": "yops_fit_transform",
            "ops": [p.name for p in yops_cfg],
        },
    )
    Y_train_fit = run_y_ops_pipeline(stage="fit_transform", y_ops=yops_cfg, Y=Y_train, ctx=rctx)

    tfit0 = time.perf_counter()
    model = get_model(cfg.model.name, cfg.model.params)
    mctx = rctx.for_plugin(category="model", name=cfg.model.name, plugin=model)
    append_round_log_event(
        rdir / "logs" / "round.log.jsonl",
        {
            "ts": now_iso(),
            "round": round_index,
            "run_id": run_id,
            "stage": "fit_start",
            "model": cfg.model.name,
            "n_train": len(id_order_train),
        },
    )
    fit_metrics = model.fit(X_train, Y_train_fit, ctx=mctx)
    fit_duration = float(time.perf_counter() - tfit0)
    log(
        req.verbose,
        f"[fit] model={cfg.model.name} | dt={fit_duration:.3f}s | oob_r2={getattr(fit_metrics, 'oob_r2', None)}",
    )

    sbatch = int(req.score_batch_size_override or cfg.scoring.score_batch_size)
    if sbatch <= 0:
        raise OpalError("score_batch_size must be a positive integer.")
    yhat_chunks: List[np.ndarray] = []
    predicted_ids: List[str] = []
    total = len(id_order_pool)
    num_batches = max(1, (total + max(1, sbatch) - 1) // max(1, sbatch))

    progress_factory = req.progress_factory if (req.verbose and req.progress_factory) else None
    factory = progress_factory or (lambda desc, total_rows: NullProgress())
    with factory("predict", int(total)) as prog:
        for bi, (batch_X, batch_ids) in enumerate(
            inputs.store.iter_transform_matrix_batches(id_order_pool, ctx=tctx, batch_size=max(1, sbatch))
        ):
            if not cfg.safety.accept_x_mismatch and X_train.shape[1] != batch_X.shape[1]:
                raise OpalError(
                    "X dimension mismatch between training and pool "
                    f"(train={X_train.shape[1]}, pool={batch_X.shape[1]}). "
                    "Set safety.accept_x_mismatch=true to override."
                )
            yhat_chunks.append(model.predict(batch_X, ctx=mctx))
            predicted_ids.extend(batch_ids)
            prog.advance(int(batch_X.shape[0]))
            append_round_log_event(
                rdir / "logs" / "round.log.jsonl",
                {
                    "ts": now_iso(),
                    "round": round_index,
                    "run_id": run_id,
                    "stage": "predict_batch",
                    "batch": int(bi + 1),
                    # Parquet batch filtering can yield more chunks than the
                    # selected-id lower bound; never log an impossible total.
                    "of": _predict_batch_total(num_batches, bi + 1),
                    "rows": int(batch_X.shape[0]),
                },
            )
    Y_hat_fit = np.vstack(yhat_chunks) if yhat_chunks else np.zeros((0, y_dim), dtype=float)

    contract = getattr(model, "__opal_contract__", None)
    produces_by_stage = getattr(contract, "produces_by_stage", None) or {}
    if produces_by_stage.get("predict"):
        mctx.postcheck_produces(stage="predict")

    missing = object()
    std_payload = mctx.get("model/<self>/std_devs", missing)
    y_pred_std_fit = None if std_payload is missing else coalesce_uncertainty_chunks(std_payload)
    Y_hat_fit, y_pred_std_fit, reordered = _align_predictions_to_requested_order(
        y_hat=Y_hat_fit,
        y_pred_std=y_pred_std_fit,
        predicted_ids=predicted_ids,
        requested_ids=list(map(str, id_order_pool)),
    )
    if reordered:
        append_round_log_event(
            rdir / "logs" / "round.log.jsonl",
            {
                "ts": now_iso(),
                "round": round_index,
                "run_id": run_id,
                "stage": "prediction_reordered",
                "rows": int(Y_hat_fit.shape[0]),
            },
        )

    log(req.verbose, f"[y-ops] inverting {len(yops_cfg)} op(s) for predictions: {([p.name for p in yops_cfg] or [])}")
    Y_hat, y_pred_std = inverse_yops_outputs(
        rctx=rctx,
        y_ops_cfg=yops_cfg,
        y_pred=Y_hat_fit,
        y_pred_std=y_pred_std_fit,
    )
    append_round_log_event(
        rdir / "logs" / "round.log.jsonl",
        {
            "ts": now_iso(),
            "round": round_index,
            "run_id": run_id,
            "stage": "yops_inverse_done",
            "ops": [p.name for p in yops_cfg],
        },
    )
    if Y_hat.shape[1] != y_dim:
        raise OpalError(f"Predicted Y dimension mismatch: expected {y_dim}, got {Y_hat.shape[1]}")

    rctx.set_core("core/labels_as_of_round", int(req.as_of_round))
    return PredictionBundle(
        model=model,
        fit_metrics=fit_metrics,
        fit_duration=fit_duration,
        Y_hat=Y_hat,
        y_pred_std=y_pred_std,
    )
