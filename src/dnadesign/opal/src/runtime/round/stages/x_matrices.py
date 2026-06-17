"""
Training and candidate X-matrix setup for an OPAL round.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from ....core.round_context import RoundCtx
from ....core.utils import OpalError, now_iso
from ....storage.artifacts import append_round_log_event
from ...memory_guard import enforce_x_matrix_memory_budget, infer_x_dim_from_series
from ..contracts import RoundInputs, XBundle
from .telemetry import log


def stage_x_matrices(
    *,
    inputs: RoundInputs,
    plan: Any,
    train_ids: List[str],
    Y_train: np.ndarray,
    tctx: Any,
    rctx: RoundCtx,
) -> XBundle:
    cfg = inputs.cfg
    req = inputs.req
    store = inputs.store
    df = inputs.df

    if len(train_ids) != len(set(train_ids)):
        if str(plan.training_dedup_policy).strip().lower() != "all_rounds":
            raise OpalError(
                "Duplicate ids detected in training labels. "
                "Set training.policy.label_cross_round_deduplication_policy='all_rounds' to allow."
            )
        seen = set()
        unique_ids = []
        for _id in train_ids:
            if _id not in seen:
                unique_ids.append(_id)
                seen.add(_id)
        X_unique, id_order_unique = store.transform_matrix_from_records(unique_ids, ctx=tctx)
        x_map = {i: X_unique[j] for j, i in enumerate(id_order_unique)}
        X_train = np.vstack([x_map[i] for i in train_ids])
        id_order_train = train_ids
    else:
        X_train, id_order_train = store.transform_matrix_from_records(train_ids, ctx=tctx)

    cand_df = plan.candidate_df
    if plan.selection_excludes_labeled and plan.candidate_filtered_out:
        log(
            req.verbose,
            f"[candidates] pool={plan.candidate_total_before_filter} → {len(cand_df)} after excluding already-labeled "
            f"(policy allow_resuggesting_candidates_until_labeled={plan.allow_resuggest})",
        )
    if plan.candidate_eligibility_reports:
        log(
            req.verbose,
            "[candidates] eligibility "
            f"{plan.candidate_total_before_eligibility} → {plan.candidate_total_before_filter} before scoring",
        )

    if cand_df.shape[0] == 0:
        raise OpalError("Candidate pool is empty after filtering; nothing to score.")

    if req.x_dim_override is not None:
        x_dim = int(req.x_dim_override)
    elif store.x_col in df.columns:
        x_dim = infer_x_dim_from_series(df[store.x_col], x_column=store.x_col)
    else:
        raise OpalError(
            f"Missing X dimension for streaming scoring. Validate records.parquet X column '{store.x_col}' first."
        )
    sbatch = int(req.score_batch_size_override or cfg.scoring.score_batch_size)
    if sbatch <= 0:
        raise OpalError("score_batch_size must be a positive integer.")
    candidate_batch_rows = min(int(len(cand_df)), int(sbatch))
    memory_estimate = enforce_x_matrix_memory_budget(
        row_count=int(len(train_ids) + candidate_batch_rows),
        x_dim=int(x_dim),
        item_size_bytes=max(8, int(req.x_item_size_bytes or 8)),
        max_gib=req.max_x_matrix_gib_override
        if req.max_x_matrix_gib_override is not None
        else cfg.safety.max_x_matrix_gib,
        context="OPAL round streaming X batch",
    )
    append_round_log_event(
        inputs.rdir / "logs" / "round.log.jsonl",
        {
            "ts": now_iso(),
            "round": int(req.as_of_round),
            "stage": "x_memory_guard_done",
            "scope": "streaming_score_batch",
            "candidate_rows": int(len(cand_df)),
            "score_batch_size": int(sbatch),
            "rows": int(memory_estimate.row_count),
            "x_dim": int(memory_estimate.x_dim),
            "raw_gib": float(memory_estimate.raw_gib),
            "estimated_gib": float(memory_estimate.estimated_gib),
            "max_gib": float(memory_estimate.max_gib),
        },
    )
    id_order_pool = cand_df["id"].astype(str).tolist()
    rctx.set_core("core/data/x_dim", int(X_train.shape[1]))
    rctx.set_core("core/data/n_scored", int(len(id_order_pool)))
    rctx.set_core("core/data/candidate_pool_total", int(plan.candidate_total_before_filter))
    rctx.set_core("core/data/candidate_pool_filtered_out", int(plan.candidate_filtered_out))
    rctx.set_core("core/data/candidate_pool_total_before_eligibility", int(plan.candidate_total_before_eligibility))
    rctx.set_core("core/data/candidate_eligibility_filtered_out", int(plan.candidate_eligibility_filtered_out))
    for report in plan.candidate_eligibility_reports:
        append_round_log_event(
            inputs.rdir / "logs" / "round.log.jsonl",
            {
                "ts": now_iso(),
                "round": int(req.as_of_round),
                "stage": "candidate_eligibility_done",
                **dict(report),
            },
        )
    if X_train.shape[0] != Y_train.shape[0]:
        raise OpalError(f"Training X/Y row mismatch: X_train={X_train.shape[0]} Y_train={Y_train.shape[0]}.")
    log(
        req.verbose,
        f"[transform] X_train: {X_train.shape} for {len(id_order_train)} ids | "
        f"X_pool: streaming batches up to {sbatch} rows for {len(id_order_pool)} ids",
    )
    return XBundle(
        X_train=X_train,
        id_order_train=id_order_train,
        id_order_pool=id_order_pool,
        cand_df=cand_df,
    )
