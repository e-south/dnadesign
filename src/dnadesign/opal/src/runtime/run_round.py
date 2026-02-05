"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/runtime/run_round.py

Executes one Opal round from training through selection and writebacks.
Coordinates round stages, artifacts, ledgers, and state updates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..core.utils import OpalError, ensure_dir, now_iso, print_stderr
from ..registries.transforms_x import get_transform_x
from ..storage.artifacts import append_round_log_event
from ..storage.data_access import RecordsStore
from ..storage.workspace import CampaignWorkspace
from .round.context import build_round_ctx
from .round.contracts import RoundInputs, RunRoundRequest, RunRoundResult
from .round.stages import stage_scoring, stage_training, stage_x_matrices
from .round.writebacks import (
    append_ledgers,
    build_run_events,
    update_campaign_state,
    write_prediction_label_hist,
    write_round_artifacts,
)


def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print_stderr(msg)


def _stage_training(
    *,
    store: RecordsStore,
    df: pd.DataFrame,
    cfg: RootConfig,
    req: RunRoundRequest,
) -> _TrainingBundle:
    # Preflight (no auto-backfill)
    rep = preflight_run(
        store,
        df,
        int(req.as_of_round),
        cfg.safety.fail_on_mixed_biotype_or_alphabet,
        auto_backfill=False,
    )
    if getattr(rep, "manual_attach_count", 0):
        raise OpalError(
            f"Detected {rep.manual_attach_count} labels in '{store.y_col}' without label_hist. "
            "Run `opal ingest-y` (preferred) or `opal label-hist attach-from-y` for legacy Y columns."
        )
    store.validate_label_hist(df, require=True)

    plan = plan_round(store, df, cfg, int(req.as_of_round), warnings=list(rep.warnings or []))
    train_df = plan.training_df
    if train_df.empty:
        raise OpalError(f"No labels ≤ round {req.as_of_round} for training.")

    train_ids = train_df["id"].astype(str).tolist()
    Y_train = np.stack(train_df["y"].map(lambda v: np.asarray(v, dtype=float)).to_list(), axis=0)
    R_train = train_df["r"].astype(int).to_numpy()
    y_dim = int(Y_train.shape[1])
    exp_len = cfg.data.y_expected_length
    if exp_len is not None and y_dim != int(exp_len):
        raise OpalError(f"Y length mismatch: expected {int(exp_len)}, got {y_dim}")

    _log(
        req.verbose,
        f"[labels] as_of_round={int(req.as_of_round)} | n_train={len(train_df)} | y_dim={y_dim} | "
        f"policy(cumulative={bool(plan.training_policy.get('cumulative_training', True))}, "
        f"dedup={plan.training_dedup_policy})",
    )
    return _TrainingBundle(
        rep=rep,
        plan=plan,
        train_df=train_df,
        train_ids=train_ids,
        Y_train=Y_train,
        R_train=R_train,
        y_dim=y_dim,
    )


def _build_round_ctx(
    *,
    cfg: RootConfig,
    as_of_round: int,
    y_dim: int,
    n_train: int,
) -> tuple[str, PluginRegistryView, RoundCtx]:
    run_id = f"r{int(as_of_round)}-{now_iso()}"
    reg = PluginRegistryView(
        model=cfg.model.name,
        objective=cfg.objective.objective.name,
        selection=cfg.selection.selection.name,
        transform_x=cfg.data.transforms_x.name,
        transform_y=cfg.data.transforms_y.name,
    )
    rctx = RoundCtx(
        core={
            "core/run_id": run_id,
            "core/round_index": int(as_of_round),
            "core/campaign_slug": cfg.campaign.slug,
            "core/labels_as_of_round": int(as_of_round),
            "core/plugins/transforms_x/name": reg.transform_x,
            "core/plugins/transforms_y/name": reg.transform_y,
            "core/plugins/model/name": reg.model,
            "core/plugins/objective/name": reg.objective,
            "core/plugins/selection/name": reg.selection,
            "core/data/y_dim": int(y_dim),
            "core/data/n_train": int(n_train),
        },
        registry=reg,
    )
    return run_id, reg, rctx


def _stage_x_matrices(
    *,
    store: RecordsStore,
    df: pd.DataFrame,
    plan: Any,
    train_ids: List[str],
    Y_train: np.ndarray,
    cfg: RootConfig,
    req: RunRoundRequest,
    tctx: Any,
    rctx: RoundCtx,
) -> _XBundle:
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
        X_unique, id_order_unique = store.transform_matrix(df, unique_ids, ctx=tctx)
        x_map = {i: X_unique[j] for j, i in enumerate(id_order_unique)}
        X_train = np.vstack([x_map[i] for i in train_ids])
        id_order_train = train_ids
    else:
        X_train, id_order_train = store.transform_matrix(df, train_ids, ctx=tctx)

    cand_df = plan.candidate_df
    if plan.selection_excludes_labeled and plan.candidate_filtered_out:
        _log(
            req.verbose,
            f"[candidates] pool={plan.candidate_total_before_filter} → {len(cand_df)} after excluding already-labeled "
            f"(policy allow_resuggesting_candidates_until_labeled={plan.allow_resuggest})",
        )

    X_pool, id_order_pool = store.transform_matrix(df, cand_df["id"], ctx=tctx)
    if X_pool.shape[0] == 0:
        raise OpalError("Candidate pool is empty after filtering; nothing to score.")
    rctx.set_core("core/data/x_dim", int(X_train.shape[1]))
    rctx.set_core("core/data/n_scored", int(len(id_order_pool)))
    rctx.set_core("core/data/candidate_pool_total", int(plan.candidate_total_before_filter))
    rctx.set_core("core/data/candidate_pool_filtered_out", int(plan.candidate_filtered_out))
    if X_train.shape[0] != Y_train.shape[0]:
        raise OpalError(f"Training X/Y row mismatch: X_train={X_train.shape[0]} Y_train={Y_train.shape[0]}.")
    if not cfg.safety.accept_x_mismatch and X_train.shape[1] != X_pool.shape[1]:
        raise OpalError(
            f"X dimension mismatch between training and pool (train={X_train.shape[1]}, pool={X_pool.shape[1]}). "
            "Set safety.accept_x_mismatch=true to override."
        )
    _log(
        req.verbose,
        f"[transform] X_train: {X_train.shape} for {len(id_order_train)} ids | "
        f"X_pool: {X_pool.shape} for {len(id_order_pool)} ids",
    )
    return _XBundle(
        X_train=X_train,
        id_order_train=id_order_train,
        X_pool=X_pool,
        id_order_pool=id_order_pool,
        cand_df=cand_df,
    )


def _stage_fit_predict_score(
    *,
    cfg: RootConfig,
    req: RunRoundRequest,
    rdir: Path,
    rctx: RoundCtx,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    R_train: np.ndarray,
    X_pool: np.ndarray,
    id_order_train: List[str],
    id_order_pool: List[str],
    y_dim: int,
) -> _ScoreBundle:
    # --- Y-ops (fit/transform) ---
    yops_cfg = getattr(cfg.training, "y_ops", []) or []
    _log(
        req.verbose,
        f"[y-ops] applying {len(yops_cfg)} op(s) to training labels: {([p.name for p in yops_cfg] or [])}",
    )
    append_round_log_event(
        rdir / "round.log.jsonl",
        {
            "ts": now_iso(),
            "stage": "yops_fit_transform",
            "ops": [p.name for p in yops_cfg],
        },
    )
    Y_train_fit = run_y_ops_pipeline(stage="fit_transform", y_ops=yops_cfg, Y=Y_train, ctx=rctx)

    # --- Model fit ---
    tfit0 = time.perf_counter()
    model = get_model(cfg.model.name, cfg.model.params)
    mctx = rctx.for_plugin(category="model", name=cfg.model.name, plugin=model)
    append_round_log_event(
        rdir / "round.log.jsonl",
        {
            "ts": now_iso(),
            "stage": "fit_start",
            "model": cfg.model.name,
            "n_train": len(id_order_train),
        },
    )
    fit_metrics = model.fit(X_train, Y_train_fit, ctx=mctx)
    tfit1 = time.perf_counter()
    fit_duration = float(tfit1 - tfit0)
    _log(
        req.verbose,
        f"[fit] model={cfg.model.name} | dt={fit_duration:.3f}s | oob_r2={getattr(fit_metrics, 'oob_r2', None)}",
    )

    # --- Predict candidates (batched) ---
    sbatch = int(req.score_batch_size_override or cfg.scoring.score_batch_size)
    if sbatch <= 0:
        raise OpalError("score_batch_size must be a positive integer.")
    yhat_chunks: List[np.ndarray] = []
    total = X_pool.shape[0]
    num_batches = max(1, (total + max(1, sbatch) - 1) // max(1, sbatch))

    progress_factory = req.progress_factory if (req.verbose and req.progress_factory) else None
    factory = progress_factory or (lambda desc, total_rows: NullProgress())
    with factory("predict", int(total)) as prog:
        for bi, i in enumerate(range(0, total, max(1, sbatch))):
            sl = slice(i, i + sbatch)
            batch_X = X_pool[sl]
            yhat_chunks.append(model.predict(batch_X, ctx=mctx))
            prog.advance(int(batch_X.shape[0]))
            append_round_log_event(
                rdir / "round.log.jsonl",
                {
                    "ts": now_iso(),
                    "stage": "predict_batch",
                    "batch": int(bi + 1),
                    "of": int(num_batches),
                    "rows": int(batch_X.shape[0]),
                },
            )
    Y_hat_fit = np.vstack(yhat_chunks) if yhat_chunks else np.zeros((0, y_dim), dtype=float)

    # --- Inverse Y-ops to objective space ---
    _log(
        req.verbose,
        f"[y-ops] inverting {len(yops_cfg)} op(s) for predictions: {([p.name for p in yops_cfg] or [])}",
    )
    Y_hat = run_y_ops_pipeline(stage="inverse", y_ops=yops_cfg, Y=Y_hat_fit, ctx=rctx)
    append_round_log_event(
        rdir / "round.log.jsonl",
        {
            "ts": now_iso(),
            "stage": "yops_inverse_done",
            "ops": [p.name for p in yops_cfg],
        },
    )
    if Y_hat.shape[1] != y_dim:
        raise OpalError(f"Predicted Y dimension mismatch: expected {y_dim}, got {Y_hat.shape[1]}")
    if cfg.selection.selection.name == "expected_improvement":
        Y_hat = np.asarray([Y_hat, std_devs])

    # --- Objective ---
    rctx.set_core("core/labels_as_of_round", int(req.as_of_round))
    obj_name = cfg.objective.objective.name
    obj_params = dict(cfg.objective.objective.params)
    obj_fn = get_objective(obj_name)
    octx = rctx.for_plugin(category="objective", name=obj_name, plugin=obj_fn)

    class _TrainView:
        def __init__(self, Y: np.ndarray, R: np.ndarray, as_of_round: int) -> None:
            self._Y = np.asarray(Y, dtype=float)
            self._R = np.asarray(R, dtype=int)
            self._as = int(as_of_round)

        def labels_count(self) -> int:
            return int(self._Y.shape[0])

        def iter_labels_y(self):
            for i in range(self._Y.shape[0]):
                yield self._Y[i, :]

        # Prefer current round's labels for round-calibrated denominator
        def iter_labels_y_current_round(self):
            mask = self._R == self._as
            for i in np.where(mask)[0].tolist():
                yield self._Y[i, :]

    tv = _TrainView(Y_train, R_train, int(req.as_of_round))
    obj_res = obj_fn(y_pred=Y_hat, params=obj_params, ctx=octx, train_view=tv)
    y_obj_scalar = np.asarray(obj_res.score, dtype=float).ravel()
    non_finite_mask = ~np.isfinite(y_obj_scalar)
    if non_finite_mask.any():
        n = int(non_finite_mask.sum())
        n_total = int(y_obj_scalar.size)
        n_nan = int(np.isnan(y_obj_scalar).sum())
        n_pinf = int(np.isposinf(y_obj_scalar).sum())
        n_ninf = int(np.isneginf(y_obj_scalar).sum())
        bad_ids = np.array(id_order_pool)[non_finite_mask]
        preview_pairs = [f"{bad_ids[i]}={y_obj_scalar[non_finite_mask][i]:.6g}" for i in range(min(20, n))]
        raise OpalError(
            "Objective produced non-finite scores for {n}/{N} candidates "
            "(NaN={na}, +Inf={pi}, -Inf={ni}). Sample: {pv}. "
            "Ensure objective/Y-ops produce finite scalars.".format(
                n=n,
                N=n_total,
                na=n_nan,
                pi=n_pinf,
                ni=n_ninf,
                pv=", ".join(preview_pairs),
            )
        )
    diag = obj_res.diagnostics or {}
    obj_mode = getattr(obj_res, "mode", "maximize")

    obj_meta = {
        "mode": obj_mode,
        "params": obj_params,
        "diagnostics_summary_keys": list(diag.keys()),
    }
    obj_sha = write_objective_meta(rdir / "objective_meta.json", obj_meta)

    obj_summary_stats = diag.get("summary_stats") if isinstance(diag, dict) else None
    if isinstance(obj_summary_stats, dict) and obj_summary_stats:
        kvs: List[str] = []
        for k in sorted(obj_summary_stats.keys()):
            v = obj_summary_stats[k]
            try:
                kvs.append(f"{k}={float(v):.6g}")
            except Exception:
                kvs.append(f"{k}={v}")
        _log(req.verbose, f"[objective:{obj_name}] " + " | ".join(kvs))
    else:
        _log(
            req.verbose,
            f"[objective:{obj_name}] scalar min/median/max = {np.nanmin(y_obj_scalar):.6g} / {np.nanmedian(y_obj_scalar):.6g} / {np.nanmax(y_obj_scalar):.6g}",  # noqa
        )
    append_round_log_event(
        rdir / "round.log.jsonl",
        {
            "ts": now_iso(),
            "stage": "objective_done",
            "objective": obj_name,
            "score_min": float(np.nanmin(y_obj_scalar)) if y_obj_scalar.size else None,
            "score_median": (float(np.nanmedian(y_obj_scalar)) if y_obj_scalar.size else None),
            "score_max": float(np.nanmax(y_obj_scalar)) if y_obj_scalar.size else None,
            "train_effect_pool_size": int(diag.get("train_effect_pool_size", 0)),
            "denom_used": (float(diag.get("denom_used", float("nan"))) if "denom_used" in diag else None),
        },
    )

    # --- Selection ---
    sel_name = cfg.selection.selection.name
    sel_params = dict(cfg.selection.selection.params)
    top_k = int(req.k_override or sel_params.get("top_k") or 0)
    if top_k <= 0:
        raise OpalError("selection.params.top_k must be > 0 (or override with --k).")
    tie_handling = sel_params.get("tie_handling", "competition_rank")
    mode = (sel_params.get("objective_mode") or "maximize").strip().lower()

    sel_fn = get_selection(sel_name, sel_params)
    sctx = rctx.for_plugin(category="selection", name=sel_name, plugin=sel_fn)
    raw_sel = sel_fn(
        ids=np.array(id_order_pool),
        scores=y_obj_scalar,
        top_k=top_k,
        tie_handling=tie_handling,
        objective=mode,
        ctx=sctx,
    )
    raw_scores = raw_sel["score"]
    del raw_sel['score']
    sel_norm = normalize_selection_result(
        raw_sel,
        ids=np.array(id_order_pool),
        scores=y_obj_scalar,
        top_k=top_k,
        tie_handling=tie_handling,
        objective=mode,
    )


    required_keys = {"rank_competition", "selected_bool"}
    missing = sorted(k for k in required_keys if k not in sel_norm)
    if missing:
        raise OpalError(
            f"normalize_selection_result is missing required key(s): {missing}. Present keys: {sorted(sel_norm.keys())}"
        )

    ranks_competition = np.asarray(sel_norm["rank_competition"]).astype(int)
    selected_bool = np.asarray(sel_norm["selected_bool"]).astype(bool)
    n = len(id_order_pool)
    if ranks_competition.shape[0] != n or selected_bool.shape[0] != n:
        raise OpalError(
            "Selection normalization shape mismatch: "
            f"expected length={n}, got rank_competition={ranks_competition.shape}, "
            f"selected_bool={selected_bool.shape}"
        )

    selected_effective = int(selected_bool.sum())
    _log(
        req.verbose,
        f"[selection:{sel_name}] objective={mode} tie={tie_handling} requested_top_k={top_k} → selected={selected_effective}",  # noqa
    )
    append_round_log_event(
        rdir / "round.log.jsonl",
        {
            "ts": now_iso(),
            "stage": "selection_done",
            "strategy": sel_name,
            "tie_handling": tie_handling,
            "objective_mode": mode,
            "requested_top_k": int(top_k),
            "effective_after_ties": int(selected_effective),
        },
    )
    rctx.set_core("core/selection/top_k_requested", int(top_k))
    rctx.set_core("core/selection/top_k_effective", int(selected_effective))
    rctx.set_core("core/selection/objective_mode", str(mode))
    rctx.set_core("core/selection/tie_handling", str(tie_handling))

    return _ScoreBundle(
        model=model,
        fit_metrics=fit_metrics,
        fit_duration=fit_duration,
        Y_hat=Y_hat,
        y_obj_scalar=y_obj_scalar,
        diag=diag,
        obj_summary_stats=obj_summary_stats if isinstance(obj_summary_stats, dict) else None,
        obj_name=obj_name,
        obj_params=obj_params,
        obj_mode=obj_mode,
        sel_name=sel_name,
        sel_params=sel_params,
        tie_handling=tie_handling,
        mode=mode,
        ranks_competition=ranks_competition,
        selected_bool=selected_bool,
        selected_effective=selected_effective,
        top_k=top_k,
        obj_sha=obj_sha,
        uq_scalar=uq_scalar,
        score=raw_scores,
    )


def run_round(store: RecordsStore, df: pd.DataFrame, req: RunRoundRequest) -> RunRoundResult:
    cfg = req.cfg
    cfg_path = req.config_path or (Path(cfg.campaign.workdir) / "campaign.yaml")
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    if not ws.state_path.exists():
        raise OpalError(f"state.json not found at {ws.state_path}. Run `opal init -c {ws.config_path}` first.")

    rdir = ws.round_dir(req.as_of_round)
    store.assert_unique_ids(df)

    if not req.allow_resume and rdir.exists() and any(rdir.iterdir()):
        raise OpalError(
            f"Round {int(req.as_of_round)} already contains artifacts in {rdir}. Use --resume to overwrite."
        )
    if req.allow_resume:
        _clear_round_dir(rdir)

    inputs = RoundInputs(cfg=cfg, req=req, ws=ws, store=store, df=df, rdir=rdir)

    try:
        yops_names = [p.name for p in (cfg.training.y_ops or [])]
    except Exception:
        yops_names = []
    _log(
        req.verbose,
        "[plugins] x=%s | y_ingest=%s | model=%s | objective=%s | selection=%s | y_ops=%s"
        % (
            cfg.data.transforms_x.name,
            cfg.data.transforms_y.name,
            cfg.model.name,
            cfg.objective.objective.name,
            cfg.selection.selection.name,
            (yops_names or "(none)"),
        ),
    )

    t0 = time.perf_counter()
    training = stage_training(inputs)

    ensure_dir(rdir)
    append_round_log_event(
        rdir / "logs" / "round.log.jsonl",
        {
            "ts": now_iso(),
            "stage": "start",
            "round": int(req.as_of_round),
            "campaign": {"slug": cfg.campaign.slug, "workdir": cfg.campaign.workdir},
            "data": {
                "x_column": cfg.data.x_column_name,
                "y_column": cfg.data.y_column_name,
            },
            "plugins": {
                "transform_x": {
                    "name": cfg.data.transforms_x.name,
                    "params": cfg.data.transforms_x.params,
                },
                "y_ingest": {
                    "name": cfg.data.transforms_y.name,
                    "params": cfg.data.transforms_y.params,
                },
                "model": {"name": cfg.model.name, "params": cfg.model.params},
                "objective": {
                    "name": cfg.objective.objective.name,
                    "params": cfg.objective.objective.params,
                },
                "selection": {
                    "name": cfg.selection.selection.name,
                    "params": cfg.selection.selection.params,
                },
                "y_ops": [{"name": p.name, "params": p.params} for p in (cfg.training.y_ops or [])],
            },
        },
    )


    run_id, _, rctx = build_round_ctx(
        cfg=cfg,
        as_of_round=int(req.as_of_round),
        y_dim=training.y_dim,
        n_train=len(training.train_ids),
    )

    tx = get_transform_x(cfg.data.transforms_x.name, cfg.data.transforms_x.params)
    tctx = rctx.for_plugin(category="transform_x", name=cfg.data.transforms_x.name, plugin=tx)

    xbundle = stage_x_matrices(
        inputs=inputs,
        plan=training.plan,
        train_ids=training.train_ids,
        Y_train=training.Y_train,
        tctx=tctx,
        rctx=rctx,
    )


    score = stage_scoring(
        inputs=inputs,
        rctx=rctx,
        X_train=xbundle.X_train,
        Y_train=training.Y_train,
        R_train=training.R_train,
        X_pool=xbundle.X_pool,
        id_order_train=xbundle.id_order_train,
        id_order_pool=xbundle.id_order_pool,
        y_dim=training.y_dim,
    )

    artifacts = write_round_artifacts(
        inputs=inputs,
        run_id=run_id,
        rctx=rctx,
        training=training,
        xbundle=xbundle,
        score=score,
    )

    run_events = build_run_events(
        inputs=inputs,
        run_id=run_id,
        training=training,
        xbundle=xbundle,
        score=score,
        artifacts=artifacts,
    )

    append_ledgers(
        ws=ws,
        run_pred_events=run_events.run_pred_events,
        run_meta_event=run_events.run_meta_event,
        verbose=req.verbose,
    )

    metrics_by_name: Dict[str, List[float]] = {"score": score.y_obj_scalar.astype(float).tolist()}
    for key in ("logic_fidelity", "effect_scaled", "effect_raw"):
        if isinstance(score.diag, dict) and key in score.diag:
            arr = np.asarray(score.diag[key], dtype=float).ravel()
            if arr.size == len(xbundle.id_order_pool):
                metrics_by_name[key] = arr.tolist()

    objective_meta = {
        "name": score.obj_name,
        "params": score.obj_params,
        "mode": score.mode,
    }
    _ = write_prediction_label_hist(
        store=store,
        df=df,
        ids=list(map(str, xbundle.id_order_pool)),
        y_hat=score.Y_hat,
        as_of_round=int(req.as_of_round),
        run_id=run_id,
        objective=objective_meta,
        metrics_by_name=metrics_by_name,
        selection_rank=score.ranks_competition,
        selection_top_k=score.selected_bool,
        verbose=req.verbose,
    )

    total_duration = time.perf_counter() - t0
    update_campaign_state(
        ws=ws,
        cfg=cfg,
        req=req,
        rep=training.rep,
        train_df=training.train_df,
        id_order_train=xbundle.id_order_train,
        id_order_pool=xbundle.id_order_pool,
        top_k=score.top_k,
        selected_effective=score.selected_effective,
        apaths=artifacts.apaths,
        run_id=run_id,
        obj_name=score.obj_name,
        store=store,
        total_duration=total_duration,
        fit_duration=score.fit_duration,
    )

    append_round_log_event(
        rdir / "logs" / "round.log.jsonl",
        {
            "ts": now_iso(),
            "round": int(req.as_of_round),
            "stage": "fit",
            "oob_r2": getattr(score.fit_metrics, "oob_r2", None),
        },
    )
    append_round_log_event(
        rdir / "logs" / "round.log.jsonl",
        {
            "ts": now_iso(),
            "round": int(req.as_of_round),
            "stage": "selection",
            "top_k": int(score.top_k),
            "effective": int(score.selected_effective),
        },
    )
    append_round_log_event(
        rdir / "logs" / "round.log.jsonl",
        {"ts": now_iso(), "round": int(req.as_of_round), "stage": "done"},
    )

    return RunRoundResult(
        ok=True,
        run_id=run_id,
        as_of_round=int(req.as_of_round),
        trained_on=len(xbundle.id_order_train),
        scored=len(xbundle.id_order_pool),
        top_k_requested=int(score.top_k),
        top_k_effective=int(score.selected_effective),
        ledger_path=str(ws.ledger_dir.resolve()),
    )
