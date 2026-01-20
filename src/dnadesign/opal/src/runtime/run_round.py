"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/runtime/run_round.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config.types import LocationLocal, LocationUSR, RootConfig
from ..core.progress import NullProgress, ProgressFactory
from ..core.round_context import PluginRegistryView, RoundCtx
from ..core.utils import OpalError, ensure_dir, file_sha256, now_iso, print_stderr
from ..registries.models import get_model
from ..registries.objectives import get_objective
from ..registries.selection import get_selection, normalize_selection_result
from ..registries.transforms_x import get_transform_x
from ..registries.transforms_y import run_y_ops_pipeline
from ..storage.artifacts import (
    ArtifactPaths,
    append_round_log_event,
    write_feature_importance_csv,
    write_labels_used_parquet,
    write_model_meta,
    write_objective_meta,
    write_round_ctx,
    write_selection_csv,
    write_selection_parquet,
)
from ..storage.data_access import RecordsStore
from ..storage.ledger import LedgerWriter
from ..storage.state import CampaignState, RoundEntry
from ..storage.workspace import CampaignWorkspace
from ..storage.writebacks import (
    SelectionEmit,
    build_run_meta_event,
    build_run_pred_events,
)
from .preflight import preflight_run
from .round_plan import plan_round


@dataclass
class RunRoundRequest:
    cfg: RootConfig
    as_of_round: int
    config_path: Optional[Path] = None
    k_override: Optional[int] = None
    score_batch_size_override: Optional[int] = None
    verbose: bool = True
    allow_resume: bool = False
    progress_factory: Optional[ProgressFactory] = None


@dataclass
class RunRoundResult:
    ok: bool
    run_id: str
    as_of_round: int
    trained_on: int
    scored: int
    top_k_requested: int
    top_k_effective: int
    ledger_path: str


@dataclass
class _TrainingBundle:
    rep: Any
    plan: Any
    train_df: pd.DataFrame
    train_ids: List[str]
    Y_train: np.ndarray
    R_train: np.ndarray
    y_dim: int


@dataclass
class _XBundle:
    X_train: np.ndarray
    id_order_train: List[str]
    X_pool: np.ndarray
    id_order_pool: List[str]
    cand_df: pd.DataFrame


@dataclass
class _ScoreBundle:
    model: Any
    fit_metrics: Any
    fit_duration: float
    Y_hat: np.ndarray
    y_obj_scalar: np.ndarray
    diag: Dict[str, Any]
    obj_summary_stats: Optional[Dict[str, Any]]
    obj_name: str
    obj_params: Dict[str, Any]
    obj_mode: str
    sel_name: str
    sel_params: Dict[str, Any]
    tie_handling: str
    mode: str
    ranks_competition: np.ndarray
    selected_bool: np.ndarray
    selected_effective: int
    top_k: int
    obj_sha: str


def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print_stderr(msg)


def _resolve_selection_objective_mode(sel_params: Dict[str, Any], *, warn: bool = True) -> str:
    mode_raw = sel_params.get("objective_mode")
    legacy_raw = sel_params.get("objective")
    if mode_raw is not None and legacy_raw is not None:
        mode_str = str(mode_raw).strip().lower()
        legacy_str = str(legacy_raw).strip().lower()
        if mode_str != legacy_str:
            raise OpalError(
                "selection.params has both 'objective_mode' and legacy 'objective' with conflicting values "
                f"({mode_str!r} vs {legacy_str!r})."
            )
    if mode_raw is None and legacy_raw is not None:
        if warn:
            print_stderr("[opal] selection.params.objective is deprecated; please use selection.params.objective_mode.")
        mode_raw = legacy_raw
    if mode_raw is None:
        mode_raw = "maximize"
    mode = str(mode_raw).strip().lower()
    if mode not in {"maximize", "minimize"}:
        raise OpalError(f"Invalid selection objective_mode: {mode!r} (expected 'maximize' or 'minimize').")
    return mode


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
    obj_mode_norm = str(obj_mode).strip().lower()
    if obj_mode_norm not in {"maximize", "minimize"}:
        raise OpalError(
            f"Objective returned invalid mode {obj_mode!r} (expected 'maximize' or 'minimize'). "
            "Fix the objective plugin or configuration."
        )

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
    mode = _resolve_selection_objective_mode(sel_params)
    sel_params["objective_mode"] = mode
    if obj_mode_norm != mode:
        raise OpalError(
            "Objective mode mismatch: objective returned "
            f"{obj_mode_norm!r} but selection objective_mode is {mode!r}. "
            "Align objective and selection modes to avoid inverted ranking."
        )

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
    )


def run_round(store: RecordsStore, df: pd.DataFrame, req: RunRoundRequest) -> RunRoundResult:
    cfg = req.cfg
    cfg_path = req.config_path or (Path(cfg.campaign.workdir) / "campaign.yaml")
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    # Be strict: require opal init to have created state.json before any heavy work
    if not ws.state_path.exists():
        raise OpalError(f"state.json not found at {ws.state_path}. Run `opal init -c {ws.config_path}` first.")

    rdir = ws.round_dir(req.as_of_round)
    ensure_dir(rdir)
    store.assert_unique_ids(df)

    # Guard against accidental double-runs unless explicitly allowed
    if not req.allow_resume:
        # Consider the directory "non-empty" if it contains any known artifacts
        known = [
            "model.joblib",
            "selection_top_k.csv",
            "round_ctx.json",
            "objective_meta.json",
        ]
        if any((rdir / k).exists() for k in known):
            raise OpalError(
                f"Round {int(req.as_of_round)} appears to have already produced artifacts in {rdir}. Use --resume to overwrite."  # noqa
            )

    # --- round start log (assurance of alignment) ---
    append_round_log_event(
        rdir / "round.log.jsonl",
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

    # Human-facing succinct plugin overview
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

    # --- Preflight + training plan ---
    t0 = time.perf_counter()
    training = _stage_training(store=store, df=df, cfg=cfg, req=req)
    rep = training.rep
    plan = training.plan
    train_df = training.train_df
    train_ids = training.train_ids
    Y_train = training.Y_train
    R_train = training.R_train
    y_dim = training.y_dim

    # --- RoundCtx baseline (registry-aware) ---
    run_id, reg, rctx = _build_round_ctx(
        cfg=cfg,
        as_of_round=int(req.as_of_round),
        y_dim=y_dim,
        n_train=len(train_ids),
    )

    # Transform X plugin context (used for training + pool transforms)
    tx = get_transform_x(cfg.data.transforms_x.name, cfg.data.transforms_x.params)
    tctx = rctx.for_plugin(category="transform_x", name=cfg.data.transforms_x.name, plugin=tx)

    # --- X matrices & candidate universe ---
    xbundle = _stage_x_matrices(
        store=store,
        df=df,
        plan=plan,
        train_ids=train_ids,
        Y_train=Y_train,
        cfg=cfg,
        req=req,
        tctx=tctx,
        rctx=rctx,
    )
    X_train = xbundle.X_train
    id_order_train = xbundle.id_order_train
    id_order_pool = xbundle.id_order_pool

    score = _stage_fit_predict_score(
        cfg=cfg,
        req=req,
        rdir=rdir,
        rctx=rctx,
        X_train=X_train,
        Y_train=Y_train,
        R_train=R_train,
        X_pool=xbundle.X_pool,
        id_order_train=id_order_train,
        id_order_pool=id_order_pool,
        y_dim=y_dim,
    )
    model = score.model
    fit_metrics = score.fit_metrics
    Y_hat = score.Y_hat
    y_obj_scalar = score.y_obj_scalar
    diag = score.diag
    obj_name = score.obj_name
    obj_params = score.obj_params
    sel_name = score.sel_name
    sel_params = score.sel_params
    tie_handling = score.tie_handling
    mode = score.mode
    ranks_competition = score.ranks_competition
    selected_bool = score.selected_bool
    selected_effective = score.selected_effective
    top_k = score.top_k
    obj_sha = score.obj_sha
    ss = score.obj_summary_stats
    fit_duration = score.fit_duration
    # --- Artifacts ---
    apaths = ArtifactPaths(
        model=rdir / "model.joblib",
        selection_csv=rdir / "selection_top_k.csv",
        round_log_jsonl=rdir / "round.log.jsonl",
        round_ctx_json=rdir / "round_ctx.json",
        objective_meta_json=rdir / "objective_meta.json",
    )
    model.save(str(apaths.model))
    model_meta = {
        "model__name": cfg.model.name,
        "model__params": model.get_params(),
        "training__y_ops": [{"name": p.name, "params": p.params} for p in (cfg.training.y_ops or [])],
        "x_dim": int(X_train.shape[1]),
        "y_dim": int(y_dim),
    }
    model_meta_sha = write_model_meta(rdir / "model_meta.json", model_meta)

    # Map sequences up-front for artifact emission
    seq_map = df.set_index("id")["sequence"].astype(str).to_dict() if "sequence" in df.columns else {}
    if seq_map:
        seq_map = {str(k): v for k, v in seq_map.items()}

    # Training snapshot (artifact only; not appended to ledger)
    labels_used_df = None
    if not train_df.empty:
        labels_used_df = pd.DataFrame(
            {
                "run_id": [run_id] * len(train_df),
                "as_of_round": [int(req.as_of_round)] * len(train_df),
                "observed_round": train_df["r"].astype(int).tolist(),
                "id": train_df["id"].astype(str).tolist(),
                "sequence": [seq_map.get(i) for i in train_df["id"].astype(str).tolist()],
                "y_obs": train_df["y"].tolist(),
                "src": ["training_snapshot"] * len(train_df),
                "note": [f"labels used in run {run_id} (as_of_round={int(req.as_of_round)})"] * len(train_df),
            }
        )
    selected_ids = [str(i) for i in id_order_pool]
    selected_df = pd.DataFrame(
        {
            "id": selected_ids,
            "sequence": [seq_map.get(i) for i in selected_ids],
            "sel__rank_competition": ranks_competition,
            "selection_score": y_obj_scalar,
            "sel__is_selected": selected_bool,
        }
    )
    selected_df = selected_df.loc[lambda d: d["sel__is_selected"]].sort_values("sel__rank_competition")
    selected_df = selected_df.assign(
        run_id=run_id,
        as_of_round=int(req.as_of_round),
        campaign_slug=cfg.campaign.slug,
        selection_name=sel_name,
        objective_name=obj_name,
        objective_mode=mode,
        tie_handling=tie_handling,
        pred__y_obj_scalar=selected_df["selection_score"],
    )
    selected_df = selected_df[
        [
            "run_id",
            "as_of_round",
            "campaign_slug",
            "selection_name",
            "objective_name",
            "objective_mode",
            "tie_handling",
            "id",
            "sequence",
            "sel__rank_competition",
            "selection_score",
            "pred__y_obj_scalar",
        ]
    ]
    sel_sha = write_selection_csv(apaths.selection_csv, selected_df)
    sel_parquet_path = apaths.selection_csv.with_suffix(".parquet")
    sel_parquet_sha = write_selection_parquet(sel_parquet_path, selected_df)
    sel_run_csv_path = apaths.selection_csv.with_name(f"selection_top_k__run_{run_id}.csv")
    sel_run_parquet_path = sel_parquet_path.with_name(f"selection_top_k__run_{run_id}.parquet")
    sel_run_csv_sha = write_selection_csv(sel_run_csv_path, selected_df)
    sel_run_parquet_sha = write_selection_parquet(sel_run_parquet_path, selected_df)

    ctx_sha = write_round_ctx(apaths.round_ctx_json, rctx.snapshot())
    labels_used_sha = write_labels_used_parquet(rdir / "labels_used.parquet", labels_used_df)

    # Collect artifact paths and hashes here so optional model artifacts can append to it.
    artifacts_paths_and_hashes: Dict[str, tuple[str, str]] = {}

    # ---- Optional model artifacts (e.g., RandomForest feature importance) ----
    if hasattr(model, "model_artifacts") and callable(getattr(model, "model_artifacts")):
        try:
            m_arts = model.model_artifacts() or {}
            if "feature_importance" in m_arts:
                fi_df = m_arts["feature_importance"]
                fi_path = rdir / "feature_importance.csv"
                fi_sha = write_feature_importance_csv(fi_path, fi_df)
                # small, helpful stats (logged; not persisted elsewhere)
                imp = fi_df["importance"].to_numpy(dtype=float)
                xdim = int(imp.shape[0])
                nonzero = int((imp > 0).sum())
                hhi = float(np.sum((imp / max(imp.sum(), 1e-12)) ** 2))
                top5 = fi_df.sort_values("importance", ascending=False).head(5).to_dict(orient="records")
                _log(
                    req.verbose,
                    f"[model] feature_importance: x_dim={xdim} nonzero={nonzero} HHI={hhi:.4f}",
                )
                append_round_log_event(
                    rdir / "round.log.jsonl",
                    {
                        "ts": now_iso(),
                        "stage": "model_feature_importance",
                        "x_dim": xdim,
                        "nonzero": nonzero,
                        "hhi": hhi,
                        "top5": top5,
                        "artifact": "feature_importance.csv",
                        "artifact_sha256": fi_sha,
                    },
                )
                artifacts_paths_and_hashes["feature_importance.csv"] = (
                    fi_sha,
                    str(fi_path.resolve()),
                )
        except Exception as e:
            raise OpalError(f"Failed to persist model artifacts: {e}")

    _log(
        req.verbose,
        "[artifacts] model, selection_csv, selection_parquet, round_ctx, objective_meta"
        + (", feature_importance" if "feature_importance.csv" in artifacts_paths_and_hashes else "")
        + " written",
    )

    # --- Events ---
    sequences_list: List[Optional[str]] = [seq_map.get(i) for i in id_order_pool]

    # Uncertainty & QC masks (explicitly None in demo)
    y_hat_model_sd = None
    y_obj_scalar_sd = None

    sel_emit = SelectionEmit(
        ranks_competition=ranks_competition,
        selected_bool=selected_bool,
        diagnostics=None,
    )

    run_pred_events = build_run_pred_events(
        run_id=run_id,
        as_of_round=int(req.as_of_round),
        ids=list(map(str, id_order_pool)),
        sequences=sequences_list,
        y_hat_model=Y_hat,
        y_obj_scalar=y_obj_scalar,
        y_dim=y_dim,
        y_hat_model_sd=y_hat_model_sd,
        y_obj_scalar_sd=y_obj_scalar_sd,
        obj_diagnostics=diag,
        sel_emit=sel_emit,
    )

    # Now add the standard artifacts (model, selection, round ctx, objective meta)
    artifacts_paths_and_hashes.update(
        {
            "model.joblib": (file_sha256(apaths.model), str(apaths.model.resolve())),
            "selection_top_k.csv": (sel_sha, str(apaths.selection_csv.resolve())),
            "selection_top_k.parquet": (sel_parquet_sha, str(sel_parquet_path.resolve())),
            f"selection_top_k__run_{run_id}.csv": (sel_run_csv_sha, str(sel_run_csv_path.resolve())),
            f"selection_top_k__run_{run_id}.parquet": (sel_run_parquet_sha, str(sel_run_parquet_path.resolve())),
            "round_ctx.json": (ctx_sha, str(apaths.round_ctx_json.resolve())),
            "objective_meta.json": (obj_sha, str(apaths.objective_meta_json.resolve())),
            "model_meta.json": (
                model_meta_sha,
                str((rdir / "model_meta.json").resolve()),
            ),
        }
    )
    artifacts_paths_and_hashes["labels_used.parquet"] = (
        labels_used_sha,
        str((rdir / "labels_used.parquet").resolve()),
    )

    run_meta_event = build_run_meta_event(
        run_id=run_id,
        as_of_round=int(req.as_of_round),
        model_name=cfg.model.name,
        model_params=model.get_params(),
        y_ops=[{"name": p.name, "params": p.params} for p in (cfg.training.y_ops or [])],
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params=dict(cfg.data.transforms_x.params),
        y_ingest_transform_name=cfg.data.transforms_y.name,
        y_ingest_transform_params=dict(cfg.data.transforms_y.params),
        objective_name=obj_name,
        objective_params=obj_params,
        selection_name=sel_name,
        selection_params=sel_params,
        selection_score_field="pred__y_obj_scalar",
        selection_objective_mode=mode,
        sel_tie_handling=tie_handling,
        stats_n_train=len(id_order_train),
        stats_n_scored=len(id_order_pool),
        unc_mean_sd=y_hat_model_sd,  # None in demo
        pred_rows_df=run_pred_events,
        artifact_paths_and_hashes=artifacts_paths_and_hashes,
        objective_summary_stats=ss,
    )

    ledger = LedgerWriter(ws)
    ledger.append_run_pred(run_pred_events)
    ledger.append_run_meta(run_meta_event)
    _log(
        req.verbose,
        f"[ledger] appended run_pred({len(run_pred_events)}), run_meta(1) under {ws.ledger_dir}",
    )

    # --- records ergonomic caches ---
    latest_scalar = dict(zip(id_order_pool, y_obj_scalar.tolist()))
    df2 = store.update_latest_cache(
        df,
        slug=cfg.campaign.slug,
        latest_as_of_round=int(req.as_of_round),
        latest_scalar_by_id=latest_scalar,
        require_columns_present=cfg.safety.write_back_requires_columns_present,
        latest_pred_run_id=run_id,
        latest_pred_as_of_round=int(req.as_of_round),
        latest_pred_written_at=now_iso(),
    )
    store.save_atomic(df2)
    _log(
        req.verbose,
        "[writeback] records caches updated: latest_as_of_round, latest_pred_scalar",
    )

    # --- state.json summary ---
    st = CampaignState.load(ws.state_path)
    # If resuming on the same round, replace existing entries for this round index
    if req.allow_resume:
        try:
            st.rounds = [r for r in st.rounds if int(getattr(r, "round_index", -1)) != int(req.as_of_round)]
        except Exception:
            pass

    st.campaign_slug = cfg.campaign.slug
    st.campaign_name = cfg.campaign.name
    st.workdir = str(ws.workdir.resolve())
    loc = cfg.data.location
    if isinstance(loc, LocationUSR):
        st.data_location = {
            "kind": "usr",
            "dataset": loc.dataset,
            "path": str(Path(loc.path).resolve()),
            "records_path": str(store.records_path.resolve()),
        }
    elif isinstance(loc, LocationLocal):
        st.data_location = {
            "kind": "local",
            "path": str(Path(loc.path).resolve()),
            "records_path": str(store.records_path.resolve()),
        }
    st.x_column_name = cfg.data.x_column_name
    st.y_column_name = cfg.data.y_column_name
    st.representation_vector_dimension = rep.x_dim or 0
    st.representation_transform = {
        "name": cfg.data.transforms_x.name,
        "params": cfg.data.transforms_x.params,
    }
    st.training_policy = dict(cfg.training.policy or {})
    st.performance = {
        "score_batch_size": req.score_batch_size_override or cfg.scoring.score_batch_size,
        "objective": obj_name,
    }
    labels_used_rounds = sorted(set(train_df["r"].astype(int).tolist()))
    st.add_round(
        RoundEntry(
            round_index=int(req.as_of_round),
            run_id=str(run_id),
            round_name=f"round_{int(req.as_of_round)}",
            round_dir=str(rdir.resolve()),
            labels_used_rounds=labels_used_rounds,
            number_of_training_examples_used_in_round=len(id_order_train),
            number_of_candidates_scored_in_round=len(id_order_pool),
            selection_top_k_requested=int(top_k),
            selection_top_k_effective_after_ties=int(selected_effective),
            model={
                "type": cfg.model.name,
                "params": cfg.model.params,
                "artifact_path": str(apaths.model.resolve()),
                "artifact_sha256": file_sha256(apaths.model),
            },
            metrics={},
            durations_sec={},  # filled below
            seeds={
                "global": cfg.model.params.get("random_state"),
                "model": cfg.model.params.get("random_state"),
            },
            artifacts={
                "selection_top_k_csv": str(apaths.selection_csv.resolve()),
                # New, explicit ledger sinks:
                "ledger_predictions_dir": str(ws.ledger_predictions_dir.resolve()),
                "ledger_runs_parquet": str(ws.ledger_runs_path.resolve()),
                "ledger_labels_parquet": str(ws.ledger_labels_path.resolve()),
                "round_ctx_json": str(apaths.round_ctx_json.resolve()),
                "objective_meta_json": str(apaths.objective_meta_json.resolve()),
                "model_meta_json": str((rdir / "model_meta.json").resolve()),
                "labels_used_parquet": str((rdir / "labels_used.parquet").resolve()),
                "round_log_jsonl": str((rdir / "round.log.jsonl").resolve()),
            },
            writebacks={
                "records_caches_updated": [
                    "opal__<slug>__latest_as_of_round",
                    "opal__<slug>__latest_pred_scalar",
                ]
            },
            warnings=[],
        )
    )
    t1 = time.perf_counter()
    # Dataclass field update (not dict-style)
    st.rounds[-1].durations_sec = {"total": t1 - t0, "fit": fit_duration}
    st.save(ws.state_path)

    append_round_log_event(
        rdir / "round.log.jsonl",
        {
            "ts": now_iso(),
            "round": int(req.as_of_round),
            "stage": "fit",
            "oob_r2": getattr(fit_metrics, "oob_r2", None),
        },
    )
    append_round_log_event(
        rdir / "round.log.jsonl",
        {
            "ts": now_iso(),
            "round": int(req.as_of_round),
            "stage": "selection",
            "top_k": int(top_k),
            "effective": int(selected_effective),
        },
    )
    append_round_log_event(
        rdir / "round.log.jsonl",
        {"ts": now_iso(), "round": int(req.as_of_round), "stage": "done"},
    )

    return RunRoundResult(
        ok=True,
        run_id=run_id,
        as_of_round=int(req.as_of_round),
        trained_on=len(id_order_train),
        scored=len(id_order_pool),
        top_k_requested=int(top_k),
        top_k_effective=int(selected_effective),
        ledger_path=str(ws.ledger_dir.resolve()),
    )
