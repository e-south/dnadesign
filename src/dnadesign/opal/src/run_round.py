"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/run_round.py

Application logic for a full OPAL round:
- preflights
- assemble training labels ≤ as_of_round
- (optional) orchestrate Y-ops (transforms prior to training)
- model fit/predict
- objective (any Y-ops are reverted before ranking)
- selection
- persist artifacts + append canonical events
- update ergonomic caches + state.json

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .artifacts import (
    ArtifactPaths,
    append_events,
    append_round_log_event,
    events_path,
    round_dir,
    write_feature_importance_csv,
    write_objective_meta,
    write_round_ctx,
    write_selection_csv,
)
from .config.types import RootConfig
from .data_access import RecordsStore
from .preflight import preflight_run
from .registries.models import get_model
from .registries.objectives import get_objective
from .registries.selections import get_selection, normalize_selection_result
from .registries.transforms_y import run_y_ops_pipeline
from .round_context import PluginRegistryView, RoundCtx
from .state import CampaignState, RoundEntry
from .utils import OpalError, ensure_dir, file_sha256, now_iso, print_stderr
from .writebacks import (
    SelectionEmit,
    build_label_events,
    build_run_meta_event,
    build_run_pred_events,
)


@dataclass
class RunRoundRequest:
    cfg: RootConfig
    as_of_round: int
    k_override: Optional[int] = None
    score_batch_size_override: Optional[int] = None
    verbose: bool = True
    allow_resume: bool = False


@dataclass
class RunRoundResult:
    ok: bool
    run_id: str
    as_of_round: int
    trained_on: int
    scored: int
    top_k_requested: int
    top_k_effective: int
    events_path: str


def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print_stderr(msg)


def run_round(store: RecordsStore, df: pd.DataFrame, req: RunRoundRequest) -> RunRoundResult:
    cfg = req.cfg
    workdir = Path(cfg.campaign.workdir)
    rdir = round_dir(workdir, req.as_of_round)
    ensure_dir(rdir)

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

    # --- Preflight (no auto-backfill or lazy fallbacks) ---
    t0 = time.perf_counter()
    rep = preflight_run(
        store,
        df,
        int(req.as_of_round),
        cfg.safety.fail_on_mixed_biotype_or_alphabet,
        auto_backfill=False,
    )
    wrote = bool(getattr(rep, "manual_attach_count", 0))
    if wrote:
        _log(req.verbose, "[preflight] reloading records after preflight writes...")
        df = store.load()

    # --- Labels (≤ round) ---
    train_df = store.training_labels_with_round(df, int(req.as_of_round))
    if train_df.empty:
        raise OpalError(f"No labels ≤ round {req.as_of_round} for training.")

    Y_train = np.stack(train_df["y"].map(lambda v: np.asarray(v, dtype=float)).to_list(), axis=0)
    R_train = train_df["r"].astype(int).to_numpy()
    y_dim = int(Y_train.shape[1])
    exp_len = cfg.data.y_expected_length
    if exp_len is not None and y_dim != int(exp_len):
        raise OpalError(f"Y length mismatch: expected {int(exp_len)}, got {y_dim}")

    _log(
        req.verbose,
        f"[labels] as_of_round={int(req.as_of_round)} | n_train={len(train_df)} | y_dim={y_dim}",
    )

    # --- X matrices & candidate universe ---
    X_train, id_order_train = store.transform_matrix(df, train_df["id"])
    cand_df = store.candidate_universe(df, int(req.as_of_round))

    # Optional: exclude already-labeled candidates (policy lives under selection.params)
    sel_name = cfg.selection.selection.name
    sel_params = dict(cfg.selection.selection.params)
    if bool(sel_params.get("exclude_already_labeled", True)):
        labeled_ids = store.labeled_id_set_leq_round(df, int(req.as_of_round))
        before = len(cand_df)
        if labeled_ids:
            mask = ~cand_df["id"].astype(str).isin(labeled_ids)
            cand_df = cand_df.loc[mask].copy()
        _log(
            req.verbose,
            f"[candidates] pool={before} → {len(cand_df)} after excluding already-labeled",
        )

    X_pool, id_order_pool = store.transform_matrix(df, cand_df["id"])
    _log(
        req.verbose,
        f"[transform] X_train: {X_train.shape} for {len(id_order_train)} ids | "
        f"X_pool: {X_pool.shape} for {len(id_order_pool)} ids",
    )

    # --- RoundCtx baseline (registry-aware) ---
    run_id = f"r{int(req.as_of_round)}-{now_iso()}"
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
            "core/round_index": int(req.as_of_round),
            "core/campaign_slug": cfg.campaign.slug,
            "core/labels_as_of_round": int(req.as_of_round),
            "core/plugins/transforms_x/name": reg.transform_x,
            "core/plugins/transforms_y/name": reg.transform_y,
            "core/plugins/model/name": reg.model,
            "core/plugins/objective/name": reg.objective,
            "core/plugins/selection/name": reg.selection,
        },
        registry=reg,
    )

    # --- Y-ops (fit/transform) via registry (decoupled) ---
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
    _log(
        req.verbose,
        f"[fit] model={cfg.model.name} | dt={tfit1 - tfit0:.3f}s | oob_r2={getattr(fit_metrics, 'oob_r2', None)}",
    )

    # --- Predict candidates (batched) in Y-ops space ---
    sbatch = int(req.score_batch_size_override or cfg.scoring.score_batch_size)
    yhat_chunks: List[np.ndarray] = []
    total = X_pool.shape[0]
    num_batches = max(1, (total + max(1, sbatch) - 1) // max(1, sbatch))

    def _progress(i: int, total_batches: int) -> None:
        if not req.verbose:
            return
        pct = (i / max(total_batches, 1)) * 100.0
        import sys as _sys

        _sys.stderr.write(f"\r[predict] {i}/{total_batches} batches ~{pct:0.1f}%")
        _sys.stderr.flush()

    for bi, i in enumerate(range(0, total, max(1, sbatch))):
        sl = slice(i, i + sbatch)
        yhat_chunks.append(model.predict(X_pool[sl], ctx=mctx))
        _progress(bi + 1, num_batches)
        append_round_log_event(
            rdir / "round.log.jsonl",
            {
                "ts": now_iso(),
                "stage": "predict_batch",
                "batch": int(bi + 1),
                "of": int(num_batches),
                "rows": int(sl.stop - sl.start),
            },
        )
    if req.verbose:
        import sys as _sys

        _sys.stderr.write("\n")
    Y_hat_fit = np.vstack(yhat_chunks) if yhat_chunks else np.zeros((0, y_dim), dtype=float)

    # --- Inverse Y-ops to objective space (decoupled) ---
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
    # Assert that objective returned finite scalars (no fallbacks)
    non_finite_mask = ~np.isfinite(y_obj_scalar)
    if non_finite_mask.any():
        n = int(non_finite_mask.sum())
        n_total = int(y_obj_scalar.size)
        n_nan = int(np.isnan(y_obj_scalar).sum())
        n_pinf = int(np.isposinf(y_obj_scalar).sum())
        n_ninf = int(np.isneginf(y_obj_scalar).sum())
        # We have id_order_pool already; build preview with ids (and optional sequences if available)
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

    # Pipeline-agnostic objective summary:
    ss = diag.get("summary_stats") if isinstance(diag, dict) else None
    if isinstance(ss, dict) and ss:
        # Print a compact "k=v" list with stable ordering
        kvs: List[str] = []
        for k in sorted(ss.keys()):
            v = ss[k]
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
    sel_norm = normalize_selection_result(
        raw_sel,
        ids=np.array(id_order_pool),
        scores=y_obj_scalar,
        top_k=top_k,
        tie_handling=tie_handling,
        objective=mode,
    )

    # ---- Assertive validation of normalized selection ----
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
    # --- Artifacts ---
    apaths = ArtifactPaths(
        model=rdir / "model.joblib",
        selection_csv=rdir / "selection_top_k.csv",
        round_log_jsonl=rdir / "round.log.jsonl",
        round_ctx_json=rdir / "round_ctx.json",
        objective_meta_json=rdir / "objective_meta.json",
    )
    model.save(str(apaths.model))

    # Map sequences up-front for artifact emission
    seq_map = df.set_index("id")["sequence"].astype(str).to_dict() if "sequence" in df.columns else {}
    selected_df = (
        pd.DataFrame(
            {
                "id": id_order_pool,
                "sequence": [seq_map.get(i) for i in id_order_pool],
                "sel__rank_competition": ranks_competition,
                "selection_score": y_obj_scalar,
                "sel__is_selected": selected_bool,
            }
        )
        .loc[lambda d: d["sel__is_selected"]]
        .sort_values("sel__rank_competition")[["id", "sequence", "sel__rank_competition", "selection_score"]]
    )
    sel_sha = write_selection_csv(apaths.selection_csv, selected_df)

    ctx_sha = write_round_ctx(apaths.round_ctx_json, rctx.snapshot())

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
        "[artifacts] model, selection_csv, round_ctx, objective_meta"
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

    # Snapshot the consolidated ground-truth labels actually used for training (≤ as_of_round)
    # Build label events grouped by their original observed round.
    label_events_frames: List[pd.DataFrame] = []
    if not train_df.empty:
        # map ids -> sequences for label events
        _seq_for = lambda ids: [seq_map.get(i) for i in ids]  # noqa
        for rr, grp in train_df.groupby("r"):
            ids_g = grp["id"].astype(str).tolist()
            y_g = grp["y"].tolist()
            seq_g = _seq_for(ids_g)
            label_events_frames.append(
                build_label_events(
                    ids=ids_g,
                    sequences=seq_g,
                    y_obs=y_g,
                    observed_round=int(rr),
                    src="training_snapshot",
                    note=f"labels used in run {run_id} (as_of_round={int(req.as_of_round)})",
                )
            )
    label_events_df = pd.concat(label_events_frames, ignore_index=True) if label_events_frames else None

    # Now add the standard artifacts (model, selection, round ctx, objective meta)
    artifacts_paths_and_hashes.update(
        {
            "model.joblib": (file_sha256(apaths.model), str(apaths.model.resolve())),
            "selection_top_k.csv": (sel_sha, str(apaths.selection_csv.resolve())),
            "round_ctx.json": (ctx_sha, str(apaths.round_ctx_json.resolve())),
            "objective_meta.json": (obj_sha, str(apaths.objective_meta_json.resolve())),
        }
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

    ev_path = events_path(workdir)
    # Always write typed sinks; do *not* write the deprecated thin index.
    append_events(
        ev_path,
        pd.concat([run_pred_events, run_meta_event], ignore_index=True),
        write_index=False,
    )
    if label_events_df is not None and not label_events_df.empty:
        append_events(ev_path, label_events_df, write_index=False)
    _log(
        req.verbose,
        f"[events] appended run_pred({len(run_pred_events)}), run_meta(1)"
        + (f", labels({len(label_events_df)})" if label_events_df is not None else "")
        + f" under {ev_path.parent}",
    )

    # --- records ergonomic caches ---
    latest_scalar = dict(zip(id_order_pool, y_obj_scalar.tolist()))
    df2 = store.update_latest_cache(
        df,
        slug=cfg.campaign.slug,
        latest_as_of_round=int(req.as_of_round),
        latest_scalar_by_id=latest_scalar,
    )
    store.save_atomic(df2)
    _log(
        req.verbose,
        "[writeback] records caches updated: latest_as_of_round, latest_pred_scalar",
    )

    # --- state.json summary ---
    st_path = Path(cfg.campaign.workdir) / "state.json"
    # Be strict: require `opal init` to have created state.json
    if not st_path.exists():
        raise OpalError(
            f"state.json not found at {st_path}. Run `opal init -c {Path(cfg.campaign.workdir) / 'campaign.yaml'}` first."  # noqa
        )
    st = CampaignState.load(st_path)
    # If resuming on the same round, replace existing entries for this round index
    if req.allow_resume:
        try:
            st.rounds = [r for r in st.rounds if int(getattr(r, "round_index", -1)) != int(req.as_of_round)]
        except Exception:
            pass

    st.campaign_slug = cfg.campaign.slug
    st.campaign_name = cfg.campaign.name
    st.workdir = str(workdir.resolve())
    st.x_column_name = cfg.data.x_column_name
    st.y_column_name = cfg.data.y_column_name
    st.representation_vector_dimension = rep.x_dim or 0
    st.performance = {
        "score_batch_size": req.score_batch_size_override or cfg.scoring.score_batch_size,
        "objective": obj_name,
    }
    st.add_round(
        RoundEntry(
            round_index=int(req.as_of_round),
            round_name=f"round_{int(req.as_of_round)}",
            round_dir=str(rdir.resolve()),
            labels_used_rounds=list(range(0, int(req.as_of_round) + 1)),
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
                # Keep legacy key for compatibility; may point to a non-existent file.
                "events_parquet": str(ev_path.resolve()),
                # New, explicit ledger sinks:
                "ledger_predictions_dir": str((ev_path.parent / "ledger.predictions").resolve()),
                "ledger_runs_parquet": str((ev_path.parent / "ledger.runs.parquet").resolve()),
                "ledger_labels_parquet": str((ev_path.parent / "ledger.labels.parquet").resolve()),
                "round_ctx_json": str(apaths.round_ctx_json.resolve()),
                "objective_meta_json": str(apaths.objective_meta_json.resolve()),
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
    st.rounds[-1].durations_sec = {"total": t1 - t0, "fit": tfit1 - tfit0}
    st.save(st_path)

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
        events_path=str(ev_path.parent.resolve()),
    )
