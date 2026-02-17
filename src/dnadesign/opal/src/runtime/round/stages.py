"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/runtime/round/stages.py

Implements core round stages for OPAL execution (training, scoring, selection).
Each stage returns a typed bundle for the next stage.

Module Author(s): Eric J. South, Elm Markert
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from ...core.objective_result import validate_objective_result_v2
from ...core.progress import NullProgress
from ...core.round_context import Contract, RoundCtx
from ...core.selection_contracts import (
    extract_selection_plugin_params,
    resolve_selection_objective_mode,
    resolve_selection_tie_handling,
)
from ...core.utils import OpalError, now_iso, print_stderr
from ...registries.models import get_model
from ...registries.objectives import get_objective
from ...registries.selection import get_selection, normalize_selection_result, validate_selection_result
from ...registries.transforms_y import get_y_op, run_y_ops_pipeline
from ...storage.artifacts import append_round_log_event, write_objective_meta
from ..preflight import preflight_run
from ..round_plan import plan_round
from .contracts import RoundInputs, ScoreBundle, TrainingBundle, XBundle


def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print_stderr(msg)


def _resolve_channel_ref(ref: str, channels: Dict[str, np.ndarray], *, label: str) -> np.ndarray:
    key = str(ref).strip()
    if not key:
        raise OpalError(f"{label} channel reference cannot be empty.")
    if key not in channels:
        raise OpalError(f"{label} channel '{key}' not found. Available: {sorted(channels.keys())}")
    return np.asarray(channels[key], dtype=float).reshape(-1)


def _format_summary_stats_for_log(summary_stats: Dict[str, Any]) -> List[str]:
    kvs: List[str] = []
    for k in sorted(summary_stats.keys()):
        v = summary_stats[k]
        if isinstance(v, (bool, np.bool_)):
            kvs.append(f"{k}={v}")
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            kvs.append(f"{k}={float(v):.6g}")
            continue
        kvs.append(f"{k}={v}")
    return kvs


def _coalesce_uncertainty_chunks(std_payload: Any) -> np.ndarray:
    if isinstance(std_payload, list):
        if len(std_payload) == 0:
            raise OpalError("model/<self>/std_devs payload is empty after prediction.")
        chunks = [np.asarray(chunk, dtype=float) for chunk in std_payload]
        dims = {arr.ndim for arr in chunks}
        if dims == {1}:
            return np.concatenate([arr.reshape(-1) for arr in chunks], axis=0)
        if dims == {2}:
            widths = {arr.shape[1] for arr in chunks}
            if len(widths) != 1:
                raise OpalError("model/<self>/std_devs chunks have inconsistent column widths.")
            return np.vstack(chunks)
        raise OpalError("model/<self>/std_devs chunks must be all 1D or all 2D arrays.")
    arr = np.asarray(std_payload, dtype=float)
    if arr.ndim in (1, 2):
        return arr
    raise OpalError("model/<self>/std_devs payload must be a 1D or 2D numeric array.")


def _inverse_yops_outputs(
    *,
    rctx: RoundCtx,
    y_ops_cfg: List[Any],
    y_pred: np.ndarray,
    y_pred_std: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    y_inv = np.asarray(y_pred, dtype=float)
    std_inv = None if y_pred_std is None else np.asarray(y_pred_std, dtype=float)
    if not y_ops_cfg:
        return y_inv, std_inv

    names = rctx.get("yops/pipeline/names", default=[])
    params_used = rctx.get("yops/pipeline/params", default=[])
    if len(names) != len(params_used):
        raise OpalError("Malformed Y-ops pipeline: names/params length mismatch.")

    for name, params in zip(reversed(names), reversed(params_used)):
        spec = get_y_op(name)
        ParamT = spec.ParamModel
        params_obj = ParamT(**params) if ParamT is not None else params
        inv_contract = Contract(
            category="yops",
            requires=spec.requires + spec.produces,
            produces=tuple(),
        )
        inv_ctx = rctx.for_plugin(category="yops", name=name, contract=inv_contract)
        inv_ctx.precheck_requires(stage="inverse")

        y_before = y_inv
        if std_inv is not None:
            inverse_std_fn = getattr(spec.inverse_fn, "inverse_std", None)
            if inverse_std_fn is None:
                raise OpalError(
                    "[round] y_ops are active but "
                    f"{name} does not support inverse_std; uncertainty channels require objective-space units."
                )
            else:
                std_inv = np.asarray(
                    inverse_std_fn(std_inv, params_obj, ctx=inv_ctx, y_pred_transformed=y_before),
                    dtype=float,
                )
                if std_inv.shape != y_before.shape:
                    raise OpalError(
                        f"Y-op inverse_std shape mismatch for {name}: expected {y_before.shape}, got {std_inv.shape}."
                    )
                if not np.all(np.isfinite(std_inv)):
                    raise OpalError(f"Y-op inverse_std produced non-finite values for {name}.")
                if np.any(std_inv < 0.0):
                    raise OpalError(f"Y-op inverse_std produced negative standard deviations for {name}.")

        y_inv = np.asarray(spec.inverse_fn(y_before, params_obj, ctx=inv_ctx), dtype=float)

    return y_inv, std_inv


def stage_training(inputs: RoundInputs) -> TrainingBundle:
    cfg = inputs.cfg
    req = inputs.req
    store = inputs.store
    df = inputs.df

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
    return TrainingBundle(
        rep=rep,
        plan=plan,
        train_df=train_df,
        train_ids=train_ids,
        Y_train=Y_train,
        R_train=R_train,
        y_dim=y_dim,
    )


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

    if cand_df.shape[0] == 0:
        raise OpalError("Candidate pool is empty after filtering; nothing to score.")

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
    return XBundle(
        X_train=X_train,
        id_order_train=id_order_train,
        X_pool=X_pool,
        id_order_pool=id_order_pool,
        cand_df=cand_df,
    )


def stage_scoring(
    *,
    inputs: RoundInputs,
    rctx: RoundCtx,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    R_train: np.ndarray,
    X_pool: np.ndarray,
    id_order_train: List[str],
    id_order_pool: List[str],
    y_dim: int,
) -> ScoreBundle:
    cfg = inputs.cfg
    req = inputs.req
    rdir = inputs.rdir

    yops_cfg = getattr(cfg.training, "y_ops", []) or []
    _log(
        req.verbose,
        f"[y-ops] applying {len(yops_cfg)} op(s) to training labels: {([p.name for p in yops_cfg] or [])}",
    )
    append_round_log_event(
        rdir / "logs" / "round.log.jsonl",
        {
            "ts": now_iso(),
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
                rdir / "logs" / "round.log.jsonl",
                {
                    "ts": now_iso(),
                    "stage": "predict_batch",
                    "batch": int(bi + 1),
                    "of": int(num_batches),
                    "rows": int(batch_X.shape[0]),
                },
            )
    Y_hat_fit = np.vstack(yhat_chunks) if yhat_chunks else np.zeros((0, y_dim), dtype=float)

    contract = getattr(model, "__opal_contract__", None)
    produces_by_stage = getattr(contract, "produces_by_stage", None) or {}
    if produces_by_stage.get("predict"):
        # Enforce predict-stage produces once after batching.
        mctx.postcheck_produces(stage="predict")

    _missing = object()
    std_payload = mctx.get("model/<self>/std_devs", _missing)
    y_pred_std_fit = None if std_payload is _missing else _coalesce_uncertainty_chunks(std_payload)

    _log(
        req.verbose,
        f"[y-ops] inverting {len(yops_cfg)} op(s) for predictions: {([p.name for p in yops_cfg] or [])}",
    )
    # UQ invariant: objectives consume mean and standard deviation in the same units.
    Y_hat, y_pred_std = _inverse_yops_outputs(
        rctx=rctx,
        y_ops_cfg=yops_cfg,
        y_pred=Y_hat_fit,
        y_pred_std=y_pred_std_fit,
    )
    append_round_log_event(
        rdir / "logs" / "round.log.jsonl",
        {
            "ts": now_iso(),
            "stage": "yops_inverse_done",
            "ops": [p.name for p in yops_cfg],
        },
    )
    if Y_hat.shape[1] != y_dim:
        raise OpalError(f"Predicted Y dimension mismatch: expected {y_dim}, got {Y_hat.shape[1]}")

    rctx.set_core("core/labels_as_of_round", int(req.as_of_round))

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

        def iter_labels_y_current_round(self):
            mask = self._R == self._as
            for i in np.where(mask)[0].tolist():
                yield self._Y[i, :]

    tv = _TrainView(Y_train, R_train, int(req.as_of_round))

    score_channels: Dict[str, np.ndarray] = {}
    uncertainty_channels: Dict[str, np.ndarray] = {}
    channel_modes: Dict[str, str] = {}
    diagnostics_by_objective: Dict[str, Dict[str, Any]] = {}
    objective_defs: List[Dict[str, Any]] = []

    for obj_ref in cfg.objectives.objectives:
        obj_name_i = obj_ref.name
        obj_params_i = dict(obj_ref.params)
        obj_fn = get_objective(obj_name_i)
        octx = rctx.for_plugin(category="objective", name=obj_name_i, plugin=obj_fn)
        try:
            raw_obj = obj_fn(
                y_pred=Y_hat,
                params=obj_params_i,
                ctx=octx,
                train_view=tv,
                y_pred_std=y_pred_std,
            )
        except OpalError:
            raise
        except Exception as e:
            raise OpalError(f"Objective plugin '{obj_name_i}' failed: {e}") from e
        obj_res = validate_objective_result_v2(
            result=raw_obj,
            objective_name=obj_name_i,
            n_rows=len(id_order_pool),
        )
        diagnostics_by_objective[obj_name_i] = dict(obj_res.diagnostics or {})

        score_refs_for_obj: List[str] = []
        for channel_name, arr in obj_res.scores_by_name.items():
            ref = f"{obj_name_i}/{channel_name}"
            if ref in score_channels:
                raise OpalError(f"Duplicate score channel reference generated: {ref}")
            score_channels[ref] = arr
            channel_modes[ref] = obj_res.modes_by_name[channel_name]
            score_refs_for_obj.append(ref)

        uncertainty_refs_for_obj: List[str] = []
        for channel_name, arr in obj_res.uncertainty_by_name.items():
            ref = f"{obj_name_i}/{channel_name}"
            if ref in uncertainty_channels:
                raise OpalError(f"Duplicate uncertainty channel reference generated: {ref}")
            uncertainty_channels[ref] = arr
            uncertainty_refs_for_obj.append(ref)

        objective_defs.append(
            {
                "name": obj_name_i,
                "params": obj_params_i,
                "score_channels": score_refs_for_obj,
                "uncertainty_channels": uncertainty_refs_for_obj,
                "diagnostics_summary_keys": list((obj_res.diagnostics or {}).keys()),
            }
        )

        summary_stats = (obj_res.diagnostics or {}).get("summary_stats", {})
        if isinstance(summary_stats, dict) and summary_stats:
            kvs = _format_summary_stats_for_log(summary_stats)
            _log(req.verbose, f"[objective:{obj_name_i}] " + " | ".join(kvs))

    sel_name = cfg.selection.selection.name
    sel_params = dict(cfg.selection.selection.params)
    if req.k_override is not None:
        top_k = int(req.k_override)
    else:
        if "top_k" not in sel_params:
            raise OpalError("selection.params.top_k is required (or override with --k).")
        top_k = int(sel_params.get("top_k"))
    if top_k <= 0:
        raise OpalError("selection.params.top_k must be > 0 (or override with --k).")
    tie_handling = resolve_selection_tie_handling(sel_params, error_cls=OpalError)
    sel_params["tie_handling"] = tie_handling
    mode = resolve_selection_objective_mode(sel_params, error_cls=OpalError)
    sel_params["objective_mode"] = mode
    score_ref = str(sel_params.get("score_ref", "")).strip()
    if not score_ref:
        raise OpalError("selection.params.score_ref is required and must reference an objective score channel.")
    y_obj_scalar = _resolve_channel_ref(score_ref, score_channels, label="score_ref")
    if score_ref not in channel_modes:
        raise OpalError(f"score_ref channel '{score_ref}' is missing objective mode metadata.")
    selected_mode = str(channel_modes[score_ref]).strip().lower()
    if selected_mode != mode:
        raise OpalError(
            "Objective mode mismatch: selected score channel "
            f"{score_ref!r} has mode {selected_mode!r} but selection objective_mode is {mode!r}."
        )

    uncertainty_ref = sel_params.get("uncertainty_ref")
    if uncertainty_ref is not None:
        uncertainty_ref = str(uncertainty_ref).strip() or None
    if sel_name == "expected_improvement" and not uncertainty_ref:
        raise OpalError("selection.params.uncertainty_ref is required for expected_improvement.")
    sq = (
        _resolve_channel_ref(uncertainty_ref, uncertainty_channels, label="uncertainty_ref")
        if uncertainty_ref
        else None
    )

    sel_fn = get_selection(sel_name, sel_params)
    sctx = rctx.for_plugin(category="selection", name=sel_name, plugin=sel_fn)
    selection_call_params = extract_selection_plugin_params(sel_params)
    try:
        raw_sel = sel_fn(
            ids=np.array(id_order_pool),
            scores=y_obj_scalar,
            top_k=top_k,
            tie_handling=tie_handling,
            objective=mode,
            ctx=sctx,
            scalar_uncertainty=sq,
            **selection_call_params,
        )
    except OpalError:
        raise
    except Exception as e:
        raise OpalError(f"Selection plugin '{sel_name}' failed: {e}") from e
    validated_sel = validate_selection_result(raw_sel, plugin_name=sel_name, expected_len=len(id_order_pool))
    sel_norm = normalize_selection_result(
        {"order_idx": validated_sel.order_idx},
        ids=np.array(id_order_pool),
        scores=validated_sel.score,
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
    selection_line = (
        f"[selection:{sel_name}] objective={mode} tie={tie_handling} "
        f"requested_top_k={top_k} → selected={selected_effective}"
    )
    _log(
        req.verbose,
        selection_line,
    )
    append_round_log_event(
        rdir / "logs" / "round.log.jsonl",
        {
            "ts": now_iso(),
            "stage": "objective_done",
            "score_ref": score_ref,
            "uncertainty_ref": uncertainty_ref,
            "objective_count": len(objective_defs),
            "score_min": float(np.nanmin(y_obj_scalar)) if y_obj_scalar.size else None,
            "score_median": (float(np.nanmedian(y_obj_scalar)) if y_obj_scalar.size else None),
            "score_max": float(np.nanmax(y_obj_scalar)) if y_obj_scalar.size else None,
        },
    )
    append_round_log_event(
        rdir / "logs" / "round.log.jsonl",
        {
            "ts": now_iso(),
            "stage": "selection_done",
            "strategy": sel_name,
            "tie_handling": tie_handling,
            "objective_mode": mode,
            "score_ref": score_ref,
            "uncertainty_ref": uncertainty_ref,
            "requested_top_k": int(top_k),
            "effective_after_ties": int(selected_effective),
        },
    )
    rctx.set_core("core/selection/top_k_requested", int(top_k))
    rctx.set_core("core/selection/top_k_effective", int(selected_effective))
    rctx.set_core("core/selection/objective_mode", str(mode))
    rctx.set_core("core/selection/tie_handling", str(tie_handling))
    rctx.set_core("core/selection/score_ref", str(score_ref))
    rctx.set_core("core/selection/uncertainty_ref", uncertainty_ref)

    selected_obj_name, _ = score_ref.split("/", 1)
    selected_obj_params = next((d["params"] for d in objective_defs if d["name"] == selected_obj_name), {})
    selected_diag = diagnostics_by_objective.get(selected_obj_name, {})
    obj_summary_stats = selected_diag.get("summary_stats") if isinstance(selected_diag, dict) else None
    obj_meta = {
        "objectives": objective_defs,
        "selection": {
            "score_ref": score_ref,
            "uncertainty_ref": uncertainty_ref,
            "objective_mode": mode,
            "tie_handling": tie_handling,
        },
    }
    obj_sha = write_objective_meta(rdir / "metadata" / "objective_meta.json", obj_meta)

    return ScoreBundle(
        model=model,
        fit_metrics=fit_metrics,
        fit_duration=fit_duration,
        Y_hat=Y_hat,
        y_obj_scalar=y_obj_scalar,
        diag=selected_diag,
        obj_summary_stats=obj_summary_stats if isinstance(obj_summary_stats, dict) else None,
        obj_name=selected_obj_name,
        obj_params=selected_obj_params,
        obj_mode=selected_mode,
        objective_defs=objective_defs,
        score_channels=score_channels,
        uncertainty_channels=uncertainty_channels,
        score_ref=score_ref,
        uncertainty_ref=uncertainty_ref,
        sel_name=sel_name,
        sel_params=sel_params,
        tie_handling=tie_handling,
        mode=mode,
        ranks_competition=ranks_competition,
        selected_bool=selected_bool,
        selected_effective=selected_effective,
        top_k=top_k,
        obj_sha=obj_sha,
        scores=validated_sel.score,
        uq_scalar=sq,
    )
