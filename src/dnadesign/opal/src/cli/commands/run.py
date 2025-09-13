"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/run.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from ...artifacts import (
    ArtifactPaths,
    append_round_log_event,
    round_dir,
    write_feature_importance,
    write_objective_meta,
    write_predictions_with_uncertainty,
    write_round_ctx,
    write_round_metrics,
    write_selection_csv,
)
from ...config import load_config
from ...locks import CampaignLock
from ...preflight import preflight_run
from ...registries.models import get_model
from ...registries.objectives import get_objective
from ...registries.selections import get_selection
from ...round_context import RoundContext
from ...state import CampaignState, RoundEntry
from ...utils import (
    ExitCodes,
    OpalError,
    ensure_dir,
    file_sha256,
    now_iso,
)
from ...writebacks import write_round_columns
from ..registry import cli_command
from ._common import internal_error, json_out, resolve_config_path, store_from_cfg


@cli_command(
    "run",
    help="Train on labels ≤ round, score, select top-k, write artifacts & records.",
)
def cmd_run(
    config: Path = typer.Option(
        None, "--config", "-c", envvar="OPAL_CONFIG", help="campaign.yaml"
    ),
    round: int = typer.Option(..., "--round", "-r"),
    k: Optional[int] = typer.Option(None, "--k", help="Top-k (default from YAML)"),
    resume: bool = typer.Option(
        False, "--resume", help="Allow overwrites for this round"
    ),
    force: bool = typer.Option(False, "--force", help="Force write-backs if needed"),
    score_batch_size: Optional[int] = typer.Option(
        None, "--score-batch-size", help="Override batch size"
    ),
):
    try:
        cfg = load_config(resolve_config_path(config))
        if k is None:
            k = cfg.selection.top_k_default

        workdir = Path(cfg.campaign.workdir)
        rdir = round_dir(workdir, round)
        ensure_dir(rdir)

        # lock — single writer per campaign
        lock = CampaignLock(workdir)
        with lock:
            store = store_from_cfg(cfg)
            df = store.load()

            # preflight
            rep = preflight_run(
                store, df, round, cfg.safety.fail_on_mixed_biotype_or_alphabet
            )

            # gather labels ≤ round
            labels_eff = store.effective_labels_latest_only(df, round)
            if labels_eff.empty:
                raise OpalError("No effective labels to train on.")
            y_vals = labels_eff["y"].tolist()
            y_mat = np.array(
                [
                    (
                        np.array(v, dtype=float)
                        if isinstance(v, list)
                        else np.array([float(v)])
                    )
                    for v in y_vals
                ],
                dtype=float,
            )

            tr_ids = labels_eff["id"].tolist()
            Xtr, _ = store.transform_matrix(df, tr_ids)

            # candidate universe
            cand = store.candidate_universe(df, round)
            cand_ids = cand["id"].tolist()

            # model
            mdl = get_model(
                cfg.training.models.name,
                cfg.training.models.params,
                cfg.training.target_scaler.__dict__,
            )

            import time

            t0 = time.perf_counter()
            metrics = mdl.fit(Xtr, y_mat)
            fit_dt = time.perf_counter() - t0

            # predict in batches
            sbatch = score_batch_size or cfg.scoring.score_batch_size
            preds, stds = [], []
            t1 = time.perf_counter()
            for i in range(0, len(cand_ids), sbatch):
                chunk_ids = cand_ids[i : i + sbatch]
                Xc, _ = store.transform_matrix(df, chunk_ids)
                yh = mdl.predict(Xc)  # (n,d)
                per_tree = mdl.predict_per_tree(Xc)  # (T,n,d)
                ystd = per_tree.std(axis=0)  # (n,d)
                preds.extend(
                    [(cid, list(map(float, row))) for cid, row in zip(chunk_ids, yh)]
                )
                stds.extend(
                    [(cid, list(map(float, row))) for cid, row in zip(chunk_ids, ystd)]
                )
            score_dt = time.perf_counter() - t1

            scored = pd.DataFrame(preds, columns=["id", "y_pred_vec"]).merge(
                pd.DataFrame(stds, columns=["id", "y_pred_std_vec"]),
                on="id",
                how="left",
            )
            scored["uncertainty_mean_all_std"] = scored["y_pred_std_vec"].map(
                lambda v: (
                    float(np.mean(v))
                    if isinstance(v, list) and len(v) > 0
                    else float("nan")
                )
            )
            seq_map = df.set_index("id")["sequence"].to_dict()
            scored["sequence"] = scored["id"].map(seq_map)

            # objective
            obj_name = cfg.objective.objective.name
            obj_params = cfg.objective.objective.params
            obj_fn = get_objective(obj_name)

            ypred_mat = np.array(
                [np.array(v, dtype=float) for v in scored["y_pred_vec"].tolist()],
                dtype=float,
            )
            # Build RoundContext (includes effect scaling pool from labels)
            setpoint = list(obj_params.get("setpoint_vector", [0, 0, 0, 1]))
            delta = float(obj_params.get("intensity_log2_offset_delta", 0.0))
            # compute effect pool from labels (vec8) -> y_lin -> w·y_lin
            effect_pool: list[float] = []
            for v in y_vals:
                if isinstance(v, list) and len(v) == 8:
                    ystar = np.asarray(v[4:8], dtype=float)
                    ylin = np.maximum(0.0, np.power(2.0, ystar) - delta)
                    s = np.asarray(setpoint, dtype=float)
                    w = (
                        np.zeros_like(s)
                        if float(np.sum(s)) <= 0
                        else (s / float(np.sum(s)))
                    )
                    effect_pool.append(float(np.sum(ylin * w)))
            percentile_cfg = obj_params.get("scaling", {})
            run_id = f"r{round}-{now_iso()}"
            rctx = RoundContext.build(
                slug=cfg.campaign.slug,
                round_index=int(round),
                setpoint=setpoint,
                label_ids=[str(_id) for _id in tr_ids],
                effect_pool_for_scaling=effect_pool,
                percentile_cfg={
                    "p": int(percentile_cfg.get("percentile", 95)),
                    "fallback_p": int(percentile_cfg.get("fallback_p", 75)),
                    "min_n": int(percentile_cfg.get("min_n", 5)),
                    "eps": float(percentile_cfg.get("eps", 1e-8)),
                },
                model_name=cfg.training.models.name,
                model_params=cfg.training.models.params,
                x_transform={
                    "name": cfg.data.transforms_x.name,
                    "params": cfg.data.transforms_x.params,
                },
                y_ingest_transform={
                    "name": cfg.data.transforms_y.name,
                    "params": cfg.data.transforms_y.params,
                },
                y_expected_length=cfg.data.y_expected_length,
                run_id=run_id,
            )

            obj_res = obj_fn(
                y_pred=ypred_mat,
                params=obj_params,
                round_ctx=rctx,
            )
            scored["selection_score"] = obj_res.score
            diag = obj_res.diagnostics or {}
            scored["logic_fidelity_l2_norm01"] = diag.get("logic_fidelity", np.nan)
            scored["effect_scaled"] = diag.get("effect_scaled", np.nan)

            # flags: simple QC
            def _flags_row(vpred: list[float], e_scaled: float) -> str:
                flags = []
                arr = np.asarray(vpred, dtype=float)
                if not np.all(np.isfinite(arr)):
                    flags.append("nan_pred")
                if float(e_scaled) >= 1.0:
                    flags.append("clip_effect")
                return "|".join(flags)

            scored["flags"] = [
                _flags_row(v, e)
                for v, e in zip(scored["y_pred_vec"], scored["effect_scaled"])
            ]

            # deterministic order and selection via registry
            ids_arr = scored["id"].astype(str).to_numpy()
            sel_fn = get_selection(cfg.selection.selection.name)
            sel_res = sel_fn(
                ids=ids_arr,
                scores=scored["selection_score"].to_numpy(dtype=float),
                top_k=int(k),
                tie_handling=cfg.selection.tie_handling or "competition_rank",
            )
            order = sel_res["order_idx"]
            scored_ranked = scored.iloc[order].copy().reset_index(drop=True)
            scored_ranked["rank_competition"] = sel_res["ranks"].astype(int)
            scored_ranked["selected_top_k_bool"] = sel_res["selected_bool"].astype(bool)
            top_k_effective = int(scored_ranked["selected_top_k_bool"].sum())

            # artifacts
            apaths = ArtifactPaths(
                model=rdir / "model.joblib",
                selection_csv=rdir / "selection_top_k.csv",
                feature_importance_csv=rdir / "feature_importance.csv",
                metrics_json=rdir / "round_model_metrics.json",
                round_log_jsonl=rdir / "round.log.jsonl",
                preds_with_uncertainty_csv=rdir / "predictions_with_uncertainty.csv",
                round_ctx_json=rdir / "round_ctx.json",
                objective_meta_json=rdir / "objective_meta.json",
            )
            mdl.save(str(apaths.model))
            model_sha = file_sha256(apaths.model)

            sel_sha = write_selection_csv(
                apaths.selection_csv,
                scored_ranked[scored_ranked["selected_top_k_bool"]][
                    ["id", "sequence", "selection_score"]
                ],
            )
            fi_sha = write_feature_importance(
                apaths.feature_importance_csv, mdl.feature_importances()
            )
            met = {
                "random_forest_oob_r2_score": metrics.oob_r2,
                "random_forest_oob_mse": metrics.oob_mse,
                "fit_seconds": fit_dt,
                "score_seconds": score_dt,
            }
            met_sha = write_round_metrics(apaths.metrics_json, met)

            # write round_ctx and objective_meta
            ctx_sha = write_round_ctx(apaths.round_ctx_json, rctx.to_dict())
            obj_meta = {
                "denom_percentile": rctx.percentile_cfg.get("p"),
                "denom_value": float(diag.get("denom_used", float("nan"))),
                "beta": float(obj_params.get("logic_exponent_beta", 1.0)),
                "gamma": float(obj_params.get("intensity_exponent_gamma", 1.0)),
            }
            objm_sha = write_objective_meta(apaths.objective_meta_json, obj_meta)

            preds_out = scored_ranked[
                [
                    "id",
                    "sequence",
                    "y_pred_vec",
                    "y_pred_std_vec",
                    "uncertainty_mean_all_std",
                    "selection_score",
                    "rank_competition",
                    "selected_top_k_bool",
                ]
            ].copy()
            preds_sha = write_predictions_with_uncertainty(
                apaths.preds_with_uncertainty_csv, preds_out
            )

            # write-backs
            df2 = write_round_columns(
                df,
                cfg.campaign.slug,
                round,
                scored_ranked.rename(columns={"y_pred_vec": "y_pred_vec"}),
                allow_overwrite=resume or force,
                objective_name=obj_name,
                fingerprint_short=rctx.fingerprint_short,
            )
            store.save_atomic(df2)

            # update state
            st_path = Path(cfg.campaign.workdir) / "state.json"
            st = CampaignState.load(st_path) if st_path.exists() else CampaignState()
            st.campaign_slug = cfg.campaign.slug
            st.campaign_name = cfg.campaign.name
            st.workdir = str(workdir.resolve())
            st.representation_column_name = cfg.data.representation_column_name
            st.label_source_column_name = cfg.data.label_source_column_name
            st.representation_vector_dimension = rep.x_dim
            st.representation_transform = {
                "name": cfg.data.transforms_x.name,
                "params": cfg.data.transforms_x.params,
            }
            st.training_policy = cfg.training.policy
            st.performance = {
                "score_batch_size": score_batch_size or cfg.scoring.score_batch_size,
                "objective": obj_name,
            }
            st.data_location = {
                "kind": store.kind,
                "records_path": str(store.records_path.resolve()),
                "records_sha256": file_sha256(store.records_path),
            }
            st.add_round(
                RoundEntry(
                    round_index=round,
                    round_name=f"round_{round}",
                    round_dir=str(rdir.resolve()),
                    labels_used_rounds=list(range(0, round + 1)),
                    number_of_training_examples_used_in_round=int(len(tr_ids)),
                    number_of_candidates_scored_in_round=int(len(scored_ranked)),
                    selection_top_k_requested=int(k),
                    selection_top_k_effective_after_ties=int(top_k_effective),
                    model={
                        "type": cfg.training.models.name,
                        "params": cfg.training.models.params,
                        "artifact_path": str(apaths.model.resolve()),
                        "artifact_sha256": model_sha,
                    },
                    metrics=met,
                    durations_sec={"fit": fit_dt, "score": score_dt},
                    seeds={
                        "global": cfg.training.models.params.get("random_state", None),
                        "model": cfg.training.models.params.get("random_state", None),
                    },
                    artifacts={
                        "selection_top_k_csv": str(apaths.selection_csv.resolve()),
                        "feature_importance_csv": str(
                            apaths.feature_importance_csv.resolve()
                        ),
                        "predictions_with_uncertainty_csv": str(
                            apaths.preds_with_uncertainty_csv.resolve()
                        ),
                        "round_model_metrics_json": str(apaths.metrics_json.resolve()),
                        "round_ctx_json": str(apaths.round_ctx_json.resolve()),
                        "objective_meta_json": str(
                            apaths.objective_meta_json.resolve()
                        ),
                        "checksums": {
                            "selection_top_k_csv": sel_sha,
                            "feature_importance_csv": fi_sha,
                            "predictions_with_uncertainty_csv": preds_sha,
                            "round_model_metrics_json": met_sha,
                            "round_ctx_json": ctx_sha,
                            "objective_meta_json": objm_sha,
                        },
                    },
                    writebacks={
                        "records_columns_written": [
                            f"opal__{cfg.campaign.slug}__r{round}__pred_y",
                            f"opal__{cfg.campaign.slug}__r{round}__selection_score__{obj_name}",
                            f"opal__{cfg.campaign.slug}__r{round}__rank_competition",
                            f"opal__{cfg.campaign.slug}__r{round}__selected_top_k_bool",
                            f"opal__{cfg.campaign.slug}__r{round}__uncertainty__mean_all_std",
                            f"opal__{cfg.campaign.slug}__r{round}__flags",
                            f"opal__{cfg.campaign.slug}__r{round}__fingerprint",
                        ]
                    },
                    warnings=[],
                )
            )
            st.save(st_path)

            # round log events (append-only)
            append_round_log_event(
                apaths.round_log_jsonl,
                {
                    "ts": now_iso(),
                    "round": int(round),
                    "stage": "fit",
                    "labels": int(len(tr_ids)),
                    "oob_r2": metrics.oob_r2,
                },
            )
            append_round_log_event(
                apaths.round_log_jsonl,
                {
                    "ts": now_iso(),
                    "round": int(round),
                    "stage": "predict",
                    "universe": int(len(cand_ids)),
                },
            )
            append_round_log_event(
                apaths.round_log_jsonl,
                {
                    "ts": now_iso(),
                    "round": int(round),
                    "stage": "objective",
                    "denom_p": rctx.percentile_cfg.get("p"),
                    "denom": float(diag.get("denom_used", float("nan"))),
                },
            )
            append_round_log_event(
                apaths.round_log_jsonl,
                {
                    "ts": now_iso(),
                    "round": int(round),
                    "stage": "selection",
                    "top_k": int(k),
                    "effective": int(top_k_effective),
                },
            )

            json_out(
                {
                    "trained_on": len(tr_ids),
                    "scored": len(scored_ranked),
                    "selection_top_k_effective_after_ties": top_k_effective,
                }
            )

    except OpalError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("run", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
