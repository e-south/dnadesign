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
    round_dir,
    write_feature_importance,
    write_predictions_with_uncertainty,
    write_round_metrics,
    write_selection_csv,
)
from ...config import load_config
from ...locks import CampaignLock
from ...preflight import preflight_run
from ...registries.models import get_model
from ...registries.objectives import get_objective
from ...selection.top_n import select_top_n_df
from ...state import CampaignState, RoundEntry
from ...utils import (
    ExitCodes,
    OpalError,
    ensure_dir,
    file_sha256,
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
                cfg.training.model.name,
                cfg.training.model.params.__dict__,
                cfg.training.model.target_scaler.__dict__,
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
            obj_fn = get_objective(cfg.selection.objective.name)
            ypred_mat = np.array(
                [np.array(v, dtype=float) for v in scored["y_pred_vec"].tolist()],
                dtype=float,
            )
            e_labels_for_scaling = []
            for v in y_vals:
                if isinstance(v, list) and len(v) >= 5:
                    e_labels_for_scaling.append(float(v[4]))
            e_labels_for_scaling = np.array(e_labels_for_scaling, dtype=float)
            obj_res = obj_fn(
                y_pred_vec5=ypred_mat,
                params=cfg.selection.objective.params.__dict__,
                e_labels_for_scaling=e_labels_for_scaling,
            )
            scored["selection_score"] = obj_res.score
            scored["logic_fidelity_l2_norm01"] = getattr(
                obj_res, "logic_fidelity", np.nan
            )
            scored["effect_scaled"] = getattr(obj_res, "effect_scaled", np.nan)

            # deterministic order and selection
            scored = scored.sort_values(
                by=["selection_score", "id"], ascending=[False, True]
            ).reset_index(drop=True)
            scored_ranked, top_k_effective = select_top_n_df(
                scored, score_col="selection_score", top_k=int(k)
            )

            # artifacts
            apaths = ArtifactPaths(
                model=rdir / "model.joblib",
                selection_csv=rdir / "selection_top_k.csv",
                feature_importance_csv=rdir / "feature_importance.csv",
                metrics_json=rdir / "round_model_metrics.json",
                round_log_jsonl=rdir / "round.log.jsonl",
                preds_with_uncertainty_csv=rdir / "predictions_with_uncertainty.csv",
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
                "name": cfg.data.representation_transform.name,
                "params": cfg.data.representation_transform.params,
            }
            st.training_policy = cfg.training.policy
            st.performance = {
                "score_batch_size": score_batch_size or cfg.scoring.score_batch_size,
                "objective": cfg.selection.objective.name,
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
                        "type": cfg.training.model.name,
                        "params": cfg.training.model.params.__dict__,
                        "artifact_path": str(apaths.model.resolve()),
                        "artifact_sha256": model_sha,
                    },
                    metrics=met,
                    durations_sec={"fit": fit_dt, "score": score_dt},
                    seeds={
                        "global": cfg.training.model.params.random_state,
                        "model": cfg.training.model.params.random_state,
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
                        "checksums": {
                            "selection_top_k_csv": sel_sha,
                            "feature_importance_csv": fi_sha,
                            "predictions_with_uncertainty_csv": preds_sha,
                            "round_model_metrics_json": met_sha,
                        },
                    },
                    writebacks={
                        "records_columns_written": [
                            f"opal__{cfg.campaign.slug}__r{round}__pred_y",
                            f"opal__{cfg.campaign.slug}__r{round}__selection_score__{cfg.selection.objective.name}",
                            f"opal__{cfg.campaign.slug}__r{round}__rank_competition",
                            f"opal__{cfg.campaign.slug}__r{round}__selected_top_k_bool",
                            f"opal__{cfg.campaign.slug}__r{round}__uncertainty__mean_all_std",
                        ]
                    },
                    warnings=[],
                )
            )
            st.save(st_path)

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
