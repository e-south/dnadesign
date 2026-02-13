"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/runtime/round/writebacks.py

Handles round artifact persistence and ledger/state writebacks. Centralizes run
meta/pred event construction and record updates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ...config.types import RootConfig
from ...core.round_context import RoundCtx
from ...core.utils import OpalError, ensure_dir, file_sha256, now_iso, print_stderr
from ...storage.artifacts import (
    ArtifactPaths,
    append_round_log_event,
    write_feature_importance_csv,
    write_labels_used_parquet,
    write_model_meta,
    write_round_ctx,
    write_selection_csv,
)
from ...storage.data_access import RecordsStore
from ...storage.ledger import LedgerWriter
from ...storage.state import CampaignState, RoundEntry
from ...storage.workspace import CampaignWorkspace
from ...storage.writebacks import (
    SelectionEmit,
    build_run_meta_event,
    build_run_pred_events,
)
from .contracts import ArtifactBundle, RoundInputs, ScoreBundle, TrainingBundle, XBundle


@dataclass(frozen=True)
class RunEvents:
    run_pred_events: pd.DataFrame
    run_meta_event: pd.DataFrame


def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print_stderr(msg)


def build_sequence_map(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    if "sequence" not in df.columns:
        return {}
    seq_map: Dict[str, Optional[str]] = {}
    for _id, seq in df[["id", "sequence"]].itertuples(index=False, name=None):
        seq_map[str(_id)] = None if pd.isna(seq) else str(seq)
    return seq_map


def write_round_artifacts(
    *,
    inputs: RoundInputs,
    run_id: str,
    rctx: RoundCtx,
    training: TrainingBundle,
    xbundle: XBundle,
    score: ScoreBundle,
) -> ArtifactBundle:
    cfg = inputs.cfg
    req = inputs.req
    rdir = inputs.rdir
    df = inputs.df

    apaths = ArtifactPaths(
        model=rdir / "model" / "model.joblib",
        model_meta_json=rdir / "model" / "model_meta.json",
        selection_csv=rdir / "selection" / "selection_top_k.csv",
        round_log_jsonl=rdir / "logs" / "round.log.jsonl",
        round_ctx_json=rdir / "metadata" / "round_ctx.json",
        objective_meta_json=rdir / "metadata" / "objective_meta.json",
        labels_used_parquet=rdir / "labels" / "labels_used.parquet",
        feature_importance_csv=rdir / "model" / "feature_importance.csv",
    )
    ensure_dir(apaths.model.parent)
    score.model.save(str(apaths.model))
    model_meta = {
        "model__name": cfg.model.name,
        "model__params": score.model.get_params(),
        "training__y_ops": [{"name": p.name, "params": p.params} for p in (cfg.training.y_ops or [])],
        "x_dim": int(xbundle.X_train.shape[1]),
        "y_dim": int(training.y_dim),
    }
    model_meta_sha = write_model_meta(apaths.model_meta_json, model_meta)

    seq_map = build_sequence_map(df)

    labels_used_df = None
    if not training.train_df.empty:
        labels_used_df = pd.DataFrame(
            {
                "run_id": [run_id] * len(training.train_df),
                "as_of_round": [int(req.as_of_round)] * len(training.train_df),
                "observed_round": training.train_df["r"].astype(int).tolist(),
                "id": training.train_df["id"].astype(str).tolist(),
                "sequence": [seq_map.get(i) for i in training.train_df["id"].astype(str).tolist()],
                "y_obs": training.train_df["y"].tolist(),
                "src": ["training_snapshot"] * len(training.train_df),
                "note": [f"labels used in run {run_id} (as_of_round={int(req.as_of_round)})"] * len(training.train_df),
            }
        )

    selected_ids = [str(i) for i in xbundle.id_order_pool]
    selected_df = pd.DataFrame(
        {
            "id": selected_ids,
            "sequence": [seq_map.get(i) for i in selected_ids],
            "sel__rank_competition": score.ranks_competition,
            "pred__y_obj_scalar": score.y_obj_scalar,
            "sel__is_selected": score.selected_bool,
        }
    )
    selected_df = selected_df.loc[lambda d: d["sel__is_selected"]].sort_values("sel__rank_competition")
    selected_df = selected_df.assign(
        run_id=run_id,
        as_of_round=int(req.as_of_round),
        campaign_slug=cfg.campaign.slug,
        selection_name=score.sel_name,
        objective_name=score.obj_name,
        objective_mode=score.mode,
        tie_handling=score.tie_handling,
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
            "pred__y_obj_scalar",
        ]
    ]
    sel_sha = write_selection_csv(apaths.selection_csv, selected_df)
    sel_run_csv_path = apaths.selection_csv.with_name(f"selection_top_k__run_{run_id}.csv")
    sel_run_csv_sha = write_selection_csv(sel_run_csv_path, selected_df)

    ctx_sha = write_round_ctx(apaths.round_ctx_json, rctx.snapshot())
    labels_used_sha = write_labels_used_parquet(apaths.labels_used_parquet, labels_used_df)

    artifacts_paths_and_hashes: Dict[str, tuple[str, str]] = {}

    if hasattr(score.model, "model_artifacts") and callable(getattr(score.model, "model_artifacts")):
        try:
            m_arts = score.model.model_artifacts() or {}
            if "feature_importance" in m_arts:
                fi_df = m_arts["feature_importance"]
                fi_path = apaths.feature_importance_csv
                fi_sha = write_feature_importance_csv(fi_path, fi_df)
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
                    apaths.round_log_jsonl,
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
                artifacts_paths_and_hashes["model/feature_importance.csv"] = (
                    fi_sha,
                    str(fi_path.resolve()),
                )
        except Exception as e:
            raise OpalError(f"Failed to persist model artifacts: {e}")

    _log(
        req.verbose,
        "[artifacts] model, selection_csv, round_ctx, objective_meta"
        + (", feature_importance" if "model/feature_importance.csv" in artifacts_paths_and_hashes else "")
        + " written",
    )

    artifacts_paths_and_hashes.update(
        {
            "model/model.joblib": (
                file_sha256(apaths.model),
                str(apaths.model.resolve()),
            ),
            "selection/selection_top_k.csv": (
                sel_sha,
                str(apaths.selection_csv.resolve()),
            ),
            f"selection/selection_top_k__run_{run_id}.csv": (
                sel_run_csv_sha,
                str(sel_run_csv_path.resolve()),
            ),
            "metadata/round_ctx.json": (ctx_sha, str(apaths.round_ctx_json.resolve())),
            "metadata/objective_meta.json": (
                score.obj_sha,
                str(apaths.objective_meta_json.resolve()),
            ),
            "model/model_meta.json": (
                model_meta_sha,
                str(apaths.model_meta_json.resolve()),
            ),
            "labels/labels_used.parquet": (
                labels_used_sha,
                str(apaths.labels_used_parquet.resolve()),
            ),
        }
    )

    return ArtifactBundle(
        apaths=apaths,
        selected_df=selected_df,
        labels_used_df=labels_used_df,
        artifacts_paths_and_hashes=artifacts_paths_and_hashes,
    )


def build_run_events(
    *,
    inputs: RoundInputs,
    run_id: str,
    training: TrainingBundle,
    xbundle: XBundle,
    score: ScoreBundle,
    artifacts: ArtifactBundle,
) -> RunEvents:
    cfg = inputs.cfg
    req = inputs.req
    seq_map = build_sequence_map(inputs.df)
    sequences_list: List[Optional[str]] = [seq_map.get(i) for i in xbundle.id_order_pool]

    y_hat_model_sd = None
    y_obj_scalar_sd = None

    sel_emit = SelectionEmit(
        ranks_competition=score.ranks_competition,
        selected_bool=score.selected_bool,
        diagnostics=None,
    )

    run_pred_events = build_run_pred_events(
        run_id=run_id,
        as_of_round=int(req.as_of_round),
        ids=list(map(str, xbundle.id_order_pool)),
        sequences=sequences_list,
        y_hat_model=score.Y_hat,
        y_obj_scalar=score.y_obj_scalar,
        y_dim=training.y_dim,
        y_hat_model_sd=y_hat_model_sd,
        y_obj_scalar_sd=y_obj_scalar_sd,
        obj_diagnostics=score.diag,
        sel_emit=sel_emit,
        uq_scalar = score.uq_scalar,
        scores = score.scores,
    )

    run_meta_event = build_run_meta_event(
        run_id=run_id,
        as_of_round=int(req.as_of_round),
        model_name=cfg.model.name,
        model_params=score.model.get_params(),
        y_ops=[{"name": p.name, "params": p.params} for p in (cfg.training.y_ops or [])],
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params=dict(cfg.data.transforms_x.params),
        y_ingest_transform_name=cfg.data.transforms_y.name,
        y_ingest_transform_params=dict(cfg.data.transforms_y.params),
        objective_name=score.obj_name,
        objective_params=score.obj_params,
        selection_name=score.sel_name,
        selection_params=score.sel_params,
        selection_score_field="pred__y_obj_scalar",
        selection_objective_mode=score.mode,
        sel_tie_handling=score.tie_handling,
        stats_n_train=len(xbundle.id_order_train),
        stats_n_scored=len(xbundle.id_order_pool),
        unc_mean_sd=y_hat_model_sd,
        pred_rows_df=run_pred_events,
        artifact_paths_and_hashes=artifacts.artifacts_paths_and_hashes,
        objective_summary_stats=score.obj_summary_stats,
    )

    return RunEvents(run_pred_events=run_pred_events, run_meta_event=run_meta_event)


def append_ledgers(
    *,
    ws: CampaignWorkspace,
    run_pred_events: pd.DataFrame,
    run_meta_event: pd.DataFrame,
    verbose: bool,
) -> None:
    ledger = LedgerWriter(ws)
    ledger.append_run_pred(run_pred_events)
    ledger.append_run_meta(run_meta_event)
    _log(
        verbose,
        f"[ledger] appended run_pred({len(run_pred_events)}), run_meta(1) under {ws.ledger_dir}",
    )


def write_prediction_label_hist(
    *,
    store: RecordsStore,
    df: pd.DataFrame,
    ids: List[str],
    y_hat: np.ndarray,
    as_of_round: int,
    run_id: str,
    objective: Dict[str, object],
    metrics_by_name: Dict[str, List[float]],
    selection_rank: np.ndarray,
    selection_top_k: np.ndarray,
    verbose: bool,
) -> pd.DataFrame:
    df2 = store.append_predictions_from_arrays(
        df,
        ids=ids,
        y_hat=y_hat,
        as_of_round=as_of_round,
        run_id=run_id,
        objective=objective,
        metrics_by_name=metrics_by_name,
        selection_rank=selection_rank,
        selection_top_k=selection_top_k,
        ts=now_iso(),
    )
    store.save_atomic(df2)
    _log(verbose, "[writeback] records label_hist updated with run-aware predictions.")
    return df2


def update_campaign_state(
    *,
    ws: CampaignWorkspace,
    cfg: RootConfig,
    req: object,
    rep: object,
    train_df: pd.DataFrame,
    id_order_train: List[str],
    id_order_pool: List[str],
    top_k: int,
    selected_effective: int,
    apaths: ArtifactPaths,
    run_id: str,
    obj_name: str,
    store: RecordsStore,
    total_duration: float,
    fit_duration: float,
) -> None:
    st = CampaignState.load(ws.state_path)
    if getattr(req, "allow_resume", False):
        try:
            st.rounds = [r for r in st.rounds if int(getattr(r, "round_index", -1)) != int(req.as_of_round)]
        except Exception:
            pass

    st.campaign_slug = cfg.campaign.slug
    st.campaign_name = cfg.campaign.name
    st.workdir = str(ws.workdir.resolve())
    loc = cfg.data.location
    if hasattr(loc, "dataset"):
        st.data_location = {
            "kind": "usr",
            "dataset": loc.dataset,
            "path": str(Path(loc.path).resolve()),
            "records_path": str(store.records_path.resolve()),
        }
    else:
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
        "score_batch_size": getattr(req, "score_batch_size_override", None) or cfg.scoring.score_batch_size,
        "objective": obj_name,
    }
    labels_used_rounds = sorted(set(train_df["r"].astype(int).tolist()))
    round_dir = ws.round_dir(req.as_of_round)
    st.add_round(
        RoundEntry(
            round_index=int(req.as_of_round),
            run_id=str(run_id),
            round_name=f"round_{int(req.as_of_round)}",
            round_dir=str(round_dir.resolve()),
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
            durations_sec={},
            seeds={
                "global": cfg.model.params.get("random_state"),
                "model": cfg.model.params.get("random_state"),
            },
            artifacts={
                "selection_top_k_csv": str(apaths.selection_csv.resolve()),
                "ledger_predictions_dir": str(ws.ledger_predictions_dir.resolve()),
                "ledger_runs_parquet": str(ws.ledger_runs_path.resolve()),
                "ledger_labels_parquet": str(ws.ledger_labels_path.resolve()),
                "round_ctx_json": str(apaths.round_ctx_json.resolve()),
                "objective_meta_json": str(apaths.objective_meta_json.resolve()),
                "model_meta_json": str(apaths.model_meta_json.resolve()),
                "labels_used_parquet": str(apaths.labels_used_parquet.resolve()),
                "round_log_jsonl": str(apaths.round_log_jsonl.resolve()),
            },
            writebacks={
                "records_label_hist_updated": [
                    "opal__<slug>__label_hist",
                ]
            },
            warnings=[],
        )
    )
    st.rounds[-1].durations_sec = {"total": total_duration, "fit": fit_duration}
    st.save(ws.state_path)
