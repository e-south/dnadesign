# ABOUTME: Executes one Opal round from training through selection and writebacks.
# ABOUTME: Coordinates round stages, artifacts, ledgers, and state updates.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/runtime/run_round.py

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


def _clear_round_dir(rdir: Path) -> None:
    if not rdir.exists():
        return
    for child in rdir.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except Exception as exc:
            raise OpalError(f"Failed to clear round directory {rdir}: {exc}") from exc


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

    objective_meta = {"name": score.obj_name, "params": score.obj_params, "mode": score.mode}
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
        rdir / "round.log.jsonl",
        {
            "ts": now_iso(),
            "round": int(req.as_of_round),
            "stage": "fit",
            "oob_r2": getattr(score.fit_metrics, "oob_r2", None),
        },
    )
    append_round_log_event(
        rdir / "round.log.jsonl",
        {
            "ts": now_iso(),
            "round": int(req.as_of_round),
            "stage": "selection",
            "top_k": int(score.top_k),
            "effective": int(score.selected_effective),
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
        trained_on=len(xbundle.id_order_train),
        scored=len(xbundle.id_order_pool),
        top_k_requested=int(score.top_k),
        top_k_effective=int(score.selected_effective),
        ledger_path=str(ws.ledger_dir.resolve()),
    )
