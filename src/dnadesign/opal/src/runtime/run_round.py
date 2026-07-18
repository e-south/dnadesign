"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/runtime/run_round.py

Executes one Opal round from training through selection and writebacks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd

from ..core.utils import OpalError, ensure_dir, now_iso, print_stderr
from ..registries.transforms_x import get_transform_x
from ..storage.artifacts import (
    RUN_ARTIFACTS_DIRECTORY,
    append_round_log_event,
    reserve_run_artifact_directory,
)
from ..storage.data_access import RecordsStore
from ..storage.ledger import LedgerWriter
from ..storage.workspace import CampaignWorkspace
from .retention import apply_runtime_artifact_retention
from .round.context import build_round_ctx
from .round.contracts import RoundInputs, RunRoundRequest, RunRoundResult
from .round.stages import stage_scoring, stage_training, stage_x_matrices
from .round.writebacks import (
    append_ledgers,
    build_run_events,
    update_campaign_state,
    write_round_artifacts,
)


def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print_stderr(msg)


def _clear_round_dir(
    rdir: Path,
    *,
    preserve_logs: bool = False,
    preserve_run_artifacts: bool = False,
) -> None:
    if not rdir.exists():
        return
    preserved_names = {
        name
        for name, preserve in (
            ("logs", preserve_logs),
            (RUN_ARTIFACTS_DIRECTORY, preserve_run_artifacts),
        )
        if preserve
    }
    for child in rdir.iterdir():
        if child.name in preserved_names:
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except Exception as exc:
            raise OpalError(f"Failed to clear round directory {rdir}: {exc}") from exc


def _round_dir_has_blocking_entries(rdir: Path) -> bool:
    entries = list(rdir.iterdir())
    if not entries:
        return False
    if len(entries) != 1 or entries[0].name != "logs" or not entries[0].is_dir():
        return True
    children = list(entries[0].iterdir())
    if len(children) != 1 or children[0].name != "round.log.jsonl":
        return True
    allowed = {
        "command_start",
        "x_validate_start",
        "x_validate_done",
        "x_memory_guard_done",
        "records_load_start",
        "records_load_done",
        "lock_acquire_start",
        "lock_acquired",
        "lock_released",
        "abort",
    }
    try:
        stages = {
            str(json.loads(line).get("stage"))
            for line in children[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except Exception:
        return True
    return not stages or not stages.issubset(allowed)


def assert_round_artifacts_writable(
    rdir: Path,
    *,
    round_index: int,
    allow_resume: bool,
) -> None:
    """Reject writes to an existing round unless an explicit resume permits them."""
    if not allow_resume and rdir.exists() and _round_dir_has_blocking_entries(rdir):
        raise OpalError(f"Round {int(round_index)} already contains artifacts in {rdir}. Use --resume to overwrite.")


def _validate_allocated_batch_k_override(req: RunRoundRequest) -> None:
    cfg = req.cfg
    if cfg.selection_batch.allocation is None or req.k_override is None:
        return
    quota_total = int(req.k_override) * len(cfg.selection_views)
    expected = cfg.selection_batch.expected_unique_count
    if expected is None or int(expected) != quota_total:
        raise OpalError(
            "The CLI top-k override is incompatible with the configured selection_batch allocation: "
            f"expected_unique_count={expected}, override_quota_sum={quota_total}. "
            "Update the campaign contract instead of changing an allocated batch implicitly."
        )


def run_round(store: RecordsStore, df: pd.DataFrame, req: RunRoundRequest) -> RunRoundResult:
    cfg = req.cfg
    _validate_allocated_batch_k_override(req)
    cfg_path = req.config_path or (Path(cfg.campaign.workdir) / "campaign.yaml")
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    if not ws.state_path.exists():
        raise OpalError(f"state.json not found at {ws.state_path}. Run `opal init -c {ws.config_path}` first.")

    rdir = ws.round_dir(req.as_of_round)
    store.assert_unique_ids(df)

    assert_round_artifacts_writable(
        rdir,
        round_index=int(req.as_of_round),
        allow_resume=bool(req.allow_resume),
    )
    if req.allow_resume:
        _clear_round_dir(rdir, preserve_logs=True, preserve_run_artifacts=True)
    round_log_path = rdir / "logs" / "round.log.jsonl"

    inputs = RoundInputs(cfg=cfg, req=req, ws=ws, store=store, df=df, rdir=rdir)

    try:
        yops_names = [p.name for p in (cfg.training.y_ops or [])]
    except Exception:
        yops_names = []
    _log(
        req.verbose,
        "[plugins] x=%s | y_ingest=%s | model=%s | selection_views=%s | y_ops=%s"
        % (
            cfg.data.transforms_x.name,
            cfg.data.transforms_y.name,
            cfg.model.name,
            [
                {"id": view.id, "objective": view.objective.name, "selection": view.selection.name}
                for view in cfg.selection_views
            ],
            (yops_names or "(none)"),
        ),
    )

    ensure_dir(rdir)
    append_round_log_event(
        round_log_path,
        {
            "ts": now_iso(),
            "stage": "start",
            "round": int(req.as_of_round),
            "campaign": {"slug": cfg.campaign.slug, "workdir": cfg.campaign.workdir},
            "data": {
                "x_column": cfg.data.x_column_name,
                "y_column": cfg.data.y_column_name,
                "label_source": getattr(cfg.labels.source, "kind", "campaign_history"),
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
                "selection_views": [
                    {
                        "id": view.id,
                        "objective": {"name": view.objective.name, "params": view.objective.params},
                        "selection": {"name": view.selection.name, "params": view.selection.params},
                    }
                    for view in cfg.selection_views
                ],
                "y_ops": [{"name": p.name, "params": p.params} for p in (cfg.training.y_ops or [])],
            },
        },
    )
    t0 = time.perf_counter()
    append_round_log_event(
        round_log_path,
        {"ts": now_iso(), "round": int(req.as_of_round), "stage": "training_start"},
    )
    training = stage_training(inputs)
    append_round_log_event(
        round_log_path,
        {
            "ts": now_iso(),
            "round": int(req.as_of_round),
            "stage": "training_done",
            "n_train": len(training.train_ids),
            "y_dim": int(training.y_dim),
        },
    )

    run_id, _, rctx = build_round_ctx(
        cfg=cfg,
        as_of_round=int(req.as_of_round),
        y_dim=training.y_dim,
        n_train=len(training.train_ids),
    )
    LedgerWriter(ws).require_run_id_available(run_id)
    reserve_run_artifact_directory(rdir, run_id=run_id)
    append_round_log_event(
        round_log_path,
        {"ts": now_iso(), "round": int(req.as_of_round), "run_id": run_id, "stage": "run_context"},
    )

    tx = get_transform_x(cfg.data.transforms_x.name, cfg.data.transforms_x.params)
    tctx = rctx.for_plugin(category="transform_x", name=cfg.data.transforms_x.name, plugin=tx)

    append_round_log_event(
        round_log_path,
        {"ts": now_iso(), "round": int(req.as_of_round), "run_id": run_id, "stage": "x_matrices_start"},
    )
    xbundle = stage_x_matrices(
        inputs=inputs,
        plan=training.plan,
        train_ids=training.train_ids,
        Y_train=training.Y_train,
        tctx=tctx,
        rctx=rctx,
    )
    append_round_log_event(
        round_log_path,
        {
            "ts": now_iso(),
            "round": int(req.as_of_round),
            "run_id": run_id,
            "stage": "x_matrices_done",
            "n_train": len(xbundle.id_order_train),
            "n_pool": len(xbundle.id_order_pool),
            "x_dim": int(xbundle.X_train.shape[1]),
            "pool_mode": "streaming",
        },
    )

    append_round_log_event(
        round_log_path,
        {
            "ts": now_iso(),
            "round": int(req.as_of_round),
            "run_id": run_id,
            "stage": "scoring_start",
            "n_pool": len(xbundle.id_order_pool),
        },
    )
    score = stage_scoring(
        inputs=inputs,
        rctx=rctx,
        X_train=xbundle.X_train,
        Y_train=training.Y_train,
        R_train=training.R_train,
        tctx=tctx,
        id_order_train=xbundle.id_order_train,
        id_order_pool=xbundle.id_order_pool,
        candidate_df=xbundle.cand_df,
        y_dim=training.y_dim,
    )

    append_round_log_event(
        round_log_path,
        {"ts": now_iso(), "round": int(req.as_of_round), "run_id": run_id, "stage": "artifacts_start"},
    )
    artifacts = write_round_artifacts(
        inputs=inputs,
        run_id=run_id,
        rctx=rctx,
        training=training,
        xbundle=xbundle,
        score=score,
    )
    append_round_log_event(
        round_log_path,
        {"ts": now_iso(), "round": int(req.as_of_round), "run_id": run_id, "stage": "artifacts_done"},
    )

    run_events = build_run_events(
        inputs=inputs,
        run_id=run_id,
        training=training,
        xbundle=xbundle,
        score=score,
        artifacts=artifacts,
    )

    append_round_log_event(
        round_log_path,
        {"ts": now_iso(), "round": int(req.as_of_round), "run_id": run_id, "stage": "ledger_append_start"},
    )
    append_ledgers(
        ws=ws,
        run_pred_events=run_events.run_pred_events,
        run_meta_event=run_events.run_meta_event,
        verbose=req.verbose,
    )
    append_round_log_event(
        round_log_path,
        {"ts": now_iso(), "round": int(req.as_of_round), "run_id": run_id, "stage": "ledger_append_done"},
    )

    records_label_hist_updated = False
    _log(req.verbose, "[writeback] candidate predictions persisted to the shared round ledger.")

    total_duration = time.perf_counter() - t0
    append_round_log_event(
        round_log_path,
        {"ts": now_iso(), "round": int(req.as_of_round), "run_id": run_id, "stage": "state_update_start"},
    )
    update_campaign_state(
        ws=ws,
        cfg=cfg,
        req=req,
        rep=training.rep,
        train_df=training.train_df,
        observed_events_df=training.observed_events_df,
        id_order_train=xbundle.id_order_train,
        id_order_pool=xbundle.id_order_pool,
        selections=score.selections,
        selection_batch=score.selection_batch,
        apaths=artifacts.apaths,
        run_id=run_id,
        store=store,
        total_duration=total_duration,
        fit_duration=score.fit_duration,
        records_label_hist_updated=records_label_hist_updated,
    )
    append_round_log_event(
        round_log_path,
        {"ts": now_iso(), "round": int(req.as_of_round), "run_id": run_id, "stage": "state_update_done"},
    )
    append_round_log_event(
        round_log_path,
        {"ts": now_iso(), "round": int(req.as_of_round), "run_id": run_id, "stage": "retention_start"},
    )
    retention_manifest = apply_runtime_artifact_retention(cfg, ws)
    append_round_log_event(
        round_log_path,
        {
            "ts": now_iso(),
            "round": int(req.as_of_round),
            "run_id": run_id,
            "stage": "retention_done",
            "status": retention_manifest.get("status"),
        },
    )

    append_round_log_event(
        round_log_path,
        {
            "ts": now_iso(),
            "round": int(req.as_of_round),
            "run_id": run_id,
            "stage": "fit",
            "oob_r2": getattr(score.fit_metrics, "oob_r2", None),
        },
    )
    append_round_log_event(
        round_log_path,
        {
            "ts": now_iso(),
            "round": int(req.as_of_round),
            "run_id": run_id,
            "stage": "selection",
            "selection_views": {
                view_id: {
                    "top_k": int(selection.top_k),
                    "effective": int(selection.selected_effective),
                }
                for view_id, selection in score.selections.items()
            },
            "selection_batch_count": int(score.selection_batch.unique_count),
        },
    )
    append_round_log_event(
        round_log_path,
        {"ts": now_iso(), "round": int(req.as_of_round), "run_id": run_id, "stage": "done"},
    )

    return RunRoundResult(
        ok=True,
        run_id=run_id,
        as_of_round=int(req.as_of_round),
        trained_on=len(xbundle.id_order_train),
        scored=len(xbundle.id_order_pool),
        selection_views={
            view_id: {
                "top_k_requested": int(selection.top_k),
                "top_k_effective": int(selection.selected_effective),
            }
            for view_id, selection in score.selections.items()
        },
        selection_batch_count=int(score.selection_batch.unique_count),
        ledger_path=str(ws.ledger_dir.resolve()),
    )
