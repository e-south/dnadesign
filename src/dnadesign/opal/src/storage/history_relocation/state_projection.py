"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/history_relocation/state_projection.py

Projects verified run artifacts into one canonical OPAL campaign state.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ...config.plugin_schemas import validate_params
from ...config.types import RootConfig
from ...core.utils import OpalError, file_sha256
from ..state import BACKLOG_COUNT_KEY, CampaignState, RoundEntry
from .contracts import HistoryRelocationPlan, RunHistory
from .inspection import canonical_sha256, jsonable, run_artifact_root


def _definitions(value: Any, *, field: str) -> list[dict[str, Any]]:
    payload = json.loads(value) if isinstance(value, str) else jsonable(value)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise OpalError(f"Run metadata field {field} must contain a list of objects.")
    return payload


def _validated_plugin_params(category: str, name: str, value: Any) -> dict[str, Any]:
    return validate_params(category, name, dict(value or {}))


def require_target_config_matches_run_history(plan: HistoryRelocationPlan, cfg: RootConfig) -> None:
    run = min((*plan.source.runs, *plan.target.runs), key=lambda item: item.round_index)
    run_objectives = [
        {
            "selection_view_id": str(item.get("selection_view_id") or ""),
            "objective_name": str(item.get("objective_name") or ""),
            "params": _validated_plugin_params(
                "objective",
                str(item.get("objective_name") or ""),
                item.get("params"),
            ),
        }
        for item in _definitions(run.run_row["objective__defs_json"], field="objective__defs_json")
    ]
    run_x_name = str(run.run_row.get("x_transform__name") or "")
    run_y_name = str(run.run_row.get("y_ingest__name") or "")
    run_contract = {
        "columns": {
            "x": str(run.run_row.get("data__x_column_name") or ""),
            "y": str(run.run_row.get("data__y_column_name") or ""),
        },
        "x_transform": {
            "name": run_x_name,
            "params": _validated_plugin_params(
                "transform_x",
                run_x_name,
                run.run_row.get("x_transform__params"),
            ),
        },
        "y_ingest": {
            "name": run_y_name,
            "params": _validated_plugin_params(
                "transform_y",
                run_y_name,
                run.run_row.get("y_ingest__params"),
            ),
        },
        "objectives": run_objectives,
    }
    config_contract = {
        "columns": {
            "x": cfg.data.x_column_name,
            "y": cfg.data.y_column_name,
        },
        "x_transform": {
            "name": cfg.data.transforms_x.name,
            "params": dict(cfg.data.transforms_x.params),
        },
        "y_ingest": {
            "name": cfg.data.transforms_y.name,
            "params": dict(cfg.data.transforms_y.params),
        },
        "objectives": [
            {
                "selection_view_id": view.id,
                "objective_name": view.objective.name,
                "params": jsonable(view.objective.params),
            }
            for view in cfg.selection_views
        ],
    }
    if canonical_sha256(config_contract) != canonical_sha256(run_contract):
        raise OpalError("Target campaign config differs from the verified run history X/Y/objective contract.")


def _state_entry(plan: HistoryRelocationPlan, *, round_index: int) -> RoundEntry | None:
    for history in (plan.source, plan.target):
        if history.state is None:
            continue
        for entry in history.state.rounds:
            if int(entry.round_index) == int(round_index):
                return entry
    return None


def _round_entry(plan: HistoryRelocationPlan, run: RunHistory) -> RoundEntry:
    row = run.run_row
    context = run.round_context
    target_round_dir = plan.target.workdir / "outputs" / "rounds" / f"round_{run.round_index}"
    source_artifact_root = run_artifact_root(run.round_dir, run_id=run.run_id)
    target_artifact_root = target_round_dir / "run_artifacts" / source_artifact_root.name
    labels_used = pd.read_parquet(source_artifact_root / "labels" / "labels_used.parquet")
    labels_used_rounds = sorted({int(value) for value in labels_used["observed_round"].tolist()})
    selections = pd.read_parquet(source_artifact_root / "selection" / "selections.parquet")
    batch = pd.read_parquet(source_artifact_root / "selection" / "selection_batch.parquet")
    objective_defs = {
        str(item["selection_view_id"]): item
        for item in _definitions(row["objective__defs_json"], field="objective__defs_json")
    }
    selection_defs = _definitions(row["selection_views__defs_json"], field="selection_views__defs_json")
    selection_views: dict[str, dict[str, Any]] = {}
    for definition in selection_defs:
        view_id = str(definition["selection_view_id"])
        observed = selections.loc[selections["selection_view_id"].astype(str) == view_id]
        selection_views[view_id] = {
            "objective_name": str(objective_defs[view_id]["objective_name"]),
            "selection_name": str(definition["selection_name"]),
            "score_ref": str(definition["score_ref"]),
            "top_k_requested": int(definition["top_k"]),
            "top_k_effective_after_ties": int(len(observed)),
        }
    allocation = jsonable(context.get("core/selection_batch/allocation") or {})
    deduplicate_values = {str(value) for value in batch["deduplicate_by"].tolist()}
    if len(deduplicate_values) != 1:
        raise OpalError(f"Round {run.round_index} selection batch has multiple deduplication keys.")
    model_path = target_round_dir / "model" / "model.joblib"
    source_model_path = source_artifact_root / "model" / "model.joblib"
    prior_entry = _state_entry(plan, round_index=run.round_index)
    artifact_paths = {
        "selections_parquet": str((target_round_dir / "selection" / "selections.parquet").resolve()),
        "selection_batch_parquet": str((target_round_dir / "selection" / "selection_batch.parquet").resolve()),
        "ledger_predictions_dir": str((plan.target.workdir / "outputs" / "ledger" / "predictions").resolve()),
        "ledger_runs_parquet": str((plan.target.workdir / "outputs" / "ledger" / "runs.parquet").resolve()),
        "ledger_labels_parquet": str((plan.target.workdir / "outputs" / "ledger" / "labels.parquet").resolve()),
        "round_ctx_json": str((target_round_dir / "metadata" / "round_ctx.json").resolve()),
        "objective_meta_json": str((target_round_dir / "metadata" / "objective_meta.json").resolve()),
        "model_meta_json": str((target_round_dir / "model" / "model_meta.json").resolve()),
        "labels_used_parquet": str((target_artifact_root / "labels" / "labels_used.parquet").resolve()),
        "observed_events_parquet": str((target_artifact_root / "labels" / "observed_events.parquet").resolve()),
        "round_log_jsonl": str((target_round_dir / "logs" / "round.log.jsonl").resolve()),
    }
    allocation_trace = target_round_dir / "selection" / "allocation_trace.parquet"
    if (source_artifact_root / "selection" / "allocation_trace.parquet").is_file():
        artifact_paths["selection_allocation_trace_parquet"] = str(allocation_trace.resolve())
    return RoundEntry(
        round_index=run.round_index,
        run_id=run.run_id,
        round_name=f"round_{run.round_index}",
        round_dir=str(target_round_dir.resolve()),
        labels_used_rounds=labels_used_rounds,
        number_of_training_examples_used_in_round=int(row["stats__n_train"]),
        number_of_candidates_scored_in_round=int(row["stats__n_scored"]),
        selection_views=selection_views,
        selection_batch={
            "deduplicate_by": next(iter(deduplicate_values)),
            "unique_count": int(len(batch)),
            "expected_unique_count": allocation.get("expected_unique_count"),
            "allocation": allocation,
        },
        model={
            "type": str(row["model__name"]),
            "params": jsonable(row["model__params"]),
            "artifact_path": str(model_path.resolve()),
            "artifact_sha256": file_sha256(source_model_path),
        },
        metrics=dict(prior_entry.metrics) if prior_entry is not None else {},
        durations_sec=dict(prior_entry.durations_sec) if prior_entry is not None else {},
        seeds={
            "global": jsonable(row["model__params"]).get("random_state")
            if isinstance(jsonable(row["model__params"]), dict)
            else None,
            "model": jsonable(row["model__params"]).get("random_state")
            if isinstance(jsonable(row["model__params"]), dict)
            else None,
        },
        artifacts=artifact_paths,
        writebacks={"prediction_records": "ledger_only", "records_label_hist_updated": []},
        warnings=list(prior_entry.warnings) if prior_entry is not None else [],
        status="completed",
    )


def _history_timestamps(plan: HistoryRelocationPlan) -> tuple[str, str]:
    created = plan.source.state.created_at if plan.source.state is not None else "1970-01-01T00:00:00+00:00"
    timestamps: list[str] = [created]
    for history in (plan.source, plan.target):
        for run in history.runs:
            log_path = run.round_dir / "logs" / "round.log.jsonl"
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("ts"):
                    timestamps.append(str(payload["ts"]))
    return created, max(timestamps)


def _pending_selection_count(plan: HistoryRelocationPlan) -> int:
    selected_ids: set[str] = set()
    all_runs = sorted((*plan.source.runs, *plan.target.runs), key=lambda item: item.round_index)
    for run in all_runs:
        artifact_root = run_artifact_root(run.round_dir, run_id=run.run_id)
        batch = pd.read_parquet(artifact_root / "selection" / "selection_batch.parquet", columns=["id"])
        selected_ids.update(batch["id"].astype(str).tolist())
    latest = all_runs[-1]
    artifact_root = run_artifact_root(latest.round_dir, run_id=latest.run_id)
    observed = pd.read_parquet(artifact_root / "labels" / "observed_events.parquet", columns=["id"])
    return len(selected_ids - set(observed["id"].astype(str).tolist()))


def build_canonical_state(
    plan: HistoryRelocationPlan,
    *,
    cfg: RootConfig,
    records_path: Path,
) -> CampaignState:
    require_target_config_matches_run_history(plan, cfg)
    created_at, updated_at = _history_timestamps(plan)
    location = cfg.data.location
    if hasattr(location, "dataset"):
        data_location = {
            "kind": "usr",
            "dataset": str(location.dataset),
            "path": str(Path(location.path).resolve()),
            "records_path": str(records_path.resolve()),
        }
    else:
        data_location = {
            "kind": "local",
            "path": str(Path(location.path).resolve()),
            "records_path": str(records_path.resolve()),
        }
    all_runs = sorted((*plan.source.runs, *plan.target.runs), key=lambda item: item.round_index)
    dimensions = {int(run.round_context["core/data/x_dim"]) for run in all_runs}
    if len(dimensions) != 1:
        raise OpalError(f"Campaign rounds use multiple representation dimensions: {sorted(dimensions)}.")
    state = CampaignState(
        campaign_slug=cfg.campaign.slug,
        campaign_name=cfg.campaign.name,
        workdir=str(plan.target.workdir.resolve()),
        data_location=data_location,
        x_column_name=cfg.data.x_column_name,
        y_column_name=cfg.data.y_column_name,
        created_at=created_at,
        updated_at=updated_at,
        representation_vector_dimension=next(iter(dimensions)),
        representation_transform={"name": cfg.data.transforms_x.name, "params": cfg.data.transforms_x.params},
        training_policy=dict(cfg.training.policy or {}),
        performance={
            "score_batch_size": cfg.scoring.score_batch_size,
            "selection_view_ids": [view.id for view in cfg.selection_views],
        },
        rounds=[_round_entry(plan, run) for run in all_runs],
        backlog={BACKLOG_COUNT_KEY: _pending_selection_count(plan)},
    )
    return state
