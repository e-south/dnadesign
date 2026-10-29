"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/reporting/progress.py

Builds machine-readable OPAL campaign progress summaries from campaign state and.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..analysis.campaign import CampaignAnalysis
from ..core.utils import OpalError
from ..storage.locks import inspect_campaign_lock
from ..storage.state import CampaignState
from ..storage.workspace import CampaignWorkspace
from .artifact_garden import build_artifact_garden_audit
from .summary import load_round_log, summarize_round_log

PROGRESS_SCHEMA_VERSION = "opal.campaign_progress.v1"


def build_campaign_progress(
    config_path: Path | None,
    *,
    round_selector: str | None = "latest",
    run_id: str | None = None,
    usr_root: str | Path | None = None,
) -> dict[str, Any]:
    analysis = CampaignAnalysis.from_config_path(config_path, allow_dir=True, usr_root=usr_root)
    cfg = analysis.config
    ws = analysis.workspace
    round_indices = _resolve_progress_rounds(ws, round_selector)
    rounds = [_round_progress(ws, round_index, run_id=run_id) for round_index in round_indices]
    event_contract = _event_contract_summary(rounds)
    lock_state = inspect_campaign_lock(ws.workdir)
    warnings = []
    if lock_state.get("active"):
        warnings.append(
            {
                "category": "ActiveLockWarning",
                "severity": "warning",
                "message": "Campaign lock is active on this host.",
                "path": lock_state.get("lockfile"),
            }
        )
    if lock_state.get("stale") or lock_state.get("unreadable"):
        warnings.append(
            {
                "category": "StaleLockWarning",
                "severity": "warning",
                "message": "Campaign lock is stale or unreadable on this host.",
                "path": lock_state.get("lockfile"),
            }
        )
    warnings.extend(_event_contract_warnings(rounds))
    artifact_garden, stale_artifacts, artifact_warnings = _artifact_garden_progress(
        analysis.config_path,
        usr_root=usr_root,
    )
    warnings.extend(artifact_warnings)
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "campaign": {
            "name": cfg.campaign.name,
            "slug": cfg.campaign.slug,
            "workdir": str(ws.workdir),
            "config_path": str(analysis.config_path),
            "x_column": cfg.data.x_column_name,
            "y_column": cfg.data.y_column_name,
        },
        "state": {
            "exists": ws.state_path.exists(),
            "path": str(ws.state_path),
        },
        "round_selector": round_selector or "latest",
        "run_id": run_id,
        "status": _campaign_status(rounds),
        "round_count": len(rounds),
        "warnings": warnings,
        "event_contract": event_contract,
        "locks": {"campaign": lock_state},
        "artifact_garden": artifact_garden,
        "stale_artifacts": stale_artifacts,
        "rounds": rounds,
    }


def render_campaign_progress_text(payload: dict[str, Any]) -> str:
    campaign = payload.get("campaign") or {}
    lines = [
        "OPAL campaign progress",
        f"campaign: {campaign.get('slug')}",
        f"workdir: {campaign.get('workdir')}",
        f"status: {payload.get('status')}",
        f"round_count: {payload.get('round_count')}",
    ]
    for row in payload.get("rounds") or []:
        predict = row.get("predict") or {}
        predict_text = ""
        if predict.get("batch") is not None:
            predict_text = f" predict={predict.get('batch')}/{predict.get('of')}"
        lines.append(
            "round={round_index} status={status} last_stage={last_stage}{predict} elapsed_sec={elapsed}".format(
                round_index=row.get("round_index"),
                status=row.get("status"),
                last_stage=row.get("last_stage"),
                predict=predict_text,
                elapsed=row.get("elapsed_sec"),
            )
        )
    return "\n".join(lines)


def _resolve_progress_rounds(ws: CampaignWorkspace, round_selector: str | None) -> list[int]:
    available = _available_round_indices(ws)
    selector = (round_selector or "latest").strip().lower()
    if selector == "all":
        return available
    if selector == "latest":
        return [available[-1]] if available else []
    try:
        requested = int(selector)
    except ValueError as exc:
        raise OpalError("--round must be an integer, latest, or all.") from exc
    if requested not in set(available):
        raise OpalError(f"--round {requested} not found. Available rounds: {available}")
    return [requested]


def _available_round_indices(ws: CampaignWorkspace) -> list[int]:
    rounds: set[int] = set()
    if ws.rounds_dir.exists():
        for path in ws.rounds_dir.glob("round_*"):
            round_index = _parse_round_dir_name(path.name)
            if round_index is not None:
                rounds.add(round_index)
    if ws.state_path.exists():
        try:
            state = CampaignState.load(ws.state_path)
        except Exception as exc:
            raise OpalError(f"Failed to load state.json at {ws.state_path}: {exc}") from exc
        rounds.update(int(row.round_index) for row in state.rounds)
    return sorted(rounds)


def _round_progress(ws: CampaignWorkspace, round_index: int, *, run_id: str | None = None) -> dict[str, Any]:
    log_path = ws.round_logs_dir(round_index) / "round.log.jsonl"
    if not log_path.exists():
        return {
            "round_index": int(round_index),
            "status": "no_log",
            "last_stage": None,
            "elapsed_sec": None,
            "events": 0,
            "predict": {"batch": None, "of": None, "rows": None},
            "summary": {"events": 0},
            "path": str(log_path),
        }
    events = load_round_log(log_path)
    summary = summarize_round_log(events, run_id=run_id)
    last_event = events[-1] if events else {}
    last_predict = next((event for event in reversed(events) if event.get("stage") == "predict_batch"), {})
    status = "done" if summary.get("done_ts") else "running_or_incomplete"
    if summary.get("aborted"):
        status = "aborted"
    if not events:
        status = "empty_log"
    elapsed_sec = summary.get("duration_sec_total")
    start_ts = _parse_ts(summary.get("start_ts"))
    if elapsed_sec is None and start_ts is not None:
        elapsed_sec = round((datetime.now(UTC) - start_ts).total_seconds(), 3)
    return {
        "round_index": int(round_index),
        "status": status,
        "last_stage": last_event.get("stage"),
        "elapsed_sec": elapsed_sec,
        "events": summary.get("events"),
        "predict": _predict_progress(last_predict),
        "summary": summary,
        "path": str(log_path.resolve()),
    }


def _parse_round_dir_name(name: str) -> int | None:
    if not name.startswith("round_"):
        return None
    try:
        return int(name.removeprefix("round_"))
    except ValueError:
        return None


def _predict_progress(event: dict[str, Any]) -> dict[str, Any]:
    batch = _int_or_none(event.get("batch"))
    total = _int_or_none(event.get("of"))
    if batch is not None and total is not None and total < batch:
        total = batch
    return {
        "batch": batch,
        "of": total,
        "rows": _int_or_none(event.get("rows")),
    }


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _event_contract_summary(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [row.get("summary") or {} for row in rounds]
    run_scopes = [summary.get("run_scope") or {} for summary in summaries]
    attempt_ids = sorted(
        {
            str(attempt_id)
            for scope in run_scopes
            for attempt_id in (scope.get("attempt_ids") or [])
            if attempt_id not in (None, "")
        }
    )
    return {
        "schema_version": "opal.progress_event_rollup.v1",
        "command_events": int(sum(int(summary.get("command_events") or 0) for summary in summaries)),
        "preflight_events": int(sum(int(summary.get("preflight_events") or 0) for summary in summaries)),
        "run_events": int(sum(int(summary.get("run_events") or 0) for summary in summaries)),
        "abort_events": int(sum(int(summary.get("abort_events") or 0) for summary in summaries)),
        "finalize_events": int(sum(int(summary.get("finalize_events") or 0) for summary in summaries)),
        "attempt_ids": attempt_ids,
        "aborted_rounds": [
            int(row["round_index"])
            for row in rounds
            if (row.get("summary") or {}).get("aborted") and row.get("round_index") is not None
        ],
        "ambiguous_rounds": [
            int(row["round_index"])
            for row in rounds
            if ((row.get("summary") or {}).get("run_scope") or {}).get("ambiguous_run_scope")
            and row.get("round_index") is not None
        ],
    }


def _event_contract_warnings(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for row in rounds:
        summary = row.get("summary") or {}
        round_index = row.get("round_index")
        if summary.get("aborted"):
            warnings.append(
                {
                    "category": "ProgressContractError",
                    "severity": "warning",
                    "message": f"Round {round_index} has an abort event in its round log.",
                    "round_index": round_index,
                }
            )
        run_scope = summary.get("run_scope") or {}
        if run_scope.get("ambiguous_run_scope"):
            warnings.append(
                {
                    "category": "RunScopeAmbiguityError",
                    "severity": "warning",
                    "message": f"Round {round_index} has multiple run_id values; pass --run-id on run-scoped surfaces.",
                    "round_index": round_index,
                    "run_ids": run_scope.get("run_ids") or [],
                }
            )
    return warnings


def _artifact_garden_progress(
    config_path: Path,
    *,
    usr_root: str | Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        audit = build_artifact_garden_audit(config_path, usr_root=usr_root)
    except Exception as exc:
        return (
            {
                "schema_version": "opal.artifact_garden.unavailable",
                "status": "unavailable",
                "error": str(exc),
            },
            [],
            [
                {
                    "category": "ArtifactGardenWarning",
                    "severity": "warning",
                    "message": f"Artifact garden audit unavailable for progress: {exc}",
                }
            ],
        )

    stale_artifacts = list(audit.get("stale_artifacts") or [])
    summary = {
        "schema_version": audit.get("schema_version"),
        "status": "ok",
        "local_only": audit.get("local_only"),
        "active_manifest_count": len(audit.get("active_manifests") or []),
        "stale_artifact_count": len(stale_artifacts),
        "bytes": audit.get("bytes") or {},
        "prune_plan": {
            "item_count": ((audit.get("prune_plan") or {}).get("item_count") or len(stale_artifacts)),
            "bytes_to_delete": ((audit.get("prune_plan") or {}).get("bytes_to_delete") or 0),
            "requires_apply": bool((audit.get("prune_plan") or {}).get("requires_apply", True)),
        },
    }
    warnings = list(audit.get("warnings") or [])
    warnings.extend(_stale_artifact_progress_warnings(stale_artifacts))
    return summary, stale_artifacts, warnings


def _stale_artifact_progress_warnings(stale_artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for row in stale_artifacts:
        path = row.get("path")
        warnings.append(
            {
                "category": "StaleArtifactWarning",
                "severity": "warning",
                "message": f"Generated artifact exists on disk but is absent from the active manifest set: {path}",
                "path": path,
                "scope": row.get("scope"),
            }
        )
    return warnings


def _campaign_status(rounds: list[dict[str, Any]]) -> str:
    if not rounds:
        return "not_started"
    if any(row.get("status") == "aborted" for row in rounds):
        return "aborted"
    if all(row.get("status") == "done" for row in rounds):
        return "done"
    if any(row.get("status") == "running_or_incomplete" for row in rounds):
        return "running_or_incomplete"
    if any(row.get("status") in {"no_log", "empty_log"} for row in rounds):
        return "attention"
    return "unknown"
