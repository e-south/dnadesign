"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/reporting/progress.py

Builds machine-readable OPAL campaign progress summaries from campaign state and
round logs. This module is campaign-generic; study probes should adapt this
surface instead of parsing OPAL scratch directories directly.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..analysis.facade import CampaignAnalysis
from ..core.utils import OpalError
from ..storage.state import CampaignState
from ..storage.workspace import CampaignWorkspace
from .summary import load_round_log, summarize_round_log

PROGRESS_SCHEMA_VERSION = "opal.campaign_progress.v1"


def build_campaign_progress(
    config_path: Path | None,
    *,
    round_selector: str | None = "latest",
) -> dict[str, Any]:
    analysis = CampaignAnalysis.from_config_path(config_path, allow_dir=True)
    cfg = analysis.config
    ws = analysis.workspace
    round_indices = _resolve_progress_rounds(ws, round_selector)
    rounds = [_round_progress(ws, round_index) for round_index in round_indices]
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
        "status": _campaign_status(rounds),
        "round_count": len(rounds),
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


def _round_progress(ws: CampaignWorkspace, round_index: int) -> dict[str, Any]:
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
    summary = summarize_round_log(events)
    last_event = events[-1] if events else {}
    last_predict = next((event for event in reversed(events) if event.get("stage") == "predict_batch"), {})
    status = "done" if summary.get("done_ts") else "running_or_incomplete"
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
        "predict": {
            "batch": last_predict.get("batch"),
            "of": last_predict.get("of"),
            "rows": last_predict.get("rows"),
        },
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


def _campaign_status(rounds: list[dict[str, Any]]) -> str:
    if not rounds:
        return "not_started"
    if all(row.get("status") == "done" for row in rounds):
        return "done"
    if any(row.get("status") == "running_or_incomplete" for row in rounds):
        return "running_or_incomplete"
    if any(row.get("status") in {"no_log", "empty_log"} for row in rounds):
        return "attention"
    return "unknown"
