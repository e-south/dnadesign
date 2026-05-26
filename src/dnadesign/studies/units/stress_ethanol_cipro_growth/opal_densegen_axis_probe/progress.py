"""Progress summaries for DenseGen axis OPAL probe run roots."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from dnadesign.opal import build_campaign_progress, render_campaign_progress_text

from .artifacts import ProbeArtifactLayout


def summarize_probe_progress(run_root: Path, *, include_opal_progress: bool = False) -> dict[str, Any]:
    layout = ProbeArtifactLayout(Path(run_root).resolve())
    if not layout.run_root.exists():
        raise RuntimeError(f"run root not found: {layout.run_root}")
    expected_round_count = _planned_round_count(layout)
    campaigns = [
        _campaign_progress(path, include_opal_progress=include_opal_progress, expected_round_count=expected_round_count)
        for path in _campaign_config_paths(layout)
    ]
    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.progress.v1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_root": str(layout.run_root),
        "detail": "full" if include_opal_progress else "compact",
        "expected_round_count": expected_round_count,
        "campaign_count": len(campaigns),
        "status": "no_campaigns" if not campaigns else _aggregate_status(campaigns),
        "campaigns": campaigns,
    }


def format_probe_progress_text(payload: dict[str, Any]) -> str:
    lines = [
        "opal_densegen_axis_probe_v0 progress",
        f"run_root: {payload.get('run_root')}",
        f"status: {payload.get('status')}",
        f"campaign_count: {payload.get('campaign_count')}",
    ]
    for campaign in payload.get("campaigns") or []:
        predict = campaign.get("predict") or {}
        predict_text = ""
        if predict.get("batch") is not None:
            predict_text = f" predict={predict.get('batch')}/{predict.get('of')}"
        lines.append(
            "{run_key} round={round_index} status={status} last_stage={last_stage}{predict} "
            "elapsed_sec={elapsed}".format(
                run_key=campaign.get("run_key"),
                round_index=campaign.get("round_index"),
                status=campaign.get("status"),
                last_stage=campaign.get("last_stage"),
                predict=predict_text,
                elapsed=campaign.get("elapsed_sec"),
            )
        )
    return "\n".join(lines)


def _campaign_config_paths(layout: ProbeArtifactLayout) -> list[Path]:
    if not layout.scratch_campaigns_dir.exists():
        return []
    return sorted(layout.scratch_campaigns_dir.glob("*/configs/campaign.yaml"))


def _campaign_progress(
    config_path: Path,
    *,
    include_opal_progress: bool,
    expected_round_count: int | None,
) -> dict[str, Any]:
    opal_progress = build_campaign_progress(config_path, round_selector="all")
    rounds = list(opal_progress.get("rounds") or [])
    latest = rounds[-1] if rounds else {}
    round_count = int(opal_progress.get("round_count") or 0)
    payload = {
        "run_key": config_path.parents[1].name,
        "config_path": str(config_path),
        "status": _scoped_campaign_status(
            status=str(opal_progress.get("status") or ""),
            round_count=round_count,
            expected_round_count=expected_round_count,
        ),
        "round_count": round_count,
        "expected_round_count": expected_round_count,
        "round_index": latest.get("round_index"),
        "last_stage": latest.get("last_stage"),
        "elapsed_sec": latest.get("elapsed_sec"),
        "events": latest.get("events"),
        "predict": latest.get("predict") or {"batch": None, "of": None, "rows": None},
        "summary": latest.get("summary") or {"events": 0},
        "path": latest.get("path"),
    }
    if include_opal_progress:
        payload["opal_progress"] = opal_progress
    return payload


def _aggregate_status(campaigns: list[dict[str, Any]]) -> str:
    if all(campaign.get("status") == "done" for campaign in campaigns):
        return "done"
    return "running_or_incomplete"


def _planned_round_count(layout: ProbeArtifactLayout) -> int | None:
    if not layout.probe_plan_path.exists():
        return None
    try:
        payload = json.loads(layout.probe_plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    plan = payload.get("plan")
    if not isinstance(plan, Mapping):
        return None
    rounds = plan.get("rounds")
    try:
        value = int(rounds)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _scoped_campaign_status(*, status: str, round_count: int, expected_round_count: int | None) -> str:
    if status == "not_started" or expected_round_count is None:
        return status
    if status == "done" and round_count >= expected_round_count:
        return "done"
    return "running_or_incomplete"


def format_opal_campaign_progress_text(payload: dict[str, Any]) -> str:
    return render_campaign_progress_text(payload)
