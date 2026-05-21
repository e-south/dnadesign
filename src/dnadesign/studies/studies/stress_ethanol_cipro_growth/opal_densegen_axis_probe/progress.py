"""Progress summaries for DenseGen axis OPAL probe run roots."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dnadesign.opal import build_campaign_progress, render_campaign_progress_text

from .artifacts import ProbeArtifactLayout


def summarize_probe_progress(run_root: Path, *, include_opal_progress: bool = False) -> dict[str, Any]:
    layout = ProbeArtifactLayout(Path(run_root).resolve())
    if not layout.run_root.exists():
        raise RuntimeError(f"run root not found: {layout.run_root}")
    campaigns = [
        _campaign_progress(path, include_opal_progress=include_opal_progress) for path in _campaign_config_paths(layout)
    ]
    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.progress.v1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_root": str(layout.run_root),
        "detail": "full" if include_opal_progress else "compact",
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


def _campaign_progress(config_path: Path, *, include_opal_progress: bool) -> dict[str, Any]:
    opal_progress = build_campaign_progress(config_path, round_selector="all")
    rounds = list(opal_progress.get("rounds") or [])
    latest = rounds[-1] if rounds else {}
    payload = {
        "run_key": config_path.parents[1].name,
        "config_path": str(config_path),
        "status": opal_progress.get("status"),
        "round_count": opal_progress.get("round_count"),
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


def format_opal_campaign_progress_text(payload: dict[str, Any]) -> str:
    return render_campaign_progress_text(payload)
