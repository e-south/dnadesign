"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/reporting/review/run_manifest.py

Run-manifest assembly for DenseGen axis probe reviews.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ...core.artifacts import ProbeArtifactLayout
from ...runtime.plan_fingerprint import load_probe_plan_record


def _build_run_manifest(
    layout: ProbeArtifactLayout,
    *,
    audit,
    metrics_payload: Mapping[str, Any],
    review_decision: str | None,
    review_status: str,
    review_problems: list[str],
    decision_reasons: list[dict[str, Any]],
    gate_results: list[dict[str, Any]],
    metric_quality: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = _artifact_inventory(layout.run_root)
    plan_record = load_probe_plan_record(layout.run_root)
    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.run_manifest.v1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_root": str(layout.run_root),
        "plan_fingerprint": plan_record.get("fingerprint") if plan_record else None,
        "plan_path": str(layout.probe_plan_path) if plan_record else None,
        "decision": review_decision,
        "persisted_decision": audit.decision,
        "status": review_status,
        "planned_campaign_count": audit.planned_campaign_count,
        "metrics_run_count": len(metrics_payload.get("runs") or []),
        "shared_sidecar_present": audit.shared_sidecar_present,
        "artifact_inventory": inventory,
        "problems": review_problems,
        "decision_reasons": decision_reasons,
        "gate_results": gate_results,
        "metric_quality": dict(metric_quality),
    }


def _artifact_inventory(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"file_count": 0, "total_bytes": 0}
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += int(path.stat().st_size)
    return {"file_count": file_count, "total_bytes": total_bytes}
