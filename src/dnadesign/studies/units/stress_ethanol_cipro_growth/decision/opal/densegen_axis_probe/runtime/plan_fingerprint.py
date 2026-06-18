"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/runtime/plan_fingerprint.py

Run-root plan fingerprinting for the DenseGen OPAL probe.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from ..core.artifacts import ProbeArtifactLayout

PLAN_SCHEMA_VERSION = "stress_ethanol_cipro_growth.opal_densegen_axis_probe.plan.v1"


def plan_fingerprint(plan_payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(plan_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def build_plan_record(plan_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "fingerprint": plan_fingerprint(plan_payload),
        "plan": dict(plan_payload),
    }


def prepare_probe_run_root(
    layout: ProbeArtifactLayout,
    *,
    plan_payload: Mapping[str, Any],
    replace_run_root: bool = False,
) -> dict[str, Any]:
    plan_record = build_plan_record(plan_payload)
    path = layout.probe_plan_path

    if replace_run_root and layout.run_root.exists():
        shutil.rmtree(layout.run_root)

    if path.exists():
        existing = _load_plan_record(path)
        if existing.get("fingerprint") == plan_record["fingerprint"]:
            return existing
        if not replace_run_root:
            raise RuntimeError(
                "probe plan fingerprint mismatch for existing run root; "
                f"existing={existing.get('fingerprint')!r} current={plan_record['fingerprint']!r}. "
                "Use a new --run-id or pass --replace-run-root to delete and rebuild this scratch root."
            )
    elif _has_existing_probe_artifacts(layout.run_root):
        if not replace_run_root:
            raise RuntimeError(
                "existing probe run root has artifacts but no probe_plan.json; "
                "use a new --run-id or pass --replace-run-root to delete and rebuild this scratch root."
            )

    layout.run_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan_record


def load_probe_plan_record(run_root: Path) -> dict[str, Any] | None:
    path = ProbeArtifactLayout(Path(run_root)).probe_plan_path
    if not path.exists():
        return None
    return _load_plan_record(path)


def _load_plan_record(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"probe_plan.json is malformed: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"probe_plan.json must contain a JSON object: {path}")
    if payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported probe_plan.json schema: {payload.get('schema_version')!r}")
    if not isinstance(payload.get("fingerprint"), str):
        raise RuntimeError("probe_plan.json missing string fingerprint")
    if not isinstance(payload.get("plan"), dict):
        raise RuntimeError("probe_plan.json missing object plan")
    expected = plan_fingerprint(payload["plan"])
    if payload["fingerprint"] != expected:
        raise RuntimeError(
            f"probe_plan.json fingerprint mismatch: stored={payload['fingerprint']!r} expected={expected!r}"
        )
    return payload


def _has_existing_probe_artifacts(root: Path) -> bool:
    if not root.exists():
        return False
    return any(root.iterdir())
