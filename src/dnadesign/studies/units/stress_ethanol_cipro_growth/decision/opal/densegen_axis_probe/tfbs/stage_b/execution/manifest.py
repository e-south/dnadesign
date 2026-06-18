"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/execution/manifest.py

Stage B execution manifest validation and writing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...stage_a.manifests import file_sha256
from ..semantics import (
    TFBS_STAGE_B_RETENTION_MODE,
    TFBS_STAGE_B_SCOPE,
    TFBS_STAGE_B_STAGE,
)
from .contracts import EXECUTION_MANIFEST_SCHEMA_VERSION, TfbsStageBExecutionConfig


def normalize_execution_config(config: TfbsStageBExecutionConfig) -> TfbsStageBExecutionConfig:
    manifest_path = Path(config.config_manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage B config manifest not found: {manifest_path}")
    repo_root = Path(config.repo_root)
    if not repo_root.exists():
        raise FileNotFoundError(f"repo root not found: {repo_root}")
    if config.rounds is not None and int(config.rounds) <= 0:
        raise ValueError("Stage B execution rounds must be positive")
    return TfbsStageBExecutionConfig(
        config_manifest_path=manifest_path,
        repo_root=repo_root,
        rounds=None if config.rounds is None else int(config.rounds),
        campaign_keys=tuple(dict.fromkeys(map(str, config.campaign_keys))),
        resume_existing=bool(config.resume_existing),
        machine_readable=bool(config.machine_readable),
    )


def validate_config_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "PASS":
        raise ValueError("Stage B execution requires config manifest status PASS")
    if manifest.get("stage") != TFBS_STAGE_B_STAGE:
        raise ValueError(f"Stage B execution requires stage {TFBS_STAGE_B_STAGE!r}")
    if manifest.get("scope") != TFBS_STAGE_B_SCOPE:
        raise ValueError(f"Stage B execution requires scope {TFBS_STAGE_B_SCOPE!r}")
    if manifest.get("validation", {}).get("status") != "PASS":
        raise ValueError("Stage B execution requires OPAL config validation PASS")
    if manifest.get("retention_mode") != TFBS_STAGE_B_RETENTION_MODE:
        raise ValueError("Stage B execution requires production_review retention mode")
    if int(manifest.get("campaign_count") or 0) <= 0:
        raise ValueError("Stage B execution requires at least one campaign")


def selected_campaign_rows(
    manifest: Mapping[str, Any],
    campaign_keys: Sequence[str],
) -> list[Mapping[str, Any]]:
    rows = manifest.get("campaigns")
    if not isinstance(rows, list):
        raise ValueError("Stage B config manifest campaigns must be a list")
    requested = set(map(str, campaign_keys))
    selected = [row for row in rows if isinstance(row, Mapping) and (not requested or row["campaign_key"] in requested)]
    found = {str(row["campaign_key"]) for row in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Stage B config manifest missing requested campaign key(s): {missing}")
    return sorted(selected, key=lambda row: str(row["campaign_key"]))


def build_execution_manifest(
    *,
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
    campaign_results: Sequence[Mapping[str, Any]],
    rounds: int,
) -> dict[str, Any]:
    return {
        "schema_version": EXECUTION_MANIFEST_SCHEMA_VERSION,
        "status": "PASS",
        "stage": TFBS_STAGE_B_STAGE,
        "scope": TFBS_STAGE_B_SCOPE,
        "source_config_manifest_path": str(source_manifest_path),
        "source_config_manifest_hash": file_sha256(source_manifest_path),
        "retention_mode": source_manifest["retention_mode"],
        "rounds": int(rounds),
        "campaign_count": len(campaign_results),
        "campaigns": sorted((dict(row) for row in campaign_results), key=lambda row: row["campaign_key"]),
    }


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
