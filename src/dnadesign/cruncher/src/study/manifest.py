"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/study/manifest.py

Manifest and status models for Study runs with atomic persistence helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel

TrialStatus = Literal["pending", "running", "success", "error", "skipped"]
StudyStatusLabel = Literal["running", "completed", "completed_with_errors", "failed"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StudyTrialRun(StrictBaseModel):
    trial_id: str
    seed: int
    target_set_index: int
    target_tfs: list[str]
    factors: dict[str, Any] = Field(default_factory=dict)
    factor_columns: dict[str, Any] = Field(default_factory=dict)
    status: TrialStatus = "pending"
    run_dir: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class StudyManifestV1(StrictBaseModel):
    schema_version: int = 1
    study_name: str
    study_id: str
    spec_path: str
    spec_sha256: str
    base_config_path: str
    base_config_sha256: str
    created_at: str
    trial_runs: list[StudyTrialRun] = Field(default_factory=list)


class StudyStatusV1(StrictBaseModel):
    schema_version: int = 1
    study_name: str
    study_id: str
    status: StudyStatusLabel = "running"
    total_runs: int = 0
    pending_runs: int = 0
    running_runs: int = 0
    success_runs: int = 0
    error_runs: int = 0
    skipped_runs: int = 0
    warnings: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    finished_at: str | None = None


def summarize_trial_statuses(trial_runs: list[StudyTrialRun]) -> dict[str, int]:
    counts = {
        "total_runs": len(trial_runs),
        "pending_runs": 0,
        "running_runs": 0,
        "success_runs": 0,
        "error_runs": 0,
        "skipped_runs": 0,
    }
    for item in trial_runs:
        key = f"{item.status}_runs"
        if key not in counts:
            raise ValueError(f"Unsupported trial status: {item.status}")
        counts[key] += 1
    return counts


def write_study_manifest(path: Path, manifest: StudyManifestV1) -> None:
    atomic_write_json(path, manifest.model_dump(mode="json"), indent=2, sort_keys=False, allow_nan=False)


def write_study_status(path: Path, status: StudyStatusV1) -> None:
    atomic_write_json(path, status.model_dump(mode="json"), indent=2, sort_keys=False, allow_nan=False)


def load_study_manifest(path: Path) -> StudyManifestV1:
    if not path.exists():
        raise FileNotFoundError(f"Study manifest not found: {path}")
    import json

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Study manifest must be a JSON object: {path}")
    return StudyManifestV1.model_validate(payload)


def load_study_status(path: Path) -> StudyStatusV1:
    if not path.exists():
        raise FileNotFoundError(f"Study status not found: {path}")
    import json

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Study status must be a JSON object: {path}")
    return StudyStatusV1.model_validate(payload)
