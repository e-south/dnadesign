"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_released_show_load.py

Load released-product snapback bundle artifacts for readback validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.snapback.released_artifacts import (
    load_released_manifest,
    load_released_status,
    released_nickase_catalog_snapshot_path,
    released_pre_nick_site_json_path,
    released_projection_json_path,
    released_release_catalog_snapshot_path,
    released_release_site_json_path,
    released_report_json_path,
    released_spec_snapshot_path,
    released_status_path,
    released_summary_csv_path,
)
from dnadesign.cruncher.snapback.released_artifacts import (
    released_manifest_path as released_manifest_artifact_path,
)
from dnadesign.cruncher.snapback.released_models import SingleNickReleasedSnapbackSpec


@dataclass(frozen=True)
class ReleasedShowArtifacts:
    run_dir: Path
    manifest: dict[str, Any]
    status: dict[str, Any]
    spec_snapshot: SingleNickReleasedSnapbackSpec
    report_payload: dict[str, Any]
    projection_payload: Any
    pre_nick_payload: dict[str, Any]
    release_payload: dict[str, Any]

    @property
    def manifest_path(self) -> Path:
        return released_manifest_artifact_path(self.run_dir)

    @property
    def status_path(self) -> Path:
        return released_status_path(self.run_dir)

    @property
    def report_path(self) -> Path:
        return released_report_json_path(self.run_dir)

    @property
    def spec_snapshot_path(self) -> Path:
        return released_spec_snapshot_path(self.run_dir)

    @property
    def nickase_catalog_snapshot_path(self) -> Path:
        return released_nickase_catalog_snapshot_path(self.run_dir)

    @property
    def release_catalog_snapshot_path(self) -> Path:
        return released_release_catalog_snapshot_path(self.run_dir)

    @property
    def projection_path(self) -> Path:
        return released_projection_json_path(self.run_dir)

    @property
    def pre_nick_site_path(self) -> Path:
        return released_pre_nick_site_json_path(self.run_dir)

    @property
    def release_site_path(self) -> Path:
        return released_release_site_json_path(self.run_dir)

    @property
    def summary_csv_path(self) -> Path:
        return released_summary_csv_path(self.run_dir)

    def required_artifact_paths(self) -> tuple[Path, ...]:
        return _required_artifact_paths(self.run_dir)


def _required_artifact_paths(run_dir: Path) -> tuple[Path, ...]:
    return (
        released_report_json_path(run_dir),
        released_spec_snapshot_path(run_dir),
        released_nickase_catalog_snapshot_path(run_dir),
        released_release_catalog_snapshot_path(run_dir),
        released_projection_json_path(run_dir),
        released_pre_nick_site_json_path(run_dir),
        released_release_site_json_path(run_dir),
        released_summary_csv_path(run_dir),
    )


def _load_json_value(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise ValueError(f"Released-product {label} JSON is invalid.") from exc


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_json_value(path, label=label)
    if not isinstance(payload, dict):
        raise ValueError(f"Released-product {label} must be a JSON object.")
    return payload


def _load_released_spec_snapshot(path: Path) -> SingleNickReleasedSnapbackSpec:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError("Released-product spec snapshot YAML is invalid.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Released-product spec snapshot integrity drift detected.")
    try:
        return SingleNickReleasedSnapbackSpec.model_validate(payload)
    except Exception as exc:
        raise ValueError("Released-product spec snapshot integrity drift detected.") from exc


def load_released_show_artifacts(run_dir: str | Path) -> ReleasedShowArtifacts:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    manifest = load_released_manifest(resolved_run_dir)
    status = load_released_status(resolved_run_dir)
    for path in _required_artifact_paths(resolved_run_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required released-product artifact missing: {path}")
    return ReleasedShowArtifacts(
        run_dir=resolved_run_dir,
        manifest=manifest,
        status=status,
        spec_snapshot=_load_released_spec_snapshot(released_spec_snapshot_path(resolved_run_dir)),
        report_payload=_load_json_mapping(released_report_json_path(resolved_run_dir), label="report"),
        projection_payload=_load_json_value(released_projection_json_path(resolved_run_dir), label="projection"),
        pre_nick_payload=_load_json_mapping(
            released_pre_nick_site_json_path(resolved_run_dir),
            label="pre-nick payload",
        ),
        release_payload=_load_json_mapping(released_release_site_json_path(resolved_run_dir), label="release payload"),
    )


__all__ = ["ReleasedShowArtifacts", "load_released_show_artifacts"]
