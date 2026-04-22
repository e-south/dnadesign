"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_released_show.py

Path-oriented integrity checks for released-product snapback bundles.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from dnadesign.cruncher.snapback.released_artifacts import (
    RELEASED_SUMMARY_FIELDNAMES,
    load_released_manifest,
    load_released_status,
    released_manifest_path,
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
from dnadesign.cruncher.utils.hashing import sha256_path


def _expected_artifact_inventory(run_dir: Path) -> list[dict[str, str]]:
    return [
        {"name": "report_json", "path": str(released_report_json_path(run_dir).relative_to(run_dir))},
        {"name": "spec_snapshot", "path": str(released_spec_snapshot_path(run_dir).relative_to(run_dir))},
        {
            "name": "nickase_catalog_snapshot",
            "path": str(released_nickase_catalog_snapshot_path(run_dir).relative_to(run_dir)),
        },
        {
            "name": "release_catalog_snapshot",
            "path": str(released_release_catalog_snapshot_path(run_dir).relative_to(run_dir)),
        },
        {"name": "projection_json", "path": str(released_projection_json_path(run_dir).relative_to(run_dir))},
        {"name": "pre_nick_site_json", "path": str(released_pre_nick_site_json_path(run_dir).relative_to(run_dir))},
        {"name": "release_site_json", "path": str(released_release_site_json_path(run_dir).relative_to(run_dir))},
        {"name": "summary_csv", "path": str(released_summary_csv_path(run_dir).relative_to(run_dir))},
    ]


def _required_manifest_sha256(manifest: dict[str, Any], *, field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Released-product manifest {field} drift detected.")
    return value


def _required_manifest_text_field(manifest: dict[str, Any], *, field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Released-product manifest {field} drift detected.")
    return value


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


def _validate_timestamp(value: object, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Released-product {label} timestamp drift detected.")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Released-product {label} timestamp drift detected.") from exc


def _validate_released_report_payload(
    report_payload: dict[str, Any],
    *,
    expected_run_dir: str,
    expected_workspace_root: str,
    expected_contract: str,
) -> None:
    if report_payload.get("workflow") != "snapback_released_design":
        raise ValueError("Released-product report workflow drift detected.")
    if report_payload.get("run_dir") != expected_run_dir:
        raise ValueError("Released-product report run_dir drift detected.")
    if report_payload.get("workspace_root") != expected_workspace_root:
        raise ValueError("Released-product report workspace_root drift detected.")
    issues = report_payload.get("issues")
    if not isinstance(issues, list):
        raise ValueError("Released-product report issues drift detected.")
    metadata = report_payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Released-product report metadata drift detected.")
    if metadata.get("kind") != expected_contract:
        raise ValueError("Released-product report contract drift detected.")
    for field in ("nick_catalog_source", "release_catalog_source"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Released-product report {field} drift detected.")
    candidate = report_payload.get("candidate")
    if candidate is None:
        return
    if not isinstance(candidate, dict):
        raise ValueError("Released-product report candidate drift detected.")
    final_target = metadata.get("final_target")
    if not isinstance(final_target, dict):
        raise ValueError("Released-product report final_target drift detected.")
    expected_target = {
        "nick_boundary_from_left": candidate.get("nick_boundary_from_left"),
        "paired_bp": candidate.get("paired_bp"),
        "cap_nt": candidate.get("cap_nt"),
    }
    if {field: final_target.get(field) for field in expected_target} != expected_target:
        raise ValueError("Released-product report final_target drift detected.")


def _expected_released_summary_rows(report_payload: dict[str, Any]) -> list[dict[str, str]]:
    candidate = report_payload.get("candidate")
    projection = report_payload.get("projection")
    pre_nick_event = report_payload.get("pre_nick_event")
    release_event = report_payload.get("release_event")
    if not all(isinstance(payload, dict) for payload in (candidate, projection, pre_nick_event, release_event)):
        return []
    try:
        sacrificial_tail_nt = int(projection["precursor_length"]) - int(projection["release_top_cut_precursor"])
        return [
            {
                "status": str(report_payload["status"]),
                "spec_name": str(report_payload["spec_name"]),
                "nickase_variant_id": str(pre_nick_event["variant_id"]),
                "release_variant_id": str(release_event["variant_id"]),
                "nick_boundary_from_left": str(candidate["nick_boundary_from_left"]),
                "paired_bp": str(candidate["paired_bp"]),
                "cap_nt": str(candidate["cap_nt"]),
                "retained_input_length_nt": str(candidate["input_length_nt"]),
                "retained_product_length_nt": str(candidate["retained_product_length_nt"]),
                "precursor_length_nt": str(projection["precursor_length"]),
                "sacrificial_downstream_tail_nt": str(sacrificial_tail_nt),
                "extra_nick_event_count": str(candidate["extra_nick_event_count"]),
            }
        ]
    except KeyError as exc:
        raise ValueError("Released-product report payload drift detected.") from exc


def _validate_released_summary_csv(run_dir: Path, *, report_payload: dict[str, Any]) -> None:
    with released_summary_csv_path(run_dir).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RELEASED_SUMMARY_FIELDNAMES:
            raise ValueError("Released-product summary CSV columns drift detected.")
        observed_rows = [{field: str(row.get(field, "")) for field in RELEASED_SUMMARY_FIELDNAMES} for row in reader]
    expected_rows = _expected_released_summary_rows(report_payload)
    if observed_rows != expected_rows:
        raise ValueError("Released-product summary CSV content drift detected.")


def _validate_released_status_payload(status_payload: dict[str, Any], *, report_payload: dict[str, Any]) -> None:
    issues = report_payload.get("issues")
    if not isinstance(issues, list):
        raise ValueError("Released-product report issues drift detected.")
    expected_message = f"released-product snapback design {status_payload.get('status')}"
    if status_payload.get("status_message") != expected_message:
        raise ValueError("Released-product status/status_message drift detected.")
    if status_payload.get("issue_count") != len(issues):
        raise ValueError("Released-product status/report issue_count drift detected.")
    _validate_timestamp(status_payload.get("updated_at"), label="status")


def released_show_payload(run_dir: str | Path) -> dict[str, object]:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    manifest = load_released_manifest(resolved_run_dir)
    status = load_released_status(resolved_run_dir)
    expected_run_dir = str(resolved_run_dir)
    expected_workspace_root = _required_manifest_text_field(manifest, field="workspace_root")
    expected_spec_path = _required_manifest_text_field(manifest, field="spec_path")
    if manifest.get("kind") != "released_explicit":
        raise ValueError("Released-product manifest kind drift detected.")
    if manifest.get("workflow") != "snapback_released_design":
        raise ValueError("Released-product manifest workflow drift detected.")
    if manifest.get("contract") != "single_nick_released_snapback_v1":
        raise ValueError("Unsupported released-product snapback manifest contract version.")
    if status.get("workflow") != "snapback_released_design":
        raise ValueError("Released-product status workflow drift detected.")
    if status.get("contract") != "single_nick_released_snapback_v1":
        raise ValueError("Unsupported released-product snapback status contract version.")
    if manifest.get("stage") != "snapback_released" or status.get("stage") != "snapback_released":
        raise ValueError("Released-product snapback stage drift detected.")
    if manifest.get("run_dir") != expected_run_dir:
        raise ValueError("Released-product manifest run_dir drift detected.")
    if status.get("run_dir") != expected_run_dir:
        raise ValueError("Released-product status run_dir drift detected.")
    if manifest.get("spec_name") != status.get("spec_name"):
        raise ValueError("Released-product manifest/status spec_name drift detected.")
    if manifest.get("status") != status.get("status"):
        raise ValueError("Released-product manifest/status status drift detected.")
    if manifest.get("artifacts") != _expected_artifact_inventory(resolved_run_dir):
        raise ValueError("Released-product manifest artifact inventory drift detected.")
    required_paths = [
        released_report_json_path(resolved_run_dir),
        released_spec_snapshot_path(resolved_run_dir),
        released_nickase_catalog_snapshot_path(resolved_run_dir),
        released_release_catalog_snapshot_path(resolved_run_dir),
        released_projection_json_path(resolved_run_dir),
        released_pre_nick_site_json_path(resolved_run_dir),
        released_release_site_json_path(resolved_run_dir),
        released_summary_csv_path(resolved_run_dir),
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required released-product artifact missing: {path}")
    spec_sha256 = _required_manifest_sha256(manifest, field="spec_sha256")
    spec_snapshot_sha256 = _required_manifest_sha256(manifest, field="spec_snapshot_sha256")
    if spec_sha256 != spec_snapshot_sha256:
        raise ValueError("Released-product spec provenance hash drift detected.")
    if sha256_path(released_spec_snapshot_path(resolved_run_dir)) != spec_snapshot_sha256:
        raise ValueError("Released-product spec snapshot integrity drift detected.")
    if sha256_path(released_nickase_catalog_snapshot_path(resolved_run_dir)) != _required_manifest_sha256(
        manifest,
        field="nickase_catalog_sha256",
    ):
        raise ValueError("Released-product nickase catalog snapshot integrity drift detected.")
    if sha256_path(released_release_catalog_snapshot_path(resolved_run_dir)) != _required_manifest_sha256(
        manifest,
        field="release_catalog_sha256",
    ):
        raise ValueError("Released-product release catalog snapshot integrity drift detected.")
    report_payload = _load_json_mapping(released_report_json_path(resolved_run_dir), label="report")
    projection_payload = _load_json_value(released_projection_json_path(resolved_run_dir), label="projection")
    pre_nick_payload = _load_json_mapping(released_pre_nick_site_json_path(resolved_run_dir), label="pre-nick payload")
    release_payload = _load_json_mapping(released_release_site_json_path(resolved_run_dir), label="release payload")
    _validate_released_report_payload(
        report_payload,
        expected_run_dir=expected_run_dir,
        expected_workspace_root=expected_workspace_root,
        expected_contract=str(manifest.get("contract")),
    )
    _validate_released_status_payload(status, report_payload=report_payload)
    if report_payload.get("spec_path") != expected_spec_path:
        raise ValueError("Released-product report/manifest spec_path drift detected.")
    if report_payload.get("spec_name") != manifest.get("spec_name"):
        raise ValueError("Released-product report/manifest spec_name drift detected.")
    if report_payload.get("status") != status.get("status"):
        raise ValueError("Released-product report/status drift detected.")
    if report_payload.get("projection") != projection_payload:
        raise ValueError("Released-product projection artifact drift detected.")
    if report_payload.get("pre_nick_site") != pre_nick_payload.get("site"):
        raise ValueError("Released-product pre-nick site artifact drift detected.")
    if report_payload.get("pre_nick_event") != pre_nick_payload.get("event"):
        raise ValueError("Released-product pre-nick event artifact drift detected.")
    if report_payload.get("release_site") != release_payload.get("site"):
        raise ValueError("Released-product release site artifact drift detected.")
    if report_payload.get("release_event") != release_payload.get("event"):
        raise ValueError("Released-product release event artifact drift detected.")
    _validate_released_summary_csv(resolved_run_dir, report_payload=report_payload)
    return {
        "kind": "released_explicit",
        "run_dir": expected_run_dir,
        "spec_name": manifest.get("spec_name"),
        "status": status.get("status"),
        "status_message": status.get("status_message"),
        "manifest_path": str(released_manifest_path(resolved_run_dir).resolve()),
        "status_path": str(released_status_path(resolved_run_dir).resolve()),
        "report_json": str(released_report_json_path(resolved_run_dir).resolve()),
        "projection_json": str(released_projection_json_path(resolved_run_dir).resolve()),
        "pre_nick_site_json": str(released_pre_nick_site_json_path(resolved_run_dir).resolve()),
        "release_site_json": str(released_release_site_json_path(resolved_run_dir).resolve()),
        "summary_csv": str(released_summary_csv_path(resolved_run_dir).resolve()),
    }


__all__ = ["released_show_payload"]
