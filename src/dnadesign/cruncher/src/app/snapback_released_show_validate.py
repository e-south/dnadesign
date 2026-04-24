"""
Validate released-product snapback bundle drift and readback invariants.
"""

from __future__ import annotations

import csv
from datetime import datetime
from typing import Any

from dnadesign.cruncher.app.snapback_released_show_load import ReleasedShowArtifacts
from dnadesign.cruncher.snapback.released_artifacts import RELEASED_SUMMARY_FIELDNAMES
from dnadesign.cruncher.snapback.released_models import ReleasedSnapbackEvaluationReport
from dnadesign.cruncher.utils.hashing import sha256_path


def _expected_artifact_inventory(artifacts: ReleasedShowArtifacts) -> list[dict[str, str]]:
    run_dir = artifacts.run_dir
    return [
        {"name": "report_json", "path": str(artifacts.report_path.relative_to(run_dir))},
        {"name": "spec_snapshot", "path": str(artifacts.spec_snapshot_path.relative_to(run_dir))},
        {
            "name": "nickase_catalog_snapshot",
            "path": str(artifacts.nickase_catalog_snapshot_path.relative_to(run_dir)),
        },
        {
            "name": "release_catalog_snapshot",
            "path": str(artifacts.release_catalog_snapshot_path.relative_to(run_dir)),
        },
        {"name": "projection_json", "path": str(artifacts.projection_path.relative_to(run_dir))},
        {"name": "pre_nick_site_json", "path": str(artifacts.pre_nick_site_path.relative_to(run_dir))},
        {"name": "release_site_json", "path": str(artifacts.release_site_path.relative_to(run_dir))},
        {"name": "summary_csv", "path": str(artifacts.summary_csv_path.relative_to(run_dir))},
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


def _required_manifest_mapping_field(manifest: dict[str, Any], *, field: str) -> dict[str, Any]:
    value = manifest.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"Released-product manifest {field} drift detected.")
    return value


def _required_status_text_field(status: dict[str, Any], *, field: str) -> str:
    value = status.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Released-product status {field} drift detected.")
    return value


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
    expected_nick_catalog_source: str,
    expected_release_catalog_source: str,
    expected_disallowed_nickase_warning_codes: list[str],
    expected_final_target: dict[str, int],
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
    if metadata.get("final_geometry_source") not in {"exposed_bottom_strand", "retained_active_strand"}:
        raise ValueError("Released-product report final_geometry_source drift detected.")
    if metadata.get("nick_catalog_source") != expected_nick_catalog_source:
        raise ValueError("Released-product report nick_catalog_source drift detected.")
    if metadata.get("release_catalog_source") != expected_release_catalog_source:
        raise ValueError("Released-product report release_catalog_source drift detected.")
    if metadata.get("disallowed_nickase_warning_codes") != expected_disallowed_nickase_warning_codes:
        raise ValueError("Released-product report disallowed_nickase_warning_codes drift detected.")
    final_target = metadata.get("final_target")
    if not isinstance(final_target, dict):
        raise ValueError("Released-product report final_target drift detected.")
    if {field: final_target.get(field) for field in expected_final_target} != expected_final_target:
        raise ValueError("Released-product report final_target drift detected.")
    candidate = report_payload.get("candidate")
    if report_payload.get("status") == "satisfied":
        for field in (
            "candidate",
            "projection",
            "pre_nick_site",
            "pre_nick_event",
            "release_site",
            "release_event",
        ):
            value = report_payload.get(field)
            if not isinstance(value, dict):
                raise ValueError(f"Released-product satisfied report {field} drift detected.")
    elif candidate is None:
        return
    if not isinstance(candidate, dict):
        raise ValueError("Released-product report candidate drift detected.")
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
        sacrificial_tail_nt = int(projection["precursor_length"]) - max(
            int(projection["release_top_cut_precursor"]),
            int(projection["release_bottom_cut_precursor"]),
        )
        return [
            {
                "status": str(report_payload["status"]),
                "spec_name": str(report_payload["spec_name"]),
                "final_geometry_source": str(report_payload["metadata"]["final_geometry_source"]),
                "route_family": str(candidate["route_family"]),
                "active_strand": str(candidate["active_strand"]),
                "retained_partner_strand": str(projection["retained_partner_strand"]),
                "physical_nicked_strand": str(candidate["physical_nicked_strand"]),
                "nickase_variant_id": str(pre_nick_event["variant_id"]),
                "release_variant_id": str(release_event["variant_id"]),
                "nick_boundary_from_left": str(candidate["nick_boundary_from_left"]),
                "paired_bp": str(candidate["paired_bp"]),
                "cap_nt": str(candidate["cap_nt"]),
                "active_product_input_length_nt": str(candidate["active_product_input_length_nt"]),
                "active_product_length_nt": str(candidate["active_product_length_nt"]),
                "retained_partner_length_nt": str(projection["retained_partner_length_nt"]),
                "precursor_length_nt": str(projection["precursor_length"]),
                "sacrificial_downstream_tail_nt": str(sacrificial_tail_nt),
                "extra_nick_event_count": str(candidate["extra_nick_event_count"]),
            }
        ]
    except KeyError as exc:
        raise ValueError("Released-product report payload drift detected.") from exc


def _validate_released_summary_csv(artifacts: ReleasedShowArtifacts) -> None:
    with artifacts.summary_csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RELEASED_SUMMARY_FIELDNAMES:
            raise ValueError("Released-product summary CSV columns drift detected.")
        observed_rows = [{field: str(row.get(field, "")) for field in RELEASED_SUMMARY_FIELDNAMES} for row in reader]
    expected_rows = _expected_released_summary_rows(artifacts.report_payload)
    if observed_rows != expected_rows:
        raise ValueError("Released-product summary CSV content drift detected.")


def _validate_released_status_payload(status_payload: dict[str, Any], *, report_payload: dict[str, Any]) -> None:
    issues = report_payload.get("issues")
    if not isinstance(issues, list):
        raise ValueError("Released-product report issues drift detected.")
    _required_status_text_field(status_payload, field="status")
    expected_message = f"released-product snapback design {status_payload.get('status')}"
    if status_payload.get("status_message") != expected_message:
        raise ValueError("Released-product status/status_message drift detected.")
    if status_payload.get("issue_count") != len(issues):
        raise ValueError("Released-product status/report issue_count drift detected.")
    _validate_timestamp(status_payload.get("updated_at"), label="status")


def validate_released_show_artifacts(artifacts: ReleasedShowArtifacts) -> None:
    manifest = artifacts.manifest
    status = artifacts.status
    report_payload = artifacts.report_payload
    expected_run_dir = str(artifacts.run_dir)
    expected_workspace_root = _required_manifest_text_field(manifest, field="workspace_root")
    expected_spec_path = _required_manifest_text_field(manifest, field="spec_path")
    expected_spec_name = _required_manifest_text_field(manifest, field="spec_name")
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
    if expected_spec_name != _required_status_text_field(status, field="spec_name"):
        raise ValueError("Released-product manifest/status spec_name drift detected.")
    if manifest.get("status") != status.get("status"):
        raise ValueError("Released-product manifest/status status drift detected.")
    if manifest.get("artifacts") != _expected_artifact_inventory(artifacts):
        raise ValueError("Released-product manifest artifact inventory drift detected.")
    spec_sha256 = _required_manifest_sha256(manifest, field="spec_sha256")
    spec_snapshot_sha256 = _required_manifest_sha256(manifest, field="spec_snapshot_sha256")
    if spec_sha256 != spec_snapshot_sha256:
        raise ValueError("Released-product spec provenance hash drift detected.")
    if sha256_path(artifacts.spec_snapshot_path) != spec_snapshot_sha256:
        raise ValueError("Released-product spec snapshot integrity drift detected.")
    if sha256_path(artifacts.nickase_catalog_snapshot_path) != _required_manifest_sha256(
        manifest,
        field="nickase_catalog_sha256",
    ):
        raise ValueError("Released-product nickase catalog snapshot integrity drift detected.")
    if sha256_path(artifacts.release_catalog_snapshot_path) != _required_manifest_sha256(
        manifest,
        field="release_catalog_sha256",
    ):
        raise ValueError("Released-product release catalog snapshot integrity drift detected.")
    try:
        ReleasedSnapbackEvaluationReport.model_validate(report_payload)
    except Exception as exc:
        raise ValueError("Released-product report payload drift detected.") from exc
    manifest_target = _required_manifest_mapping_field(manifest, field="final_target")
    expected_final_target = {
        "nick_boundary_from_left": manifest_target.get("nick_boundary_from_left"),
        "paired_bp": manifest_target.get("paired_bp"),
        "cap_nt": manifest_target.get("cap_nt"),
    }
    _validate_released_report_payload(
        report_payload,
        expected_run_dir=expected_run_dir,
        expected_workspace_root=expected_workspace_root,
        expected_contract=str(manifest.get("contract")),
        expected_nick_catalog_source=_required_manifest_text_field(manifest, field="nick_catalog_source"),
        expected_release_catalog_source=_required_manifest_text_field(manifest, field="release_catalog_source"),
        expected_disallowed_nickase_warning_codes=list(
            artifacts.spec_snapshot.constraints.disallowed_nickase_warning_codes
        ),
        expected_final_target=expected_final_target,
    )
    _validate_released_status_payload(status, report_payload=report_payload)
    if report_payload.get("spec_path") != expected_spec_path:
        raise ValueError("Released-product report/manifest spec_path drift detected.")
    if report_payload.get("spec_name") != expected_spec_name:
        raise ValueError("Released-product report/manifest spec_name drift detected.")
    if report_payload.get("status") != status.get("status"):
        raise ValueError("Released-product report/status drift detected.")
    if report_payload.get("projection") != artifacts.projection_payload:
        raise ValueError("Released-product projection artifact drift detected.")
    if (
        isinstance(report_payload.get("candidate"), dict)
        and isinstance(report_payload.get("projection"), dict)
        and report_payload["candidate"].get("route_family") != report_payload["projection"].get("route_family")
    ):
        raise ValueError("Released-product candidate/projection route_family drift detected.")
    if (
        isinstance(report_payload.get("candidate"), dict)
        and isinstance(report_payload.get("projection"), dict)
        and report_payload["candidate"].get("active_strand") != report_payload["projection"].get("active_strand")
    ):
        raise ValueError("Released-product candidate/projection active_strand drift detected.")
    if (
        isinstance(report_payload.get("candidate"), dict)
        and isinstance(report_payload.get("projection"), dict)
        and report_payload["candidate"].get("physical_nicked_strand")
        != report_payload["projection"].get("physical_nicked_strand")
    ):
        raise ValueError("Released-product candidate/projection physical_nicked_strand drift detected.")
    if report_payload.get("pre_nick_site") != artifacts.pre_nick_payload.get("site"):
        raise ValueError("Released-product pre-nick site artifact drift detected.")
    if report_payload.get("pre_nick_event") != artifacts.pre_nick_payload.get("event"):
        raise ValueError("Released-product pre-nick event artifact drift detected.")
    if report_payload.get("release_site") != artifacts.release_payload.get("site"):
        raise ValueError("Released-product release site artifact drift detected.")
    if report_payload.get("release_event") != artifacts.release_payload.get("event"):
        raise ValueError("Released-product release event artifact drift detected.")
    _validate_released_summary_csv(artifacts)


__all__ = ["validate_released_show_artifacts"]
