"""
Typed presentation surface for released-product snapback readback.
"""

from __future__ import annotations

from typing import Any, Literal

from dnadesign.cruncher.app.snapback_released_show_load import ReleasedShowArtifacts
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel


class ReleasedShowOutcome(StrictBaseModel):
    kind: Literal["released_explicit"] = "released_explicit"
    run_dir: str
    spec_name: str
    status: str
    status_message: str
    final_target: dict[str, int]
    final_geometry_source: str
    nick_catalog_source: str
    release_catalog_source: str
    manifest_path: str
    status_path: str
    report_json: str
    spec_snapshot: str
    nickase_catalog_snapshot: str
    release_catalog_snapshot: str
    projection_json: str
    pre_nick_site_json: str
    release_site_json: str
    summary_csv: str


def _required_text(value: object, *, error_message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error_message)
    return value


def _required_mapping(value: object, *, error_message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(error_message)
    return value


def build_released_show_payload(artifacts: ReleasedShowArtifacts) -> dict[str, object]:
    report_metadata = _required_mapping(
        artifacts.report_payload.get("metadata"),
        error_message="Released-product report metadata drift detected.",
    )
    return ReleasedShowOutcome(
        run_dir=str(artifacts.run_dir.resolve()),
        spec_name=_required_text(
            artifacts.manifest.get("spec_name"),
            error_message="Released-product manifest spec_name drift detected.",
        ),
        status=_required_text(
            artifacts.status.get("status"),
            error_message="Released-product status status drift detected.",
        ),
        status_message=_required_text(
            artifacts.status.get("status_message"),
            error_message="Released-product status status_message drift detected.",
        ),
        final_target=artifacts.spec_snapshot.final_target.model_dump(mode="json"),
        final_geometry_source=_required_text(
            report_metadata.get("final_geometry_source"),
            error_message="Released-product report final_geometry_source drift detected.",
        ),
        nick_catalog_source=_required_text(
            artifacts.manifest.get("nick_catalog_source"),
            error_message="Released-product manifest nick_catalog_source drift detected.",
        ),
        release_catalog_source=_required_text(
            artifacts.manifest.get("release_catalog_source"),
            error_message="Released-product manifest release_catalog_source drift detected.",
        ),
        manifest_path=str(artifacts.manifest_path.resolve()),
        status_path=str(artifacts.status_path.resolve()),
        report_json=str(artifacts.report_path.resolve()),
        spec_snapshot=str(artifacts.spec_snapshot_path.resolve()),
        nickase_catalog_snapshot=str(artifacts.nickase_catalog_snapshot_path.resolve()),
        release_catalog_snapshot=str(artifacts.release_catalog_snapshot_path.resolve()),
        projection_json=str(artifacts.projection_path.resolve()),
        pre_nick_site_json=str(artifacts.pre_nick_site_path.resolve()),
        release_site_json=str(artifacts.release_site_path.resolve()),
        summary_csv=str(artifacts.summary_csv_path.resolve()),
    ).model_dump(mode="json")


__all__ = ["ReleasedShowOutcome", "build_released_show_payload"]
