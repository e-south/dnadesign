"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_released_show.py

Path-oriented integrity checks for released-product snapback bundles.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.cruncher.snapback.released_artifacts import (
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


def released_show_payload(run_dir: str | Path) -> dict[str, object]:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    manifest = load_released_manifest(resolved_run_dir)
    status = load_released_status(resolved_run_dir)
    expected_run_dir = str(resolved_run_dir)
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
    report_payload = json.loads(released_report_json_path(resolved_run_dir).read_text(encoding="utf-8"))
    projection_payload = json.loads(released_projection_json_path(resolved_run_dir).read_text(encoding="utf-8"))
    pre_nick_payload = json.loads(released_pre_nick_site_json_path(resolved_run_dir).read_text(encoding="utf-8"))
    release_payload = json.loads(released_release_site_json_path(resolved_run_dir).read_text(encoding="utf-8"))
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
