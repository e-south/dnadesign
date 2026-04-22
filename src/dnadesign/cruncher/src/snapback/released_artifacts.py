"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_artifacts.py

Artifact paths and persistence helpers for released-product snapback bundles.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_text
from dnadesign.cruncher.snapback.released_models import ReleasedSnapbackEvaluationReport
from dnadesign.cruncher.utils.hashing import sha256_bytes, sha256_path

RUN_META_DIR = "meta"
RUN_PROVENANCE_DIR = "provenance"
RUN_ANALYSIS_DIR = "analysis"
RUN_EXPORT_DIR = "export"


def released_design_id(*, spec_bytes: bytes, nick_catalog_bytes: bytes, release_catalog_bytes: bytes) -> str:
    return sha256_bytes(spec_bytes + b"\n" + nick_catalog_bytes + b"\n" + release_catalog_bytes)[:12]


def _scoped_run_dir(workspace_root: Path, *parts: Path | str) -> Path:
    resolved_workspace_root = workspace_root.resolve()
    candidate = resolved_workspace_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(resolved_workspace_root)
    except ValueError as exc:
        raise ValueError(
            f"Released-product snapback run directory must stay inside workspace {resolved_workspace_root}: {candidate}"
        ) from exc
    return candidate


def build_released_run_dir(*, workspace_root: Path, run_root: Path, released_design_run_id: str) -> Path:
    del released_design_run_id
    return _scoped_run_dir(workspace_root, run_root)


def ensure_released_run_dirs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_META_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_PROVENANCE_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_EXPORT_DIR).mkdir(parents=True, exist_ok=True)


def released_manifest_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "released_snapback_manifest.json"


def released_status_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "released_snapback_status.json"


def released_spec_snapshot_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR / "spec.snapshot.yaml"


def released_nickase_catalog_snapshot_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR / "nickase_catalog.yaml"


def released_release_catalog_snapshot_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR / "release_catalog.yaml"


def released_report_json_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / "report.json"


def released_projection_json_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / "released_product_projection.json"


def released_pre_nick_site_json_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / "pre_nick_site.json"


def released_release_site_json_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / "release_site.json"


def released_summary_csv_path(run_dir: Path) -> Path:
    return run_dir / RUN_EXPORT_DIR / "released_design_summary.csv"


def build_released_manifest(
    *,
    run_dir: Path,
    workspace_root: Path,
    spec_path: Path,
    report: ReleasedSnapbackEvaluationReport,
) -> dict[str, Any]:
    return {
        "kind": "released_explicit",
        "stage": "snapback_released",
        "workflow": "snapback_released_design",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "spec_name": report.spec_name,
        "status": report.status,
        "contract": report.metadata.kind,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256_path(spec_path),
        "spec_snapshot_sha256": sha256_path(released_spec_snapshot_path(run_dir)),
        "nickase_catalog_sha256": sha256_path(released_nickase_catalog_snapshot_path(run_dir)),
        "release_catalog_sha256": sha256_path(released_release_catalog_snapshot_path(run_dir)),
        "artifacts": [
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
        ],
    }


def write_released_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = released_manifest_path(run_dir)
    atomic_write_json(path, manifest)
    return path


def write_released_status(run_dir: Path, *, report: ReleasedSnapbackEvaluationReport) -> Path:
    path = released_status_path(run_dir)
    payload = {
        "workflow": "snapback_released_design",
        "stage": "snapback_released",
        "contract": report.metadata.kind,
        "status": report.status,
        "status_message": f"released-product snapback design {report.status}",
        "run_dir": str(run_dir.resolve()),
        "spec_name": report.spec_name,
        "issue_count": len(report.issues),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(path, payload)
    return path


def snapshot_released_inputs(
    run_dir: Path,
    *,
    spec_path: Path,
    nick_catalog_yaml: str,
    release_catalog_yaml: str,
) -> None:
    atomic_write_text(released_spec_snapshot_path(run_dir), spec_path.read_text(encoding="utf-8"))
    atomic_write_text(released_nickase_catalog_snapshot_path(run_dir), nick_catalog_yaml)
    atomic_write_text(released_release_catalog_snapshot_path(run_dir), release_catalog_yaml)


def write_released_report(run_dir: Path, report: ReleasedSnapbackEvaluationReport) -> None:
    atomic_write_json(released_report_json_path(run_dir), report.model_dump(mode="json"))
    projection_payload = report.projection.model_dump(mode="json") if report.projection is not None else None
    atomic_write_json(released_projection_json_path(run_dir), projection_payload)
    atomic_write_json(
        released_pre_nick_site_json_path(run_dir),
        {
            "site": report.pre_nick_site.model_dump(mode="json") if report.pre_nick_site is not None else None,
            "event": report.pre_nick_event.model_dump(mode="json") if report.pre_nick_event is not None else None,
        },
    )
    atomic_write_json(
        released_release_site_json_path(run_dir),
        {
            "site": report.release_site.model_dump(mode="json") if report.release_site is not None else None,
            "event": report.release_event.model_dump(mode="json") if report.release_event is not None else None,
        },
    )


def write_released_summary_table(run_dir: Path, report: ReleasedSnapbackEvaluationReport) -> None:
    fieldnames = [
        "status",
        "spec_name",
        "nickase_variant_id",
        "release_variant_id",
        "nick_boundary_from_left",
        "paired_bp",
        "cap_nt",
        "retained_input_length_nt",
        "retained_product_length_nt",
        "precursor_length_nt",
        "sacrificial_downstream_tail_nt",
        "extra_nick_event_count",
    ]
    with released_summary_csv_path(run_dir).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if (
            report.candidate is None
            or report.projection is None
            or report.pre_nick_event is None
            or report.release_event is None
        ):
            return
        sacrificial_tail_nt = report.projection.precursor_length - report.projection.release_top_cut_precursor
        writer.writerow(
            {
                "status": report.status,
                "spec_name": report.spec_name,
                "nickase_variant_id": report.pre_nick_event.variant_id,
                "release_variant_id": report.release_event.variant_id,
                "nick_boundary_from_left": report.candidate.nick_boundary_from_left,
                "paired_bp": report.candidate.paired_bp,
                "cap_nt": report.candidate.cap_nt,
                "retained_input_length_nt": report.candidate.input_length_nt,
                "retained_product_length_nt": report.candidate.retained_product_length_nt,
                "precursor_length_nt": report.projection.precursor_length,
                "sacrificial_downstream_tail_nt": sacrificial_tail_nt,
                "extra_nick_event_count": report.candidate.extra_nick_event_count,
            }
        )


def load_released_manifest(run_dir: Path) -> dict[str, Any]:
    path = released_manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Released-product snapback manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_released_status(run_dir: Path) -> dict[str, Any]:
    path = released_status_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Released-product snapback status missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "build_released_manifest",
    "build_released_run_dir",
    "ensure_released_run_dirs",
    "load_released_manifest",
    "load_released_status",
    "released_design_id",
    "released_manifest_path",
    "released_nickase_catalog_snapshot_path",
    "released_pre_nick_site_json_path",
    "released_projection_json_path",
    "released_release_catalog_snapshot_path",
    "released_release_site_json_path",
    "released_report_json_path",
    "released_spec_snapshot_path",
    "released_status_path",
    "released_summary_csv_path",
    "snapshot_released_inputs",
    "write_released_manifest",
    "write_released_report",
    "write_released_status",
    "write_released_summary_table",
]
