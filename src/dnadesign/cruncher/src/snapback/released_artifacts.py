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
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_text
from dnadesign.cruncher.snapback.released_models import ReleasedSnapbackEvaluationReport, ReleasedSolveReport
from dnadesign.cruncher.utils.hashing import sha256_path

RUN_META_DIR = "meta"
RUN_PROVENANCE_DIR = "provenance"
RUN_ANALYSIS_DIR = "analysis"
RUN_MATERIALIZED_HITS_DIR = "materialized_hits"
RUN_EXPORT_DIR = "export"
RELEASED_SUMMARY_FIELDNAMES = [
    "status",
    "spec_name",
    "final_geometry_source",
    "route_family",
    "active_strand",
    "retained_partner_strand",
    "physical_nicked_strand",
    "nickase_variant_id",
    "release_variant_id",
    "nick_boundary_from_left",
    "paired_bp",
    "cap_nt",
    "active_product_input_length_nt",
    "active_product_length_nt",
    "retained_partner_length_nt",
    "precursor_length_nt",
    "sacrificial_downstream_tail_nt",
    "extra_nick_event_count",
]
RELEASED_SOLVE_SUMMARY_FIELDNAMES = [
    "rank",
    "hit_kind",
    "final_geometry_source",
    "route_family",
    "active_strand",
    "retained_partner_strand",
    "physical_nicked_strand",
    "nickase_variant_id",
    "release_variant_id",
    "nick_boundary_from_left",
    "paired_bp",
    "cap_nt",
    "active_product_input_length_nt",
    "active_product_length_nt",
    "retained_partner_length_nt",
    "precursor_length_nt",
    "extra_nick_event_count",
    "extra_target_strand_nick_count",
    "materialized_run_dir",
    "render_job_path",
    "rendered_plot_path",
]


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


def build_released_run_dir(*, workspace_root: Path, run_root: Path) -> Path:
    return _scoped_run_dir(workspace_root, run_root)


def ensure_released_run_dirs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_META_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_PROVENANCE_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_EXPORT_DIR).mkdir(parents=True, exist_ok=True)


def ensure_released_solve_run_dirs(run_dir: Path) -> None:
    ensure_released_run_dirs(run_dir)
    released_solve_materialized_hits_dir(run_dir).mkdir(parents=True, exist_ok=True)


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


def released_solve_manifest_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "released_solve_manifest.json"


def released_solve_status_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "released_solve_status.json"


def released_solve_request_snapshot_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR / "request.snapshot.yaml"


def released_solve_report_json_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / "solve_report.json"


def released_solve_hits_csv_path(run_dir: Path) -> Path:
    return run_dir / RUN_EXPORT_DIR / "table__hits.csv"


def released_solve_materialized_hits_dir(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / RUN_MATERIALIZED_HITS_DIR


def released_solve_hit_run_dir(run_dir: Path, *, rank: int) -> Path:
    return released_solve_materialized_hits_dir(run_dir) / f"hit_{rank:02d}"


def released_solve_hit_json_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / "target_search_hit.json"


def released_solve_hit_plot_context_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / "released_hit_plot_context.json"


def released_solve_hit_plot_path(run_dir: Path, *, fmt: str) -> Path:
    return run_dir / "plots" / f"released_hit_triptych.{fmt}"


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
        "nick_catalog_source": report.metadata.nick_catalog_source,
        "release_catalog_source": report.metadata.release_catalog_source,
        "final_target": report.metadata.final_target.model_dump(mode="json"),
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
    with released_summary_csv_path(run_dir).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RELEASED_SUMMARY_FIELDNAMES)
        writer.writeheader()
        if (
            report.candidate is None
            or report.projection is None
            or report.pre_nick_event is None
            or report.release_event is None
        ):
            return
        sacrificial_tail_nt = report.projection.precursor_length - max(
            report.projection.release_top_cut_precursor,
            report.projection.release_bottom_cut_precursor,
        )
        writer.writerow(
            {
                "status": report.status,
                "spec_name": report.spec_name,
                "final_geometry_source": report.metadata.final_geometry_source,
                "route_family": report.candidate.route_family,
                "active_strand": report.candidate.active_strand,
                "retained_partner_strand": report.projection.retained_partner_strand,
                "physical_nicked_strand": report.candidate.physical_nicked_strand,
                "nickase_variant_id": report.pre_nick_event.variant_id,
                "release_variant_id": report.release_event.variant_id,
                "nick_boundary_from_left": report.candidate.nick_boundary_from_left,
                "paired_bp": report.candidate.paired_bp,
                "cap_nt": report.candidate.cap_nt,
                "active_product_input_length_nt": report.candidate.active_product_input_length_nt,
                "active_product_length_nt": report.candidate.active_product_length_nt,
                "retained_partner_length_nt": report.projection.retained_partner_length_nt,
                "precursor_length_nt": report.projection.precursor_length,
                "sacrificial_downstream_tail_nt": sacrificial_tail_nt,
                "extra_nick_event_count": report.candidate.extra_nick_event_count,
            }
        )


def snapshot_released_solve_inputs(
    run_dir: Path,
    *,
    request_yaml: str,
    nick_catalog_yaml: str,
    release_catalog_yaml: str,
) -> None:
    atomic_write_text(released_solve_request_snapshot_path(run_dir), request_yaml)
    atomic_write_text(released_nickase_catalog_snapshot_path(run_dir), nick_catalog_yaml)
    atomic_write_text(released_release_catalog_snapshot_path(run_dir), release_catalog_yaml)


def write_released_solve_report(run_dir: Path, report: ReleasedSolveReport) -> None:
    atomic_write_json(released_solve_report_json_path(run_dir), report.model_dump(mode="json"))


def write_released_solve_summary_table(run_dir: Path, report: ReleasedSolveReport) -> None:
    with released_solve_hits_csv_path(run_dir).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RELEASED_SOLVE_SUMMARY_FIELDNAMES)
        writer.writeheader()
        for hit in report.hits:
            target_hit = hit.target_search_hit
            writer.writerow(
                {
                    "rank": hit.rank,
                    "hit_kind": hit.hit_kind,
                    "final_geometry_source": target_hit.projection.final_geometry_source,
                    "route_family": target_hit.route_family,
                    "active_strand": target_hit.active_strand,
                    "retained_partner_strand": target_hit.projection.retained_partner_strand,
                    "physical_nicked_strand": target_hit.physical_nicked_strand,
                    "nickase_variant_id": hit.nickase_variant_id,
                    "release_variant_id": hit.release_variant_id,
                    "nick_boundary_from_left": target_hit.nick_boundary_from_left,
                    "paired_bp": target_hit.final_candidate.paired_bp,
                    "cap_nt": target_hit.final_candidate.cap_nt,
                    "active_product_input_length_nt": target_hit.active_product_input_length_nt,
                    "active_product_length_nt": target_hit.active_product_length_nt,
                    "retained_partner_length_nt": target_hit.projection.retained_partner_length_nt,
                    "precursor_length_nt": target_hit.precursor_length_nt,
                    "extra_nick_event_count": target_hit.extra_nick_event_count,
                    "extra_target_strand_nick_count": target_hit.extra_target_strand_nick_count,
                    "materialized_run_dir": hit.materialized_run_dir,
                    "render_job_path": hit.render_job_path,
                    "rendered_plot_path": hit.rendered_plot_path,
                }
            )


def build_released_solve_manifest(
    *,
    run_dir: Path,
    workspace_root: Path,
    report: ReleasedSolveReport,
) -> dict[str, Any]:
    return {
        "kind": "released_solve",
        "stage": "snapback_released",
        "workflow": "snapback_released_solve",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "status": report.status,
        "contract": report.metadata.kind,
        "request_snapshot_sha256": sha256_path(released_solve_request_snapshot_path(run_dir)),
        "nickase_catalog_sha256": sha256_path(released_nickase_catalog_snapshot_path(run_dir)),
        "release_catalog_sha256": sha256_path(released_release_catalog_snapshot_path(run_dir)),
        "nick_catalog_source": report.metadata.nick_catalog_source,
        "release_catalog_source": report.metadata.release_catalog_source,
        "selected_hit_kind": report.metadata.selected_hit_kind,
        "materialized_hit_count": report.metadata.materialized_hit_count,
        "artifacts": [
            {"name": "solve_report_json", "path": str(released_solve_report_json_path(run_dir).relative_to(run_dir))},
            {
                "name": "request_snapshot",
                "path": str(released_solve_request_snapshot_path(run_dir).relative_to(run_dir)),
            },
            {
                "name": "nickase_catalog_snapshot",
                "path": str(released_nickase_catalog_snapshot_path(run_dir).relative_to(run_dir)),
            },
            {
                "name": "release_catalog_snapshot",
                "path": str(released_release_catalog_snapshot_path(run_dir).relative_to(run_dir)),
            },
            {"name": "hits_table", "path": str(released_solve_hits_csv_path(run_dir).relative_to(run_dir))},
            {
                "name": "materialized_hits",
                "path": str(released_solve_materialized_hits_dir(run_dir).relative_to(run_dir)),
            },
        ],
    }


def write_released_solve_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = released_solve_manifest_path(run_dir)
    atomic_write_json(path, manifest)
    return path


def write_released_solve_status(run_dir: Path, *, report: ReleasedSolveReport) -> Path:
    path = released_solve_status_path(run_dir)
    payload = {
        "workflow": "snapback_released_solve",
        "stage": "snapback_released",
        "contract": report.metadata.kind,
        "status": report.status,
        "status_message": f"released-product snapback solve {report.status}",
        "run_dir": str(run_dir.resolve()),
        "issue_count": len(report.issues),
        "materialized_hit_count": report.metadata.materialized_hit_count,
        "selected_hit_kind": report.metadata.selected_hit_kind,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(path, payload)
    return path


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


def load_released_manifest(run_dir: Path) -> dict[str, Any]:
    path = released_manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Released-product snapback manifest missing: {path}")
    return _load_json_mapping(path, label="manifest")


def load_released_status(run_dir: Path) -> dict[str, Any]:
    path = released_status_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Released-product snapback status missing: {path}")
    return _load_json_mapping(path, label="status record")


def load_released_solve_manifest(run_dir: Path) -> dict[str, Any]:
    path = released_solve_manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Released-product snapback solve manifest missing: {path}")
    return _load_json_mapping(path, label="solve manifest")


def load_released_solve_status(run_dir: Path) -> dict[str, Any]:
    path = released_solve_status_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Released-product snapback solve status missing: {path}")
    return _load_json_mapping(path, label="solve status record")


__all__ = [
    "RELEASED_SUMMARY_FIELDNAMES",
    "RELEASED_SOLVE_SUMMARY_FIELDNAMES",
    "build_released_manifest",
    "build_released_run_dir",
    "build_released_solve_manifest",
    "ensure_released_run_dirs",
    "ensure_released_solve_run_dirs",
    "load_released_solve_manifest",
    "load_released_solve_status",
    "load_released_manifest",
    "load_released_status",
    "released_manifest_path",
    "released_nickase_catalog_snapshot_path",
    "released_pre_nick_site_json_path",
    "released_projection_json_path",
    "released_release_catalog_snapshot_path",
    "released_release_site_json_path",
    "released_report_json_path",
    "released_solve_hit_json_path",
    "released_solve_hit_plot_context_path",
    "released_solve_hit_plot_path",
    "released_solve_hit_run_dir",
    "released_solve_hits_csv_path",
    "released_solve_manifest_path",
    "released_solve_materialized_hits_dir",
    "released_solve_report_json_path",
    "released_solve_request_snapshot_path",
    "released_solve_status_path",
    "released_spec_snapshot_path",
    "released_status_path",
    "released_summary_csv_path",
    "snapshot_released_inputs",
    "snapshot_released_solve_inputs",
    "write_released_manifest",
    "write_released_report",
    "write_released_solve_manifest",
    "write_released_solve_report",
    "write_released_solve_status",
    "write_released_solve_summary_table",
    "write_released_status",
    "write_released_summary_table",
]
