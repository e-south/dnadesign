"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/artifacts.py

Artifact paths and persistence helpers for cassette design runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.cassette.models import CassetteEvaluationReport
from dnadesign.cruncher.cassette.solve_models import SolveReport
from dnadesign.cruncher.utils.hashing import sha256_bytes, sha256_path

RUN_META_DIR = "meta"
RUN_PROVENANCE_DIR = "provenance"
RUN_ANALYSIS_DIR = "analysis"
RUN_ANALYSIS_REPORTS_DIR = "reports"
RUN_EXPORT_DIR = "export"
RUN_VIEWS_DIR = "views"
RUN_BASERENDER_JOBS_DIR = "baserender_jobs"
RUN_RENDERS_DIR = "renders"


def design_id(*, spec_bytes: bytes, catalog_bytes: bytes) -> str:
    return sha256_bytes(spec_bytes + b"\n" + catalog_bytes)[:12]


def solve_id(*, spec_bytes: bytes, catalog_bytes: bytes) -> str:
    return sha256_bytes(spec_bytes + b"\n" + catalog_bytes)[:12]


def _scoped_run_dir(workspace_root: Path, *parts: Path | str) -> Path:
    resolved_workspace_root = workspace_root.resolve()
    candidate = resolved_workspace_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(resolved_workspace_root)
    except ValueError as exc:
        raise ValueError(
            f"Cassette run directory must stay inside workspace {resolved_workspace_root}: {candidate}"
        ) from exc
    return candidate


def build_run_dir(*, workspace_root: Path, run_root: Path, spec_name: str, cassette_design_id: str) -> Path:
    return _scoped_run_dir(workspace_root, run_root, spec_name, cassette_design_id)


def build_solve_run_dir(*, workspace_root: Path, run_root: Path, cassette_solve_id: str) -> Path:
    return _scoped_run_dir(workspace_root, run_root, cassette_solve_id)


def ensure_run_dirs(run_dir: Path) -> None:
    (run_dir / RUN_META_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_PROVENANCE_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_ANALYSIS_DIR / RUN_ANALYSIS_REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_EXPORT_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_VIEWS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_BASERENDER_JOBS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_RENDERS_DIR).mkdir(parents=True, exist_ok=True)


def ensure_solve_run_dirs(run_dir: Path) -> None:
    (run_dir / "hits").mkdir(parents=True, exist_ok=True)
    (run_dir / "specs").mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_VIEWS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_BASERENDER_JOBS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_RENDERS_DIR).mkdir(parents=True, exist_ok=True)


def cassette_manifest_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "cassette_manifest.json"


def cassette_status_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "cassette_status.json"


def spec_snapshot_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR / "spec_used.yaml"


def catalog_snapshot_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR / "nickase_catalog.yaml"


def report_json_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / RUN_ANALYSIS_REPORTS_DIR / "report.json"


def report_md_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / RUN_ANALYSIS_REPORTS_DIR / "report.md"


def views_dir(run_dir: Path) -> Path:
    return run_dir / RUN_VIEWS_DIR


def baserender_jobs_dir(run_dir: Path) -> Path:
    return run_dir / RUN_BASERENDER_JOBS_DIR


def renders_dir(run_dir: Path) -> Path:
    return run_dir / RUN_RENDERS_DIR


def linear_duplex_view_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "linear_duplex.v1.json"


def hairpin_view_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "ssdna_hairpin.v1.json"


def views_manifest_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "views_manifest.v1.json"


def linear_duplex_job_path(run_dir: Path) -> Path:
    return baserender_jobs_dir(run_dir) / "linear_duplex.job.yaml"


def hairpin_job_path(run_dir: Path) -> Path:
    return baserender_jobs_dir(run_dir) / "ssdna_hairpin.job.yaml"


def candidate_table_path(run_dir: Path) -> Path:
    return run_dir / RUN_EXPORT_DIR / "table__candidates.csv"


def solve_report_json_path(run_dir: Path) -> Path:
    return run_dir / "solve_report.json"


def solve_report_md_path(run_dir: Path) -> Path:
    return run_dir / "solve_report.md"


def solve_manifest_path(run_dir: Path) -> Path:
    return run_dir / "solve_manifest.json"


def solve_status_path(run_dir: Path) -> Path:
    return run_dir / "solve_status.json"


def solve_hits_table_path(run_dir: Path) -> Path:
    return run_dir / "table__hits.csv"


def top_hits_linear_duplex_jsonl_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "top_hits.linear_duplex.v1.jsonl"


def top_hits_hairpin_jsonl_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "top_hits.ssdna_hairpin.v1.jsonl"


def top_hits_duplex_job_path(run_dir: Path) -> Path:
    return baserender_jobs_dir(run_dir) / "top_hits_duplex.job.yaml"


def top_hits_hairpin_job_path(run_dir: Path) -> Path:
    return baserender_jobs_dir(run_dir) / "top_hits_hairpin.job.yaml"


def solve_input_spec_path(run_dir: Path) -> Path:
    return run_dir / "specs" / "input_solve_spec.yaml"


def solve_resolved_catalog_path(run_dir: Path) -> Path:
    return run_dir / "specs" / "resolved_catalog.yaml"


def solve_hit_dir(run_dir: Path, *, rank: int, hit_id: str) -> Path:
    return run_dir / "hits" / f"hit_{rank:03d}_{hit_id}"


def build_manifest(
    *,
    run_dir: Path,
    workspace_root: Path,
    spec_path: Path,
    catalog_path: Path,
    report: CassetteEvaluationReport,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = [
        {"name": "report_json", "path": str(report_json_path(run_dir).relative_to(run_dir))},
        {"name": "report_md", "path": str(report_md_path(run_dir).relative_to(run_dir))},
        {"name": "candidate_table", "path": str(candidate_table_path(run_dir).relative_to(run_dir))},
        {"name": "spec_snapshot", "path": str(spec_snapshot_path(run_dir).relative_to(run_dir))},
        {"name": "catalog_snapshot", "path": str(catalog_snapshot_path(run_dir).relative_to(run_dir))},
    ]
    if views_manifest_path(run_dir).exists():
        artifacts.extend(
            [
                {"name": "linear_duplex_view", "path": str(linear_duplex_view_path(run_dir).relative_to(run_dir))},
                {"name": "hairpin_view", "path": str(hairpin_view_path(run_dir).relative_to(run_dir))},
                {"name": "views_manifest", "path": str(views_manifest_path(run_dir).relative_to(run_dir))},
            ]
        )
    if linear_duplex_job_path(run_dir).exists():
        artifacts.append(
            {"name": "linear_duplex_job", "path": str(linear_duplex_job_path(run_dir).relative_to(run_dir))}
        )
    if hairpin_job_path(run_dir).exists():
        artifacts.append({"name": "hairpin_job", "path": str(hairpin_job_path(run_dir).relative_to(run_dir))})
    return {
        "stage": "cassette",
        "workflow": "cassette_design",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "spec_name": report.spec_name,
        "status": report.status,
        "spec_schema_version": report.metadata.spec_schema_version,
        "coordinate_semantics": report.metadata.coordinate_semantics,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256_path(spec_path),
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": sha256_path(catalog_path),
        "artifacts": artifacts,
    }


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = cassette_manifest_path(run_dir)
    atomic_write_json(path, manifest)
    return path


def write_status(run_dir: Path, *, status: str, status_message: str, report: CassetteEvaluationReport) -> Path:
    path = cassette_status_path(run_dir)
    payload = {
        "stage": "cassette",
        "status": status,
        "status_message": status_message,
        "run_dir": str(run_dir.resolve()),
        "spec_name": report.spec_name,
        "spec_schema_version": report.metadata.spec_schema_version,
        "coordinate_semantics": report.metadata.coordinate_semantics,
        "issue_count": len(report.issues),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(path, payload)
    return path


def snapshot_inputs(run_dir: Path, *, spec_path: Path, catalog_path: Path) -> None:
    shutil.copyfile(spec_path, spec_snapshot_path(run_dir))
    shutil.copyfile(catalog_path, catalog_snapshot_path(run_dir))


def write_report(run_dir: Path, report: CassetteEvaluationReport, *, markdown: str) -> None:
    atomic_write_json(report_json_path(run_dir), report.model_dump(mode="json"))
    report_md_path(run_dir).write_text(markdown, encoding="utf-8")


def write_candidate_table(run_dir: Path, report: CassetteEvaluationReport) -> None:
    path = candidate_table_path(run_dir)
    fieldnames = [
        "spec_name",
        "status",
        "cassette_sequence",
        "cassette_length_nt",
        "target_strand",
        "intended_left_variant",
        "intended_left_boundary",
        "intended_right_variant",
        "intended_right_boundary",
        "bounded_nicked_segment_start",
        "bounded_nicked_segment_end",
        "bounded_nicked_segment_length",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if report.candidate is None:
            return
        writer.writerow(
            {
                "spec_name": report.spec_name,
                "status": report.status,
                "cassette_sequence": report.candidate.cassette_sequence,
                "cassette_length_nt": report.candidate.cassette_length_nt,
                "target_strand": report.target_strand,
                "intended_left_variant": report.candidate.intended_left_nick.variant_id,
                "intended_left_boundary": report.candidate.intended_left_nick.boundary,
                "intended_right_variant": report.candidate.intended_right_nick.variant_id,
                "intended_right_boundary": report.candidate.intended_right_nick.boundary,
                "bounded_nicked_segment_start": report.candidate.bounded_nicked_segment.start_boundary,
                "bounded_nicked_segment_end": report.candidate.bounded_nicked_segment.end_boundary,
                "bounded_nicked_segment_length": report.candidate.bounded_nicked_segment.length_nt,
            }
        )


def write_view_bundle(
    run_dir: Path,
    *,
    linear_duplex: dict[str, Any],
    hairpin: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    atomic_write_json(linear_duplex_view_path(run_dir), linear_duplex)
    atomic_write_json(hairpin_view_path(run_dir), hairpin)
    atomic_write_json(views_manifest_path(run_dir), manifest)


def write_jsonl_records(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else "")
    path.write_text(payload, encoding="utf-8")
    return path


def write_baserender_job(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = cassette_manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing cassette manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_status(run_dir: Path) -> dict[str, Any]:
    path = cassette_status_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing cassette status: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_solve_inputs(run_dir: Path, *, spec_path: Path | None, resolved_catalog_yaml: str | None = None) -> None:
    if spec_path is not None and spec_path.exists():
        shutil.copyfile(spec_path, solve_input_spec_path(run_dir))
    if resolved_catalog_yaml is not None:
        solve_resolved_catalog_path(run_dir).write_text(resolved_catalog_yaml, encoding="utf-8")


def write_solve_report(run_dir: Path, report: SolveReport, *, markdown: str) -> None:
    atomic_write_json(solve_report_json_path(run_dir), report.model_dump(mode="json"))
    solve_report_md_path(run_dir).write_text(markdown, encoding="utf-8")


def write_solve_hits_table(run_dir: Path, report: SolveReport) -> None:
    fieldnames = [
        "rank",
        "solution_id",
        "score",
        "score_tuple",
        "base_penalty_vector",
        "hit_id",
        "cassette_sequence",
        "stem5p_arm",
        "loop",
        "left_variant_id",
        "right_variant_id",
        "left_nick_boundary",
        "right_nick_boundary",
        "target_strand",
        "bounded_segment_length",
        "extra_site_count",
        "gc_fraction",
        "selection_policy",
        "selection_rank_reason",
        "distance_to_previous_selected",
        "explicit_design_id",
        "views_manifest_path",
        "linear_duplex_job_path",
        "ssdna_hairpin_job_path",
    ]
    with solve_hits_table_path(run_dir).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for hit in report.hits:
            writer.writerow(
                {
                    "rank": hit.rank,
                    "solution_id": hit.solution_id,
                    "score": json.dumps(hit.score),
                    "score_tuple": json.dumps(hit.score),
                    "base_penalty_vector": json.dumps(hit.base_penalty_vector),
                    "hit_id": hit.hit_id,
                    "cassette_sequence": hit.cassette_sequence,
                    "stem5p_arm": hit.stem5p_arm,
                    "loop": hit.loop,
                    "left_variant_id": hit.left_variant_id,
                    "right_variant_id": hit.right_variant_id,
                    "left_nick_boundary": hit.left_nick_boundary,
                    "right_nick_boundary": hit.right_nick_boundary,
                    "target_strand": hit.target_strand,
                    "bounded_segment_length": hit.bounded_segment_length,
                    "extra_site_count": hit.extra_site_count,
                    "gc_fraction": hit.gc_fraction,
                    "selection_policy": (
                        report.selection_summary.policy if report.selection_summary is not None else None
                    ),
                    "selection_rank_reason": hit.selection_rank_reason,
                    "distance_to_previous_selected": hit.distance_to_previous_selected,
                    "explicit_design_id": hit.explicit_design_id,
                    "views_manifest_path": hit.views_manifest_path,
                    "linear_duplex_job_path": hit.linear_duplex_job_path,
                    "ssdna_hairpin_job_path": hit.ssdna_hairpin_job_path,
                }
            )


def build_solve_manifest(
    *,
    run_dir: Path,
    workspace_root: Path,
    spec_path: Path,
    report: SolveReport,
) -> dict[str, Any]:
    artifacts = [
        {"name": "solve_report_json", "path": "solve_report.json"},
        {"name": "solve_report_md", "path": "solve_report.md"},
        {"name": "solve_manifest", "path": "solve_manifest.json"},
        {"name": "solve_status", "path": "solve_status.json"},
    ]
    if solve_hits_table_path(run_dir).exists():
        artifacts.append({"name": "hit_table", "path": "table__hits.csv"})
    if solve_input_spec_path(run_dir).exists():
        artifacts.append({"name": "input_spec", "path": "specs/input_solve_spec.yaml"})
    if solve_resolved_catalog_path(run_dir).exists():
        artifacts.append({"name": "resolved_catalog", "path": "specs/resolved_catalog.yaml"})
    if top_hits_linear_duplex_jsonl_path(run_dir).exists():
        artifacts.append({"name": "top_hits_linear_duplex_jsonl", "path": "views/top_hits.linear_duplex.v1.jsonl"})
    if top_hits_hairpin_jsonl_path(run_dir).exists():
        artifacts.append({"name": "top_hits_hairpin_jsonl", "path": "views/top_hits.ssdna_hairpin.v1.jsonl"})
    if top_hits_duplex_job_path(run_dir).exists():
        artifacts.append({"name": "top_hits_duplex_job", "path": "baserender_jobs/top_hits_duplex.job.yaml"})
    if top_hits_hairpin_job_path(run_dir).exists():
        artifacts.append({"name": "top_hits_hairpin_job", "path": "baserender_jobs/top_hits_hairpin.job.yaml"})
    return {
        "stage": "cassette_solve",
        "workflow": "cassette_solve",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256_path(spec_path),
        "resolved_catalog_path": (
            str(solve_resolved_catalog_path(run_dir).resolve())
            if solve_resolved_catalog_path(run_dir).exists()
            else None
        ),
        "resolved_catalog_sha256": (
            sha256_path(solve_resolved_catalog_path(run_dir)) if solve_resolved_catalog_path(run_dir).exists() else None
        ),
        "catalog_preset": report.metadata.catalog_preset,
        "catalog_additional_paths": list(report.metadata.catalog_additional_paths),
        "status": report.status,
        "artifacts": artifacts,
    }


def write_solve_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = solve_manifest_path(run_dir)
    atomic_write_json(path, manifest)
    return path


def write_solve_status(run_dir: Path, *, report: SolveReport, status_message: str) -> Path:
    path = solve_status_path(run_dir)
    warnings = list(report.metadata.warnings)
    warning_codes = list(report.metadata.warning_codes)
    selection_summary = report.selection_summary
    payload = {
        "stage": "cassette_solve",
        "status": report.status,
        "status_message": status_message,
        "run_dir": str(run_dir.resolve()),
        "solve_id": report.solve_id,
        "hit_count": len(report.hits),
        "issue_count": len(report.issues),
        "warning_count": len(warnings),
        "warnings": warnings,
        "warning_codes": warning_codes,
        "search_truncated": any(
            code in {"MAX_SEARCH_NODES_REACHED", "MAX_ENUMERATED_CANDIDATES_REACHED"} for code in warning_codes
        ),
        "search_bounded": any(
            code in {"MAX_SEARCH_NODES_REACHED", "MAX_ENUMERATED_CANDIDATES_REACHED"} for code in warning_codes
        ),
        "accepted_pool_truncated": (
            selection_summary.accepted_pool_truncated if selection_summary is not None else False
        ),
        "pool_bounded": selection_summary.accepted_pool_truncated if selection_summary is not None else False,
        "selection": (
            {
                "policy": selection_summary.policy,
                "distance_metric": selection_summary.distance_metric,
                "diversity_weight": selection_summary.diversity_weight,
                "pool_size": selection_summary.pool_size,
                "accepted_candidate_count": selection_summary.accepted_candidate_count,
                "accepted_pool_size": selection_summary.accepted_pool_size,
                "accepted_pool_admitted_count": selection_summary.accepted_pool_admitted_count,
                "accepted_pool_rejected_count": selection_summary.accepted_pool_rejected_count,
                "accepted_pool_truncated": selection_summary.accepted_pool_truncated,
                "accepted_pool_worst_score_at_close": selection_summary.accepted_pool_worst_score_at_close,
                "selected_hit_count": selection_summary.selected_hit_count,
                "selected_hit_ids": selection_summary.selected_hit_ids,
                "selection_policy_defaulted": selection_summary.selection_policy_defaulted,
                "selection_pool_non_exhaustive_reason": selection_summary.selection_pool_non_exhaustive_reason,
                "policy_limited_hit_count": selection_summary.policy_limited_hit_count,
                "policy_underfilled": selection_summary.policy_underfilled,
                "policy_underfilled_reason": selection_summary.policy_underfilled_reason,
                "pairwise_distance_summary": selection_summary.pairwise_distance_summary.model_dump(mode="json"),
            }
            if selection_summary is not None
            else None
        ),
        "top_hit_batch_scope": "selected_hits",
        "top_hits_linear_duplex_jsonl": (
            str(top_hits_linear_duplex_jsonl_path(run_dir).resolve())
            if top_hits_linear_duplex_jsonl_path(run_dir).exists()
            else None
        ),
        "top_hits_hairpin_jsonl": (
            str(top_hits_hairpin_jsonl_path(run_dir).resolve())
            if top_hits_hairpin_jsonl_path(run_dir).exists()
            else None
        ),
        "top_hits_duplex_job": (
            str(top_hits_duplex_job_path(run_dir).resolve()) if top_hits_duplex_job_path(run_dir).exists() else None
        ),
        "top_hits_hairpin_job": (
            str(top_hits_hairpin_job_path(run_dir).resolve()) if top_hits_hairpin_job_path(run_dir).exists() else None
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(path, payload)
    return path


def write_solve_hit_bundle(
    *,
    hit_dir: Path,
    resolved_spec_payload: dict[str, Any],
    report: CassetteEvaluationReport,
    markdown: str,
) -> None:
    hit_dir.mkdir(parents=True, exist_ok=True)
    (hit_dir / "resolved_candidate.cassette.yaml").write_text(
        yaml.safe_dump(resolved_spec_payload, sort_keys=False),
        encoding="utf-8",
    )
    atomic_write_json(hit_dir / "report.json", report.model_dump(mode="json"))
    (hit_dir / "report.md").write_text(markdown, encoding="utf-8")
    atomic_write_json(
        hit_dir / "manifest.json",
        {
            "status": report.status,
            "spec_name": report.spec_name,
            "report_json": "report.json",
            "report_md": "report.md",
            "resolved_candidate_spec": "resolved_candidate.cassette.yaml",
        },
    )
    atomic_write_json(
        hit_dir / "status.json",
        {
            "status": report.status,
            "issue_count": len(report.issues),
            "spec_name": report.spec_name,
            "run_dir": str(hit_dir.resolve()),
        },
    )
