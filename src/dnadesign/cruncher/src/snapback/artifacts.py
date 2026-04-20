"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/artifacts.py

Artifact paths and persistence helpers for explicit and solve snapback runs.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.snapback.models import SnapbackEvaluationReport
from dnadesign.cruncher.snapback.solve_models import SnapbackSolveReport
from dnadesign.cruncher.utils.hashing import sha256_bytes, sha256_path

RUN_META_DIR = "meta"
RUN_PROVENANCE_DIR = "provenance"
RUN_ANALYSIS_DIR = "analysis"
RUN_ANALYSIS_REPORTS_DIR = "reports"
RUN_EXPORT_DIR = "export"
RUN_VIEWS_DIR = "views"
RUN_BASERENDER_JOBS_DIR = "baserender_jobs"
RUN_RENDERS_DIR = "renders"
SOLVE_HITS_DIR = "hits"
SOLVE_SPECS_DIR = "specs"


def design_id(*, spec_bytes: bytes, catalog_bytes: bytes) -> str:
    return sha256_bytes(spec_bytes + b"\n" + catalog_bytes)[:12]


def solve_id(*, spec_bytes: bytes, catalog_bytes: bytes) -> str:
    return sha256_bytes(b"solve\n" + spec_bytes + b"\n" + catalog_bytes)[:12]


def _scoped_run_dir(workspace_root: Path, *parts: Path | str) -> Path:
    resolved_workspace_root = workspace_root.resolve()
    candidate = resolved_workspace_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(resolved_workspace_root)
    except ValueError as exc:
        raise ValueError(
            f"Snapback run directory must stay inside workspace {resolved_workspace_root}: {candidate}"
        ) from exc
    return candidate


def build_run_dir(*, workspace_root: Path, run_root: Path, spec_name: str, snapback_design_id: str) -> Path:
    return _scoped_run_dir(workspace_root, run_root, spec_name, snapback_design_id)


def build_solve_run_dir(*, workspace_root: Path, run_root: Path, snapback_solve_id: str) -> Path:
    return _scoped_run_dir(workspace_root, run_root, snapback_solve_id)


def solve_hit_run_dir(run_dir: Path, *, rank: int, explicit_design_id: str) -> Path:
    return run_dir / SOLVE_HITS_DIR / f"{rank:02d}__{explicit_design_id}"


def ensure_run_dirs(run_dir: Path) -> None:
    (run_dir / RUN_META_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_PROVENANCE_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_ANALYSIS_DIR / RUN_ANALYSIS_REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_EXPORT_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_VIEWS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_BASERENDER_JOBS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_RENDERS_DIR).mkdir(parents=True, exist_ok=True)


def ensure_solve_run_dirs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SOLVE_HITS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / SOLVE_SPECS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_BASERENDER_JOBS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_RENDERS_DIR).mkdir(parents=True, exist_ok=True)


def snapback_manifest_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "snapback_manifest.json"


def snapback_status_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "snapback_status.json"


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


def pre_nick_duplex_view_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "pre_nick_duplex.v1.json"


def post_nick_exposed_view_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "post_nick_exposed.v1.json"


def post_nick_foldback_view_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "post_nick_foldback.v1.json"


def pre_nick_duplex_visual_contract_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "pre_nick_duplex.snapback_visual.v1.json"


def post_nick_exposed_visual_contract_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "post_nick_exposed.snapback_visual.v1.json"


def post_nick_foldback_visual_contract_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "post_nick_foldback.snapback_visual.v1.json"


def views_manifest_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "views_manifest.v1.json"


def pre_nick_duplex_job_path(run_dir: Path) -> Path:
    return baserender_jobs_dir(run_dir) / "pre_nick_duplex.job.yaml"


def post_nick_exposed_job_path(run_dir: Path) -> Path:
    return baserender_jobs_dir(run_dir) / "post_nick_exposed.job.yaml"


def post_nick_foldback_job_path(run_dir: Path) -> Path:
    return baserender_jobs_dir(run_dir) / "post_nick_foldback.job.yaml"


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


def solve_input_spec_path(run_dir: Path) -> Path:
    return run_dir / SOLVE_SPECS_DIR / "input_solve_spec.yaml"


def solve_resolved_catalog_path(run_dir: Path) -> Path:
    return run_dir / SOLVE_SPECS_DIR / "resolved_catalog.yaml"


def build_manifest(
    *,
    run_dir: Path,
    workspace_root: Path,
    spec_path: Path,
    report: SnapbackEvaluationReport,
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
                {"name": "pre_nick_duplex_view", "path": str(pre_nick_duplex_view_path(run_dir).relative_to(run_dir))},
                {
                    "name": "post_nick_exposed_view",
                    "path": str(post_nick_exposed_view_path(run_dir).relative_to(run_dir)),
                },
                {
                    "name": "post_nick_foldback_view",
                    "path": str(post_nick_foldback_view_path(run_dir).relative_to(run_dir)),
                },
                {"name": "views_manifest", "path": str(views_manifest_path(run_dir).relative_to(run_dir))},
            ]
        )
    if pre_nick_duplex_visual_contract_path(run_dir).exists():
        artifacts.append(
            {
                "name": "pre_nick_duplex_visual_contract",
                "path": str(pre_nick_duplex_visual_contract_path(run_dir).relative_to(run_dir)),
            }
        )
    if post_nick_exposed_visual_contract_path(run_dir).exists():
        artifacts.append(
            {
                "name": "post_nick_exposed_visual_contract",
                "path": str(post_nick_exposed_visual_contract_path(run_dir).relative_to(run_dir)),
            }
        )
    if post_nick_foldback_visual_contract_path(run_dir).exists():
        artifacts.append(
            {
                "name": "post_nick_foldback_visual_contract",
                "path": str(post_nick_foldback_visual_contract_path(run_dir).relative_to(run_dir)),
            }
        )
    for name, path in (
        ("pre_nick_duplex_job", pre_nick_duplex_job_path(run_dir)),
        ("post_nick_exposed_job", post_nick_exposed_job_path(run_dir)),
        ("post_nick_foldback_job", post_nick_foldback_job_path(run_dir)),
    ):
        if path.exists():
            artifacts.append({"name": name, "path": str(path.relative_to(run_dir))})
    return {
        "kind": "explicit",
        "stage": "snapback",
        "workflow": "snapback_design",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "spec_name": report.spec_name,
        "status": report.status,
        "contract": report.metadata.contract,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256_path(spec_path),
        "catalog_source": report.catalog_source,
        "artifacts": artifacts,
    }


def build_solve_manifest(
    *,
    run_dir: Path,
    workspace_root: Path,
    spec_path: Path,
    report: SnapbackSolveReport,
) -> dict[str, Any]:
    return {
        "kind": "solve",
        "stage": "snapback",
        "workflow": "snapback_solve",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "spec_name": report.spec_name,
        "status": report.status,
        "contract": report.metadata.contract,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256_path(spec_path),
        "artifacts": [
            {"name": "solve_report_json", "path": "solve_report.json"},
            {"name": "solve_report_md", "path": "solve_report.md"},
            {"name": "solve_hits_table", "path": "table__hits.csv"},
            {"name": "solve_status", "path": "solve_status.json"},
            {"name": "input_solve_spec", "path": "specs/input_solve_spec.yaml"},
            {"name": "resolved_catalog", "path": "specs/resolved_catalog.yaml"},
            {"name": "hits", "path": "hits"},
        ],
    }


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = snapback_manifest_path(run_dir)
    atomic_write_json(path, manifest)
    return path


def write_solve_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = solve_manifest_path(run_dir)
    atomic_write_json(path, manifest)
    return path


def write_status(run_dir: Path, *, report: SnapbackEvaluationReport) -> Path:
    path = snapback_status_path(run_dir)
    payload = {
        "workflow": "snapback_design",
        "stage": "snapback",
        "contract": report.metadata.contract,
        "status": report.status,
        "status_message": f"snapback design {report.status}",
        "run_dir": str(run_dir.resolve()),
        "spec_name": report.spec_name,
        "issue_count": len(report.issues),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(path, payload)
    return path


def write_solve_status(run_dir: Path, *, report: SnapbackSolveReport, status_message: str) -> Path:
    path = solve_status_path(run_dir)
    payload = {
        "workflow": "snapback_solve",
        "stage": "snapback",
        "contract": report.metadata.contract,
        "status": report.status,
        "status_message": status_message,
        "run_dir": str(run_dir.resolve()),
        "spec_name": report.spec_name,
        "hit_count": len(report.hits),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(path, payload)
    return path


def snapshot_explicit_inputs(run_dir: Path, *, spec_path: Path, catalog_yaml: str) -> None:
    spec_snapshot_path(run_dir).write_text(spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    catalog_snapshot_path(run_dir).write_text(catalog_yaml, encoding="utf-8")


def write_solve_inputs(run_dir: Path, *, spec_path: Path, catalog_yaml: str) -> None:
    solve_input_spec_path(run_dir).write_text(spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    solve_resolved_catalog_path(run_dir).write_text(catalog_yaml, encoding="utf-8")


def write_report(run_dir: Path, report: SnapbackEvaluationReport, *, markdown: str) -> None:
    atomic_write_json(report_json_path(run_dir), report.model_dump(mode="json"))
    report_md_path(run_dir).write_text(markdown, encoding="utf-8")


def write_solve_report(run_dir: Path, report: SnapbackSolveReport, *, markdown: str) -> None:
    atomic_write_json(solve_report_json_path(run_dir), report.model_dump(mode="json"))
    solve_report_md_path(run_dir).write_text(markdown, encoding="utf-8")


def write_candidate_table(run_dir: Path, report: SnapbackEvaluationReport) -> None:
    fieldnames = [
        "status",
        "spec_name",
        "variant_id",
        "nick_boundary",
        "nick_boundary_from_left",
        "released_prefix_nt",
        "retained_start_from_nick",
        "cap_nt",
        "paired_bp",
        "mismatch_count",
        "terminal_ligatable_duplex_bp",
        "max_uninterrupted_duplex_bp",
        "added_nt",
        "extra_nick_event_count",
        "gc_fraction_added",
        "designed_sequence",
    ]
    with candidate_table_path(run_dir).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        candidate = report.candidate
        if candidate is None:
            return
        writer.writerow(
            {
                "status": report.status,
                "spec_name": report.spec_name,
                "variant_id": candidate.intended_nick.variant_id,
                "nick_boundary": candidate.nick_boundary,
                "nick_boundary_from_left": candidate.nick_boundary_from_left,
                "released_prefix_nt": candidate.released_prefix_nt,
                "retained_start_from_nick": candidate.retained_start_from_nick,
                "cap_nt": candidate.cap_nt,
                "paired_bp": candidate.paired_bp,
                "mismatch_count": candidate.mismatch_count,
                "terminal_ligatable_duplex_bp": candidate.terminal_ligatable_duplex_bp,
                "max_uninterrupted_duplex_bp": candidate.max_uninterrupted_duplex_bp,
                "added_nt": candidate.added_nt,
                "extra_nick_event_count": candidate.extra_nick_event_count,
                "gc_fraction_added": candidate.gc_fraction_added,
                "designed_sequence": candidate.designed_sequence,
            }
        )


def write_solve_hits_table(run_dir: Path, report: SnapbackSolveReport) -> None:
    fieldnames = [
        "rank",
        "hit_id",
        "variant_id",
        "nick_boundary",
        "nick_boundary_from_left",
        "retained_start_from_nick",
        "cap_sequence",
        "foldback_arm",
        "added_nt",
        "paired_bp",
        "mismatch_count",
        "terminal_ligatable_duplex_bp",
        "max_uninterrupted_duplex_bp",
        "extra_nick_event_count",
        "gc_fraction_added",
        "materialized_run_dir",
    ]
    with solve_hits_table_path(run_dir).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for hit in report.hits:
            writer.writerow(
                {
                    "rank": hit.rank,
                    "hit_id": hit.hit_id,
                    "variant_id": hit.variant_id,
                    "nick_boundary": hit.nick_boundary,
                    "nick_boundary_from_left": hit.nick_boundary_from_left,
                    "retained_start_from_nick": hit.retained_start_from_nick,
                    "cap_sequence": hit.cap_sequence,
                    "foldback_arm": hit.foldback_arm,
                    "added_nt": hit.added_nt,
                    "paired_bp": hit.paired_bp,
                    "mismatch_count": hit.mismatch_count,
                    "terminal_ligatable_duplex_bp": hit.terminal_ligatable_duplex_bp,
                    "max_uninterrupted_duplex_bp": hit.max_uninterrupted_duplex_bp,
                    "extra_nick_event_count": hit.extra_nick_event_count,
                    "gc_fraction_added": hit.gc_fraction_added,
                    "materialized_run_dir": hit.materialized_run_dir,
                }
            )


def write_view_bundle(
    run_dir: Path,
    *,
    pre_nick_duplex: dict[str, Any],
    post_nick_exposed: dict[str, Any],
    post_nick_foldback: dict[str, Any],
    pre_nick_duplex_visual_contract: dict[str, Any],
    post_nick_exposed_visual_contract: dict[str, Any],
    post_nick_foldback_visual_contract: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    atomic_write_json(pre_nick_duplex_view_path(run_dir), pre_nick_duplex)
    atomic_write_json(post_nick_exposed_view_path(run_dir), post_nick_exposed)
    atomic_write_json(post_nick_foldback_view_path(run_dir), post_nick_foldback)
    atomic_write_json(pre_nick_duplex_visual_contract_path(run_dir), pre_nick_duplex_visual_contract)
    atomic_write_json(post_nick_exposed_visual_contract_path(run_dir), post_nick_exposed_visual_contract)
    atomic_write_json(post_nick_foldback_visual_contract_path(run_dir), post_nick_foldback_visual_contract)
    atomic_write_json(views_manifest_path(run_dir), manifest)


def write_baserender_job(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = snapback_manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Snapback manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_status(run_dir: Path) -> dict[str, Any]:
    path = snapback_status_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Snapback status missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_solve_manifest(run_dir: Path) -> dict[str, Any]:
    path = solve_manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Snapback solve manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_solve_status(run_dir: Path) -> dict[str, Any]:
    path = solve_status_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Snapback solve status missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
