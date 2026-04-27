"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/artifacts.py

Artifact paths and persistence helpers for explicit and solve snapback runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_text, atomic_write_yaml
from dnadesign.cruncher.snapback.models import SnapbackEvaluationReport
from dnadesign.cruncher.snapback.solve_models import SnapbackSolveReport
from dnadesign.cruncher.utils.hashing import sha256_bytes, sha256_path

RUN_META_DIR = "meta"
RUN_PROVENANCE_DIR = "provenance"
RUN_ANALYSIS_DIR = "analysis"
RUN_ANALYSIS_REPORTS_DIR = "reports"
RUN_ANALYSIS_VIEWS_DIR = "views"
RUN_EXPORT_DIR = "export"
RUN_PLOTS_DIR = "plots"
RUN_BASERENDER_JOBS_DIR = "baserender_jobs"
SOLVE_HITS_DIR = "materialized_hits"


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


def display_workspace_relative(value: str | Path, *, workspace_root: str | Path) -> str:
    root = Path(workspace_root).expanduser().resolve()
    rendered: list[str] = []
    for part in str(value).split(", "):
        candidate = Path(part).expanduser()
        if not candidate.is_absolute():
            rendered.append(part)
            continue
        try:
            rendered.append(str(candidate.resolve().relative_to(root)))
        except ValueError:
            rendered.append(part)
    return ", ".join(rendered)


def build_run_dir(*, workspace_root: Path, run_root: Path, spec_name: str, snapback_design_id: str) -> Path:
    del spec_name, snapback_design_id
    return _scoped_run_dir(workspace_root, run_root)


def build_solve_run_dir(*, workspace_root: Path, run_root: Path, snapback_solve_id: str) -> Path:
    del snapback_solve_id
    return _scoped_run_dir(workspace_root, run_root)


def solve_hit_run_dir(run_dir: Path, *, rank: int) -> Path:
    return materialized_hits_dir(run_dir) / f"hit_{rank:02d}"


def ensure_run_dirs(
    run_dir: Path,
    *,
    include_visual_contracts: bool = False,
    include_baserender_jobs: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_META_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_PROVENANCE_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_ANALYSIS_DIR / RUN_ANALYSIS_REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_EXPORT_DIR).mkdir(parents=True, exist_ok=True)
    if include_visual_contracts:
        views_dir(run_dir).mkdir(parents=True, exist_ok=True)
    if include_baserender_jobs:
        baserender_jobs_dir(run_dir).mkdir(parents=True, exist_ok=True)


def ensure_solve_run_dirs(run_dir: Path) -> None:
    ensure_run_dirs(run_dir)
    materialized_hits_dir(run_dir).mkdir(parents=True, exist_ok=True)


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


def analysis_dir(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR


def views_dir(run_dir: Path) -> Path:
    return analysis_dir(run_dir) / RUN_ANALYSIS_VIEWS_DIR


def materialized_hits_dir(run_dir: Path) -> Path:
    return analysis_dir(run_dir) / SOLVE_HITS_DIR


def baserender_jobs_dir(run_dir: Path) -> Path:
    return run_dir / RUN_BASERENDER_JOBS_DIR


def renders_dir(run_dir: Path) -> Path:
    return run_dir / RUN_PLOTS_DIR


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


def snapback_triptych_visual_contracts_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "snapback_triptych.snapback_visual.v1.jsonl"


def snapback_triptych_job_path(run_dir: Path) -> Path:
    return baserender_jobs_dir(run_dir) / "snapback_triptych.job.yaml"


def snapback_triptych_render_path(run_dir: Path, *, fmt: str) -> Path:
    return renders_dir(run_dir) / f"snapback_triptych.{fmt}"


def candidate_table_path(run_dir: Path) -> Path:
    return run_dir / RUN_EXPORT_DIR / "table__candidates.csv"


def solve_report_json_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / RUN_ANALYSIS_REPORTS_DIR / "solve_report.json"


def solve_report_md_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / RUN_ANALYSIS_REPORTS_DIR / "solve_report.md"


def solve_manifest_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "solve_manifest.json"


def solve_status_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "solve_status.json"


def solve_hits_table_path(run_dir: Path) -> Path:
    return run_dir / RUN_EXPORT_DIR / "table__hits.csv"


def solve_frontier_table_path(run_dir: Path) -> Path:
    return run_dir / RUN_EXPORT_DIR / "table__frontier.csv"


def solve_input_spec_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR / "input_solve_spec.yaml"


def solve_resolved_catalog_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR / "resolved_catalog.yaml"


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
    if snapback_triptych_visual_contracts_path(run_dir).exists():
        artifacts.append(
            {
                "name": "snapback_triptych_visual_contracts",
                "path": str(snapback_triptych_visual_contracts_path(run_dir).relative_to(run_dir)),
            }
        )
    if snapback_triptych_job_path(run_dir).exists():
        artifacts.append(
            {
                "name": "snapback_triptych_job",
                "path": str(snapback_triptych_job_path(run_dir).relative_to(run_dir)),
            }
        )
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
            {"name": "solve_report_json", "path": str(solve_report_json_path(run_dir).relative_to(run_dir))},
            {"name": "solve_report_md", "path": str(solve_report_md_path(run_dir).relative_to(run_dir))},
            {"name": "solve_hits_table", "path": str(solve_hits_table_path(run_dir).relative_to(run_dir))},
            {"name": "solve_frontier_table", "path": str(solve_frontier_table_path(run_dir).relative_to(run_dir))},
            {"name": "solve_status", "path": str(solve_status_path(run_dir).relative_to(run_dir))},
            {"name": "input_solve_spec", "path": str(solve_input_spec_path(run_dir).relative_to(run_dir))},
            {"name": "resolved_catalog", "path": str(solve_resolved_catalog_path(run_dir).relative_to(run_dir))},
            {"name": "materialized_hits", "path": str(materialized_hits_dir(run_dir).relative_to(run_dir))},
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
    atomic_write_text(spec_snapshot_path(run_dir), spec_path.read_text(encoding="utf-8"))
    atomic_write_text(catalog_snapshot_path(run_dir), catalog_yaml)


def write_materialized_explicit_inputs(
    run_dir: Path,
    *,
    spec_payload: dict[str, object],
    catalog_yaml: str,
) -> None:
    atomic_write_yaml(spec_snapshot_path(run_dir), spec_payload, sort_keys=False)
    atomic_write_text(catalog_snapshot_path(run_dir), catalog_yaml)


def write_solve_inputs(run_dir: Path, *, spec_path: Path, catalog_yaml: str) -> None:
    atomic_write_text(solve_input_spec_path(run_dir), spec_path.read_text(encoding="utf-8"))
    atomic_write_text(solve_resolved_catalog_path(run_dir), catalog_yaml)


def write_report(run_dir: Path, report: SnapbackEvaluationReport, *, markdown: str) -> None:
    atomic_write_json(report_json_path(run_dir), report.model_dump(mode="json"))
    report_md_path(run_dir).write_text(markdown, encoding="utf-8")


def write_solve_report(run_dir: Path, report: SnapbackSolveReport, *, markdown: str) -> None:
    atomic_write_json(solve_report_json_path(run_dir), report.model_dump(mode="json"))
    solve_report_md_path(run_dir).write_text(markdown, encoding="utf-8")


def write_candidate_table(run_dir: Path, report: SnapbackEvaluationReport) -> None:
    catalog_entry = report.metadata.catalog_variants[0] if report.metadata.catalog_variants else None
    fieldnames = [
        "status",
        "spec_name",
        "variant_id",
        "nicked_strand",
        "active_cut_offset",
        "outside_site",
        "snapback_tier",
        "vendor",
        "intended_site_sequence",
        "nick_boundary",
        "nick_boundary_from_left",
        "site_mutation_count",
        "released_prefix_nt",
        "retained_start_from_nick",
        "cap_nt",
        "cap_extension_nt",
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
                "nicked_strand": catalog_entry.nicked_strand if catalog_entry is not None else None,
                "active_cut_offset": catalog_entry.active_cut_offset if catalog_entry is not None else None,
                "outside_site": (
                    catalog_entry.selection.outside_site
                    if catalog_entry is not None and catalog_entry.selection is not None
                    else None
                ),
                "snapback_tier": (
                    catalog_entry.selection.snapback_tier
                    if catalog_entry is not None and catalog_entry.selection is not None
                    else None
                ),
                "vendor": (catalog_entry.vendor or catalog_entry.source if catalog_entry is not None else None),
                "intended_site_sequence": candidate.intended_site.matched_span_sequence,
                "nick_boundary": candidate.nick_boundary,
                "nick_boundary_from_left": candidate.nick_boundary_from_left,
                "site_mutation_count": candidate.site_mutation_count,
                "released_prefix_nt": candidate.released_prefix_nt,
                "retained_start_from_nick": candidate.retained_start_from_nick,
                "cap_nt": candidate.cap_nt,
                "cap_extension_nt": candidate.cap_extension_nt,
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
        "nicked_strand",
        "active_cut_offset",
        "outside_site",
        "snapback_tier",
        "commercial_confidence",
        "vendor",
        "warning_codes",
        "intended_site_orientation",
        "intended_site_sequence",
        "nick_boundary",
        "nick_boundary_from_left",
        "site_mutation_count",
        "retained_start_from_nick",
        "cap_nt",
        "cap_extension_nt",
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
                    "nicked_strand": hit.nickase.nicked_strand,
                    "active_cut_offset": hit.nickase.active_cut_offset,
                    "outside_site": hit.nickase.selection.outside_site if hit.nickase.selection is not None else None,
                    "snapback_tier": hit.nickase.selection.snapback_tier if hit.nickase.selection is not None else None,
                    "commercial_confidence": (
                        hit.nickase.selection.commercial_confidence if hit.nickase.selection is not None else None
                    ),
                    "vendor": hit.nickase.vendor or hit.nickase.source,
                    "warning_codes": (
                        ",".join(hit.nickase.selection.warning_codes) if hit.nickase.selection is not None else ""
                    ),
                    "intended_site_orientation": hit.intended_site_orientation,
                    "intended_site_sequence": hit.intended_site_sequence,
                    "nick_boundary": hit.nick_boundary,
                    "nick_boundary_from_left": hit.nick_boundary_from_left,
                    "site_mutation_count": hit.site_mutation_count,
                    "retained_start_from_nick": hit.retained_start_from_nick,
                    "cap_nt": hit.cap_nt,
                    "cap_extension_nt": hit.cap_extension_nt,
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


def write_solve_frontier_table(run_dir: Path, report: SnapbackSolveReport) -> None:
    fieldnames = [
        "nick_boundary_from_left",
        "paired_bp",
        "cap_extension_nt",
        "codesigned_input_count",
        "enumerated_candidate_count",
        "accepted_candidate_count",
    ]
    with solve_frontier_table_path(run_dir).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.frontier:
            writer.writerow(row.model_dump(mode="json"))


def write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> Path:
    payload = "\n".join(json.dumps(record) for record in records)
    if payload:
        payload += "\n"
    atomic_write_text(path, payload)
    return path


def write_view_bundle(
    run_dir: Path,
    *,
    pre_nick_duplex: dict[str, Any],
    post_nick_exposed: dict[str, Any],
    post_nick_foldback: dict[str, Any],
    pre_nick_duplex_visual_contract: dict[str, Any],
    post_nick_exposed_visual_contract: dict[str, Any],
    post_nick_foldback_visual_contract: dict[str, Any],
    triptych_visual_contracts: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    atomic_write_json(pre_nick_duplex_view_path(run_dir), pre_nick_duplex)
    atomic_write_json(post_nick_exposed_view_path(run_dir), post_nick_exposed)
    atomic_write_json(post_nick_foldback_view_path(run_dir), post_nick_foldback)
    atomic_write_json(pre_nick_duplex_visual_contract_path(run_dir), pre_nick_duplex_visual_contract)
    atomic_write_json(post_nick_exposed_visual_contract_path(run_dir), post_nick_exposed_visual_contract)
    atomic_write_json(post_nick_foldback_visual_contract_path(run_dir), post_nick_foldback_visual_contract)
    write_jsonl_records(snapback_triptych_visual_contracts_path(run_dir), triptych_visual_contracts)
    atomic_write_json(views_manifest_path(run_dir), manifest)


def write_baserender_job(path: Path, payload: dict[str, Any]) -> Path:
    atomic_write_yaml(path, payload, sort_keys=False)
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
