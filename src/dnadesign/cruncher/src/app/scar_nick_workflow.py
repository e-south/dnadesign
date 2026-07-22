"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/scar_nick_workflow.py

Application orchestration for scar-nick validation, design, and show.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from dnadesign.cruncher.nickases.catalog import (
    dump_nickase_catalog_yaml,
    load_merged_nickase_catalog,
)
from dnadesign.cruncher.release_enzymes.catalog import (
    dump_release_enzyme_catalog_yaml,
    load_merged_release_enzyme_catalog,
)
from dnadesign.cruncher.scar_nick.artifacts import (
    assert_manifest_artifacts_present,
    assert_manifest_hashes_current,
    assert_provenance_current,
    build_manifest,
    build_run_dir,
    candidate_pair_call_table_path,
    candidate_profiles_path,
    candidate_table_path,
    ensure_run_dirs,
    load_manifest,
    load_status,
    manifest_path,
    nickase_catalog_snapshot_path,
    nickase_geometry_audit_path,
    nickase_geometry_audit_table_path,
    post_terminal_nick_visual_contract_path,
    release_catalog_snapshot_path,
    report_json_path,
    report_md_path,
    scar_nick_terminal_nick_job_path,
    scar_nick_terminal_nick_visual_contracts_path,
    snapshot_inputs,
    spec_snapshot_path,
    status_path,
    views_manifest_path,
    write_manifest,
    write_report,
    write_status,
)
from dnadesign.cruncher.scar_nick.load import load_scar_nick_spec
from dnadesign.cruncher.scar_nick.models import ScarNickEvaluationReport, ScarNickSpecDocument
from dnadesign.cruncher.scar_nick.planner import build_scar_nick_report
from dnadesign.cruncher.scar_nick.reporting import render_markdown_report
from dnadesign.cruncher.scar_nick.tables import (
    write_candidate_pair_call_table,
    write_candidate_table,
    write_nickase_geometry_audit_table,
)
from dnadesign.cruncher.scar_nick.visual_publication import (
    assert_visual_publication_current,
    publish_scar_nick_visuals,
)


def _load_catalogs(spec: ScarNickSpecDocument, *, workspace_root: Path):
    release_ref = spec.processing.release.catalog
    nick_ref = spec.processing.nick.catalog
    release_catalog, _release_paths = load_merged_release_enzyme_catalog(
        preset_id=release_ref.preset,
        additional_preset_ids=release_ref.additional_presets,
        additional_paths=release_ref.additional_paths,
        workspace_root=workspace_root,
    )
    nickase_catalog, _nick_paths = load_merged_nickase_catalog(
        preset_id=nick_ref.preset,
        additional_preset_ids=nick_ref.additional_presets,
        additional_paths=nick_ref.additional_paths,
        workspace_root=workspace_root,
    )
    return release_catalog, nickase_catalog


def validate_scar_nick_spec(path: str | Path) -> ScarNickEvaluationReport:
    spec, spec_path, workspace_root = load_scar_nick_spec(path)
    release_catalog, nickase_catalog = _load_catalogs(spec, workspace_root=workspace_root)
    return build_scar_nick_report(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        release_catalog=release_catalog,
        nickase_catalog=nickase_catalog,
    )


def run_scar_nick_design(path: str | Path, *, force_overwrite: bool = False) -> tuple[Path, ScarNickEvaluationReport]:
    spec, spec_path, workspace_root = load_scar_nick_spec(path)
    release_catalog, nickase_catalog = _load_catalogs(spec, workspace_root=workspace_root)
    report = build_scar_nick_report(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        release_catalog=release_catalog,
        nickase_catalog=nickase_catalog,
    )
    run_dir = build_run_dir(workspace_root=workspace_root, run_dir=spec.output.run_dir)
    if report.status != "satisfied":
        issue_codes = ", ".join(issue.code for issue in report.issues) or "unknown"
        raise ValueError(f"Scar-nick design is unsatisfied; run validate for details. issues={issue_codes}")
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(f"Scar-nick run directory already exists: {run_dir}. Use --force-overwrite to replace it.")
        shutil.rmtree(run_dir)
    ensure_run_dirs(run_dir)
    snapshot_inputs(
        run_dir,
        spec_path=spec_path,
        release_catalog_yaml=dump_release_enzyme_catalog_yaml(release_catalog),
        nickase_catalog_yaml=dump_nickase_catalog_yaml(nickase_catalog),
    )
    report = report.model_copy(update={"run_dir": str(run_dir.resolve())})
    write_report(run_dir, report, markdown=render_markdown_report(report))
    write_candidate_table(run_dir, report)
    write_candidate_pair_call_table(run_dir, report)
    write_nickase_geometry_audit_table(run_dir, report)
    publish_scar_nick_visuals(run_dir=run_dir, report=report, spec=spec)
    manifest = build_manifest(run_dir=run_dir, workspace_root=workspace_root, spec_path=spec_path, report=report)
    write_manifest(run_dir, manifest)
    write_status(
        run_dir,
        report=report,
        status_message=(
            f"scar-nick design {report.status} (accepted={len(report.candidates)}, issues={len(report.issues)})"
        ),
    )
    return run_dir, report


def scar_nick_show_payload(run_dir: str | Path) -> dict[str, object]:
    resolved = Path(run_dir).expanduser().resolve()
    manifest = load_manifest(resolved)
    status = load_status(resolved)
    report_file = report_json_path(resolved)
    if not report_file.exists():
        raise FileNotFoundError(f"Missing scar-nick report: {report_file}")
    assert_provenance_current(resolved, manifest)
    assert_manifest_artifacts_present(resolved, manifest)
    assert_manifest_hashes_current(resolved, manifest)
    report_payload = json.loads(report_file.read_text(encoding="utf-8"))
    report = ScarNickEvaluationReport.model_validate(report_payload)
    if status.get("status") != "completed" or report.status != "satisfied":
        raise ValueError("Scar-nick show requires a satisfied completed run.")
    assert_visual_publication_current(resolved, report)
    return {
        "run_dir": str(resolved),
        "spec_name": manifest.get("spec_name"),
        "status": status.get("status"),
        "status_message": status.get("status_message"),
        "candidate_count": len(report.candidates),
        "manifest_path": str(manifest_path(resolved).resolve()),
        "status_path": str(status_path(resolved).resolve()),
        "report_json": str(report_file.resolve()),
        "report_md": str(report_md_path(resolved).resolve()),
        "candidate_profiles": str(candidate_profiles_path(resolved).resolve()),
        "nickase_geometry_audit": str(nickase_geometry_audit_path(resolved).resolve()),
        "candidate_table": str(candidate_table_path(resolved).resolve()),
        "candidate_pair_call_table": str(candidate_pair_call_table_path(resolved).resolve()),
        "nickase_geometry_audit_table": str(nickase_geometry_audit_table_path(resolved).resolve()),
        "views_manifest": (
            str(views_manifest_path(resolved).resolve()) if views_manifest_path(resolved).exists() else None
        ),
        "terminal_nick_visual_contract": (
            str(post_terminal_nick_visual_contract_path(resolved).resolve())
            if post_terminal_nick_visual_contract_path(resolved).exists()
            else None
        ),
        "scar_nick_terminal_nick_visual_contracts": (
            str(scar_nick_terminal_nick_visual_contracts_path(resolved).resolve())
            if scar_nick_terminal_nick_visual_contracts_path(resolved).exists()
            else None
        ),
        "baserender_job": (
            str(scar_nick_terminal_nick_job_path(resolved).resolve())
            if scar_nick_terminal_nick_job_path(resolved).exists()
            else None
        ),
        "spec_snapshot": str(spec_snapshot_path(resolved).resolve()),
        "nickase_catalog": str(nickase_catalog_snapshot_path(resolved).resolve()),
        "release_catalog": str(release_catalog_snapshot_path(resolved).resolve()),
        "artifacts": manifest.get("artifacts", []),
    }


__all__ = ["run_scar_nick_design", "scar_nick_show_payload", "validate_scar_nick_spec"]
