"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/cassette_workflow.py

Application orchestration for the cassette workflow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dnadesign.cruncher.cassette.artifacts import (
    build_manifest,
    build_run_dir,
    catalog_snapshot_path,
    design_id,
    ensure_run_dirs,
    hairpin_job_path,
    hairpin_view_path,
    linear_duplex_job_path,
    linear_duplex_view_path,
    load_manifest,
    load_status,
    report_json_path,
    report_md_path,
    snapshot_inputs,
    spec_snapshot_path,
    views_manifest_path,
    write_baserender_job,
    write_candidate_table,
    write_manifest,
    write_report,
    write_status,
    write_view_bundle,
)
from dnadesign.cruncher.cassette.catalog import load_nickase_catalog, resolve_catalog_path
from dnadesign.cruncher.cassette.load import load_cassette_spec
from dnadesign.cruncher.cassette.models import CassetteEvaluationReport
from dnadesign.cruncher.cassette.planner import build_cassette_report, render_markdown_report
from dnadesign.cruncher.cassette.view_contracts import (
    build_hairpin_topology_view,
    build_linear_duplex_view,
    build_single_view_job,
    build_views_manifest,
)


def validate_cassette_spec(path: str | Path) -> CassetteEvaluationReport:
    spec, spec_path, workspace_root = load_cassette_spec(path)
    catalog_path = resolve_catalog_path(spec, workspace_root=workspace_root)
    catalog = load_nickase_catalog(catalog_path)
    report = build_cassette_report(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        catalog_path=catalog_path,
        catalog=catalog,
    )
    return report


def run_cassette_design(path: str | Path, *, force_overwrite: bool = False) -> tuple[Path, CassetteEvaluationReport]:
    spec, spec_path, workspace_root = load_cassette_spec(path)
    catalog_path = resolve_catalog_path(spec, workspace_root=workspace_root)
    catalog = load_nickase_catalog(catalog_path)
    report = build_cassette_report(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        catalog_path=catalog_path,
        catalog=catalog,
    )
    spec_bytes = spec_path.read_bytes()
    catalog_bytes = catalog_path.read_bytes()
    cassette_design_id = design_id(spec_bytes=spec_bytes, catalog_bytes=catalog_bytes)
    run_dir = build_run_dir(
        workspace_root=workspace_root,
        run_root=spec.output.run_dir,
        spec_name=spec.name,
        cassette_design_id=cassette_design_id,
    )
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(f"Cassette run directory already exists: {run_dir}. Use --force-overwrite to replace it.")
        shutil.rmtree(run_dir)
    ensure_run_dirs(run_dir)
    snapshot_inputs(run_dir, spec_path=spec_path, catalog_path=catalog_path)
    report = report.model_copy(update={"run_dir": str(run_dir.resolve())})
    write_report(run_dir, report, markdown=render_markdown_report(report))
    write_candidate_table(run_dir, report)
    if spec.output.emit_visual_contracts and report.candidate is not None:
        linear_view = build_linear_duplex_view(
            report=report,
            solution_id=cassette_design_id,
            title=f"{report.spec_name} [{cassette_design_id}] - Linear duplex",
        )
        hairpin_view = build_hairpin_topology_view(
            report=report,
            solution_id=cassette_design_id,
            title=f"{report.spec_name} [{cassette_design_id}] - ssDNA hairpin",
        )
        manifest = build_views_manifest(
            solution_id=cassette_design_id,
            rank=None,
            include_jobs=spec.output.emit_baserender_jobs,
        )
        write_view_bundle(
            run_dir,
            linear_duplex=linear_view.model_dump(mode="json"),
            hairpin=hairpin_view.model_dump(mode="json"),
            manifest=manifest.model_dump(mode="json"),
        )
        if spec.output.emit_baserender_jobs:
            if "duplex_qa" in spec.output.baserender_profiles:
                write_baserender_job(
                    linear_duplex_job_path(run_dir),
                    build_single_view_job(
                        input_filename=linear_duplex_view_path(run_dir).name,
                        adapter_kind="duplex_sequence_v1",
                        renderer="sequence_rows",
                        style_preset="cassette_duplex_qa",
                        output_filename="linear_duplex.pdf",
                    ),
                )
            if "hairpin_qa" in spec.output.baserender_profiles:
                write_baserender_job(
                    hairpin_job_path(run_dir),
                    build_single_view_job(
                        input_filename=hairpin_view_path(run_dir).name,
                        adapter_kind="hairpin_topology_v1",
                        renderer="hairpin_cartoon",
                        style_preset="cassette_hairpin_qa",
                        output_filename="ssdna_hairpin.pdf",
                    ),
                )
    manifest = build_manifest(
        run_dir=run_dir,
        workspace_root=workspace_root,
        spec_path=spec_path,
        catalog_path=catalog_path,
        report=report,
    )
    write_manifest(run_dir, manifest)
    status = "completed" if report.status == "satisfied" else "unsatisfied"
    message = (
        f"cassette design {report.status} "
        f"(schema v{report.metadata.spec_schema_version}, {report.metadata.coordinate_semantics})"
    )
    write_status(run_dir, status=status, status_message=message, report=report)
    return run_dir, report


def cassette_show_payload(run_dir: str | Path) -> dict[str, object]:
    resolved = Path(run_dir).expanduser().resolve()
    manifest = load_manifest(resolved)
    status = load_status(resolved)
    report_file = report_json_path(resolved)
    payload = {
        "run_dir": str(resolved),
        "spec_name": manifest.get("spec_name"),
        "status": status.get("status"),
        "status_message": status.get("status_message"),
        "manifest_path": str((resolved / "meta" / "cassette_manifest.json").resolve()),
        "status_path": str((resolved / "meta" / "cassette_status.json").resolve()),
        "report_json": str(report_file.resolve()),
        "report_md": str(report_md_path(resolved).resolve()),
        "spec_snapshot": str(spec_snapshot_path(resolved).resolve()),
        "catalog_snapshot": str(catalog_snapshot_path(resolved).resolve()),
        "views_manifest": (
            str(views_manifest_path(resolved).resolve()) if views_manifest_path(resolved).exists() else None
        ),
        "linear_duplex_view": (
            str(linear_duplex_view_path(resolved).resolve()) if linear_duplex_view_path(resolved).exists() else None
        ),
        "ssdna_hairpin_view": (
            str(hairpin_view_path(resolved).resolve()) if hairpin_view_path(resolved).exists() else None
        ),
        "linear_duplex_job": (
            str(linear_duplex_job_path(resolved).resolve()) if linear_duplex_job_path(resolved).exists() else None
        ),
        "ssdna_hairpin_job": (
            str(hairpin_job_path(resolved).resolve()) if hairpin_job_path(resolved).exists() else None
        ),
        "artifacts": manifest.get("artifacts", []),
    }
    return payload
