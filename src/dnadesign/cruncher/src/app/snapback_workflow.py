"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_workflow.py

Application orchestration for v2 explicit snapback workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dnadesign.cruncher.app.snapback_catalogs import (
    resolve_snapback_catalog,
    unresolved_snapback_catalog_source,
)
from dnadesign.cruncher.app.snapback_publish import (
    build_publication_bundle,
    write_publication_bundle,
)
from dnadesign.cruncher.app.snapback_show_load import load_snapback_show_artifacts
from dnadesign.cruncher.app.snapback_show_present import build_snapback_show_payload
from dnadesign.cruncher.app.snapback_show_validate import validate_snapback_show_artifacts
from dnadesign.cruncher.nickases.errors import NickaseCatalogError
from dnadesign.cruncher.snapback.artifacts import (
    build_manifest,
    build_run_dir,
    design_id,
    ensure_run_dirs,
    snapshot_explicit_inputs,
    write_candidate_table,
    write_manifest,
    write_report,
    write_status,
)
from dnadesign.cruncher.snapback.load import load_snapback_spec
from dnadesign.cruncher.snapback.planner import (
    build_invalid_catalog_report,
    build_snapback_report,
    render_markdown_report,
)


def validate_snapback_spec(path: str | Path):
    spec, spec_path, workspace_root = load_snapback_spec(path)
    catalog_source = unresolved_snapback_catalog_source(sources=spec.design.nickase.catalog)
    try:
        resolved_catalog = resolve_snapback_catalog(
            sources=spec.design.nickase.catalog,
            workspace_root=workspace_root,
        )
    except (FileNotFoundError, NickaseCatalogError) as exc:
        return build_invalid_catalog_report(
            spec,
            spec_path=spec_path,
            workspace_root=workspace_root,
            catalog_source=catalog_source,
            code="CATALOG_LOAD_FAILED",
            message=str(exc),
        )
    return build_snapback_report(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        catalog=resolved_catalog.catalog,
        catalog_source=resolved_catalog.catalog_source,
    )


def run_snapback_design(path: str | Path, *, force_overwrite: bool = False):
    spec, spec_path, workspace_root = load_snapback_spec(path)
    resolved_catalog = resolve_snapback_catalog(
        sources=spec.design.nickase.catalog,
        workspace_root=workspace_root,
    )
    report = build_snapback_report(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        catalog=resolved_catalog.catalog,
        catalog_source=resolved_catalog.catalog_source,
    )
    snapback_design_id = design_id(
        spec_bytes=spec_path.read_bytes(),
        catalog_bytes=resolved_catalog.catalog_yaml.encode("utf-8"),
    )
    run_dir = build_run_dir(
        workspace_root=workspace_root,
        run_root=spec.output.run_dir,
        spec_name=spec.name,
        snapback_design_id=snapback_design_id,
    )
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(f"Snapback run directory already exists: {run_dir}. Use --force-overwrite to replace it.")
        shutil.rmtree(run_dir)
    ensure_run_dirs(
        run_dir,
        include_visual_contracts=spec.output.emit_visual_contracts,
        include_baserender_jobs=spec.output.emit_baserender_jobs,
    )
    snapshot_explicit_inputs(run_dir, spec_path=spec_path, catalog_yaml=resolved_catalog.catalog_yaml)
    report = report.model_copy(update={"run_dir": str(run_dir.resolve())})
    write_report(run_dir, report, markdown=render_markdown_report(report))
    write_candidate_table(run_dir, report)
    if spec.output.emit_visual_contracts and report.candidate is not None:
        write_publication_bundle(
            run_dir,
            bundle=build_publication_bundle(
                report=report,
                solution_id=snapback_design_id,
                include_jobs=spec.output.emit_baserender_jobs,
                render_format=spec.output.render_format,
            ),
        )
    write_manifest(
        run_dir,
        build_manifest(
            run_dir=run_dir,
            workspace_root=workspace_root,
            spec_path=spec_path,
            report=report,
        ),
    )
    write_status(run_dir, report=report)
    return run_dir, report


def snapback_show_payload(run_dir: str | Path) -> dict[str, object]:
    artifacts = load_snapback_show_artifacts(run_dir)
    validate_snapback_show_artifacts(artifacts)
    return build_snapback_show_payload(artifacts)
