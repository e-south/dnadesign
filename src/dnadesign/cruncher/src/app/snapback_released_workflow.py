"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_released_workflow.py

Application orchestration for released-product snapback explicit workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dnadesign.cruncher.app.snapback_released_catalogs import (
    released_catalog_sources_summary,
    resolve_released_catalogs,
)
from dnadesign.cruncher.nickases.errors import NickaseCatalogError
from dnadesign.cruncher.release_enzymes.errors import ReleaseEnzymeCatalogError
from dnadesign.cruncher.snapback.load import load_released_snapback_spec
from dnadesign.cruncher.snapback.released_artifacts import (
    build_released_manifest,
    build_released_run_dir,
    ensure_released_run_dirs,
    snapshot_released_inputs,
    write_released_manifest,
    write_released_report,
    write_released_status,
    write_released_summary_table,
)
from dnadesign.cruncher.snapback.released_explicit_evaluation import (
    build_invalid_catalog_report,
    build_released_explicit_report,
)
from dnadesign.cruncher.snapback.released_models import ReleasedSnapbackEvaluationReport


def validate_released_snapback_spec(path: str | Path) -> ReleasedSnapbackEvaluationReport:
    spec, spec_path, workspace_root = load_released_snapback_spec(path)
    catalog_summary = released_catalog_sources_summary(
        nick_sources=spec.nick_stage.catalog,
        release_sources=spec.release_stage.catalog,
    )
    try:
        resolved_catalogs = resolve_released_catalogs(
            nick_sources=spec.nick_stage.catalog,
            release_sources=spec.release_stage.catalog,
            workspace_root=workspace_root,
        )
    except (FileNotFoundError, NickaseCatalogError, ReleaseEnzymeCatalogError) as exc:
        return build_invalid_catalog_report(
            spec,
            spec_path=spec_path,
            workspace_root=workspace_root,
            nick_catalog_source=catalog_summary.nick_catalog_source,
            release_catalog_source=catalog_summary.release_catalog_source,
            disallowed_nickase_warning_codes=spec.constraints.disallowed_nickase_warning_codes,
            code="CATALOG_LOAD_FAILED",
            message=str(exc),
        )
    return build_released_explicit_report(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        nick_catalog=resolved_catalogs.nick_catalog,
        release_catalog=resolved_catalogs.release_catalog,
        nick_catalog_source=resolved_catalogs.nick_catalog_source,
        release_catalog_source=resolved_catalogs.release_catalog_source,
    )


def run_released_snapback_design(path: str | Path, *, force_overwrite: bool = False):
    spec, spec_path, workspace_root = load_released_snapback_spec(path)
    resolved_catalogs = resolve_released_catalogs(
        nick_sources=spec.nick_stage.catalog,
        release_sources=spec.release_stage.catalog,
        workspace_root=workspace_root,
    )
    report = build_released_explicit_report(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        nick_catalog=resolved_catalogs.nick_catalog,
        release_catalog=resolved_catalogs.release_catalog,
        nick_catalog_source=resolved_catalogs.nick_catalog_source,
        release_catalog_source=resolved_catalogs.release_catalog_source,
    )
    run_dir = build_released_run_dir(
        workspace_root=workspace_root,
        run_root=spec.output.run_dir,
    )
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(
                "Released-product snapback run directory already exists: "
                f"{run_dir}. Use --force-overwrite to replace it."
            )
        shutil.rmtree(run_dir)
    ensure_released_run_dirs(run_dir)
    snapshot_released_inputs(
        run_dir,
        spec_path=spec_path,
        nick_catalog_yaml=resolved_catalogs.nick_catalog_yaml,
        release_catalog_yaml=resolved_catalogs.release_catalog_yaml,
    )
    report = report.model_copy(update={"run_dir": str(run_dir.resolve())})
    write_released_report(run_dir, report)
    write_released_summary_table(run_dir, report)
    write_released_manifest(
        run_dir,
        build_released_manifest(
            run_dir=run_dir,
            workspace_root=workspace_root,
            spec_path=spec_path,
            report=report,
        ),
    )
    write_released_status(run_dir, report=report)
    return run_dir, report


__all__ = ["run_released_snapback_design", "validate_released_snapback_spec"]
