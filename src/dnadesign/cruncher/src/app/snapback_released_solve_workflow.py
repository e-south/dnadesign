"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_released_solve_workflow.py

Application orchestration for released-product snapback solve/materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dnadesign.cruncher.app.snapback_released_catalogs import resolve_released_catalogs
from dnadesign.cruncher.app.snapback_released_solve_materialize import materialize_released_solve_hits
from dnadesign.cruncher.app.snapback_released_solve_reporting import (
    build_released_solve_report,
    select_released_solve_hits,
)
from dnadesign.cruncher.app.snapback_released_solve_snapshot import dump_released_solve_request_snapshot_yaml
from dnadesign.cruncher.snapback.released_artifacts import (
    build_released_run_dir,
    build_released_solve_manifest,
    ensure_released_solve_run_dirs,
    snapshot_released_solve_inputs,
    write_released_solve_manifest,
    write_released_solve_report,
    write_released_solve_status,
    write_released_solve_summary_table,
)
from dnadesign.cruncher.snapback.released_models import (
    ReleasedSolveOutputConfig,
    ReleasedSolveReport,
    SingleNickReleasedTargetSearchRequest,
)
from dnadesign.cruncher.snapback.released_target_search import search_released_target_hits


def run_released_snapback_solve(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    output: ReleasedSolveOutputConfig,
    workspace_root: Path,
    force_overwrite: bool = False,
) -> tuple[Path, ReleasedSolveReport]:
    resolved_catalogs = resolve_released_catalogs(
        nick_sources=request.nick_sources,
        release_sources=request.release_sources,
        workspace_root=workspace_root,
    )
    search_report = search_released_target_hits(
        request=request,
        nick_catalog=resolved_catalogs.nick_catalog,
        release_catalog=resolved_catalogs.release_catalog,
        workspace_root=workspace_root,
        nick_catalog_source=resolved_catalogs.nick_catalog_source,
        release_catalog_source=resolved_catalogs.release_catalog_source,
    )

    request_yaml = dump_released_solve_request_snapshot_yaml(request=request, output=output)
    run_dir = build_released_run_dir(
        workspace_root=workspace_root,
        run_root=output.run_dir,
    )
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(
                f"Released-product snapback solve run directory already exists: {run_dir}. "
                "Use --force-overwrite to replace it."
            )
        shutil.rmtree(run_dir)
    ensure_released_solve_run_dirs(run_dir)
    snapshot_released_solve_inputs(
        run_dir,
        request_yaml=request_yaml,
        nick_catalog_yaml=resolved_catalogs.nick_catalog_yaml,
        release_catalog_yaml=resolved_catalogs.release_catalog_yaml,
    )

    selection = select_released_solve_hits(search_report)
    materialized_hits = materialize_released_solve_hits(
        hits=selection.hits,
        run_dir=run_dir,
        workspace_root=workspace_root,
        output=output,
    )
    report = build_released_solve_report(
        search_report=search_report,
        request=request,
        output=output,
        resolved_catalogs=resolved_catalogs,
        workspace_root=workspace_root,
        run_dir=run_dir,
        materialized_hits=materialized_hits,
        selected_hit_kind=selection.selected_hit_kind,
    )
    write_released_solve_report(run_dir, report)
    write_released_solve_summary_table(run_dir, report)
    write_released_solve_manifest(
        run_dir,
        build_released_solve_manifest(run_dir=run_dir, workspace_root=workspace_root, report=report),
    )
    write_released_solve_status(run_dir, report=report)
    return run_dir, report


__all__ = ["run_released_snapback_solve"]
