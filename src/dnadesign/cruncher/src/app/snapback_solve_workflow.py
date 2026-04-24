"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_solve_workflow.py

Application orchestration for v3 co-design snapback solve workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dnadesign.cruncher.app.snapback_catalogs import resolve_snapback_catalog
from dnadesign.cruncher.app.snapback_solve_materialize import materialize_snapback_solve_hits
from dnadesign.cruncher.app.snapback_solve_reporting import build_snapback_solve_report
from dnadesign.cruncher.snapback.artifacts import (
    build_solve_manifest,
    build_solve_run_dir,
    ensure_solve_run_dirs,
    solve_id,
    write_solve_frontier_table,
    write_solve_hits_table,
    write_solve_inputs,
    write_solve_manifest,
    write_solve_report,
    write_solve_status,
)
from dnadesign.cruncher.snapback.load import load_snapback_solve_spec
from dnadesign.cruncher.snapback.solver import render_solve_markdown_report, solve_snapback_search


def run_snapback_solve(path: str | Path, *, force_overwrite: bool = False):
    spec, spec_path, workspace_root = load_snapback_solve_spec(path)
    resolved_catalog = resolve_snapback_catalog(sources=spec.catalog, workspace_root=workspace_root)
    report = solve_snapback_search(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        catalog=resolved_catalog.catalog,
    )
    snapback_solve_id = solve_id(
        spec_bytes=spec_path.read_bytes(),
        catalog_bytes=resolved_catalog.catalog_yaml.encode("utf-8"),
    )
    run_dir = build_solve_run_dir(
        workspace_root=workspace_root,
        run_root=spec.output.run_dir,
        snapback_solve_id=snapback_solve_id,
    )
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(
                f"Snapback solve run directory already exists: {run_dir}. Use --force-overwrite to replace it."
            )
        shutil.rmtree(run_dir)
    ensure_solve_run_dirs(run_dir)
    write_solve_inputs(run_dir, spec_path=spec_path, catalog_yaml=resolved_catalog.catalog_yaml)

    materialized_hits = materialize_snapback_solve_hits(
        spec=spec,
        hits=report.hits,
        run_dir=run_dir,
        workspace_root=workspace_root,
        catalog_yaml=resolved_catalog.catalog_yaml,
        catalog_source=resolved_catalog.catalog_source,
    )
    report = build_snapback_solve_report(
        report=report,
        solve_id=snapback_solve_id,
        run_dir=run_dir,
        materialized_hits=materialized_hits,
    )
    write_solve_report(run_dir, report, markdown=render_solve_markdown_report(report))
    write_solve_hits_table(run_dir, report)
    write_solve_frontier_table(run_dir, report)
    write_solve_manifest(
        run_dir,
        build_solve_manifest(run_dir=run_dir, workspace_root=workspace_root, spec_path=spec_path, report=report),
    )
    write_solve_status(
        run_dir,
        report=report,
        status_message=(
            f"snapback solve {report.status} (hits={len(report.hits)}, materialized={len(materialized_hits)})"
        ),
    )
    return run_dir, report
