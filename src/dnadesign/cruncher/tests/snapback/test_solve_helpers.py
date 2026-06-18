"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/snapback/test_solve_helpers.py

Focused helper tests for preserved-site Snapback solve seams.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.cruncher.app.snapback_catalogs import resolve_snapback_catalog
from dnadesign.cruncher.app.snapback_solve_materialize import materialize_snapback_solve_hit
from dnadesign.cruncher.app.snapback_solve_reporting import build_snapback_solve_report
from dnadesign.cruncher.app.snapback_solve_snapshot import (
    build_snapback_explicit_spec_payload_for_hit,
    dump_snapback_explicit_spec_yaml_for_hit,
)
from dnadesign.cruncher.snapback.artifacts import ensure_solve_run_dirs, solve_hit_run_dir
from dnadesign.cruncher.snapback.load import load_snapback_solve_spec
from dnadesign.cruncher.snapback.solver import solve_snapback_search
from dnadesign.cruncher.tests.snapback.builders import write_snapback_workspace


def _solve_fixture(tmp_path: Path):
    fixture = write_snapback_workspace(tmp_path)
    spec, spec_path, workspace_root = load_snapback_solve_spec(fixture.solve_path)
    resolved_catalog = resolve_snapback_catalog(sources=spec.catalog, workspace_root=workspace_root)
    report = solve_snapback_search(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        catalog=resolved_catalog.catalog,
    )
    return fixture, spec, workspace_root, resolved_catalog, report


def test_build_snapback_explicit_spec_payload_for_hit_freezes_workspace_relative_output(tmp_path: Path) -> None:
    fixture, spec, workspace_root, _resolved_catalog, report = _solve_fixture(tmp_path)
    hit = report.hits[0]
    hit_run_dir = fixture.workspace_root / "outputs" / "solve_helper" / "analysis" / "materialized_hits" / "hit_01"

    payload = build_snapback_explicit_spec_payload_for_hit(
        spec,
        hit=hit,
        workspace_root=workspace_root,
        hit_run_dir=hit_run_dir,
    )
    dumped = yaml.safe_load(
        dump_snapback_explicit_spec_yaml_for_hit(
            spec,
            hit=hit,
            workspace_root=workspace_root,
            hit_run_dir=hit_run_dir,
        )
    )

    assert payload["snapback"]["name"] == "demo_snapback_solve__hit_01"
    assert payload["design"]["nickase"]["variant_id"] == hit.variant_id
    assert payload["output"]["run_dir"] == "outputs/solve_helper/analysis/materialized_hits/hit_01"
    assert payload["output"]["emit_visual_contracts"] is True
    assert dumped == payload


def test_materialize_snapback_solve_hit_writes_expected_bundle(tmp_path: Path) -> None:
    fixture, spec, workspace_root, resolved_catalog, report = _solve_fixture(tmp_path)
    run_dir = fixture.workspace_root / "outputs" / "solve_helper"
    ensure_solve_run_dirs(run_dir)

    materialized = materialize_snapback_solve_hit(
        spec=spec,
        hit=report.hits[0],
        run_dir=run_dir,
        workspace_root=workspace_root,
        catalog_yaml=resolved_catalog.catalog_yaml,
        catalog_source=resolved_catalog.catalog_source,
    )

    hit_run_dir = solve_hit_run_dir(run_dir, rank=1)
    assert materialized.materialized_run_dir == "outputs/solve_helper/analysis/materialized_hits/hit_01"
    assert (hit_run_dir / "analysis" / "reports" / "report.json").exists()
    assert (hit_run_dir / "meta" / "snapback_manifest.json").exists()
    assert (hit_run_dir / "analysis" / "views" / "post_nick_foldback.snapback_visual.v1.json").exists()


def test_build_snapback_solve_report_preserves_materialized_count_and_run_dir(tmp_path: Path) -> None:
    fixture, _spec, _workspace_root, _resolved_catalog, report = _solve_fixture(tmp_path)
    run_dir = fixture.workspace_root / "outputs" / "solve_helper"
    materialized_hit = report.hits[0].model_copy(
        update={"materialized_run_dir": "outputs/solve_helper/analysis/materialized_hits/hit_01"}
    )

    updated = build_snapback_solve_report(
        report=report,
        solve_id="demo12345678",
        run_dir=run_dir,
        materialized_hits=[materialized_hit],
    )

    assert updated.solve_id == "demo12345678"
    assert updated.run_dir == str(run_dir.resolve())
    assert updated.metadata.materialized_hit_count == 1
    assert updated.hits[0].materialized_run_dir == "outputs/solve_helper/analysis/materialized_hits/hit_01"
