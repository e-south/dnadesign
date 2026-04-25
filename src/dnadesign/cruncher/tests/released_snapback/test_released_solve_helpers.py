"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/released_snapback/test_released_solve_helpers.py

Focused helper tests for released-product Snapback solve seams.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.cruncher.app.snapback_released_catalogs import resolve_released_catalogs
from dnadesign.cruncher.app.snapback_released_solve_materialize import materialize_released_solve_hit
from dnadesign.cruncher.app.snapback_released_solve_reporting import (
    build_released_solve_report,
    select_released_solve_hits,
)
from dnadesign.cruncher.app.snapback_released_solve_snapshot import (
    build_released_solve_request_snapshot_payload,
    dump_released_solve_request_snapshot_yaml,
)
from dnadesign.cruncher.app.snapback_released_target_search_workflow import run_released_snapback_target_search
from dnadesign.cruncher.snapback.models import CatalogSources
from dnadesign.cruncher.snapback.released_artifacts import (
    ensure_released_solve_run_dirs,
    released_solve_hit_json_path,
    released_solve_hit_plot_context_path,
    released_solve_hit_run_dir,
)
from dnadesign.cruncher.snapback.released_models import (
    ReleaseCatalogSources,
    ReleasedFinalTargetGeometry,
    ReleasedSolveOutputConfig,
    ReleasedTargetSearchConfig,
    SingleNickReleasedTargetSearchRequest,
)
from dnadesign.cruncher.tests.released_snapback.builders import write_released_workspace


def _local_request() -> SingleNickReleasedTargetSearchRequest:
    return SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(additional_paths=[Path("inputs/nickases/local.nickases.yaml")]),
        release_sources=ReleaseCatalogSources(additional_paths=[Path("inputs/release_enzymes/local.release.yaml")]),
        search=ReleasedTargetSearchConfig(max_results=2, near_boundary_search_limit=2),
    )


def test_build_released_solve_request_snapshot_payload_freezes_request_and_output() -> None:
    request = _local_request()
    output = ReleasedSolveOutputConfig(
        run_dir=Path("outputs/released_solve"),
        materialize_top_k=2,
        render_format="svg",
        emit_renders=True,
    )

    payload = build_released_solve_request_snapshot_payload(request=request, output=output)
    dumped = yaml.safe_load(dump_released_solve_request_snapshot_yaml(request=request, output=output))

    assert payload["released_solve"]["kind"] == "single_nick_released_solve_v1"
    assert payload["target"] == {"nick_boundary_from_left": 0, "paired_bp": 3, "cap_nt": 3}
    assert payload["nick_sources"]["additional_paths"] == ["inputs/nickases/local.nickases.yaml"]
    assert payload["release_sources"]["additional_paths"] == ["inputs/release_enzymes/local.release.yaml"]
    assert payload["output"]["materialize_top_k"] == 2
    assert payload["output"]["render_format"] == "svg"
    assert dumped == payload


def test_select_released_solve_hits_prefers_exact_then_nearest(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)
    report = run_released_snapback_target_search(
        request=_local_request(),
        workspace_root=fixture.workspace_root,
    )

    exact_selection = select_released_solve_hits(report)
    near_only_report = report.model_copy(
        update={
            "status": "near_hits_only",
            "exact_hits": [],
            "metadata": report.metadata.model_copy(
                update={
                    "pre_truncation_exact_hit_count": 0,
                    "post_truncation_exact_hit_count": 0,
                }
            ),
        }
    )
    near_selection = select_released_solve_hits(near_only_report)

    assert exact_selection.selected_hit_kind == "exact"
    assert exact_selection.hits == list(report.exact_hits)
    assert near_selection.selected_hit_kind == "nearest"
    assert near_selection.hits == list(report.near_hits)


def test_materialize_released_solve_hit_writes_expected_bundle_without_renders(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)
    request = _local_request()
    report = run_released_snapback_target_search(request=request, workspace_root=fixture.workspace_root)
    run_dir = fixture.workspace_root / "outputs" / "released_solve_helper"
    ensure_released_solve_run_dirs(run_dir)

    materialized = materialize_released_solve_hit(
        hit=report.exact_hits[0],
        rank=1,
        run_dir=run_dir,
        workspace_root=fixture.workspace_root,
        output=ReleasedSolveOutputConfig(
            run_dir=Path("outputs/released_solve_helper"),
            materialize_top_k=1,
            render_format="pdf",
            emit_renders=False,
        ),
    )

    hit_run_dir = released_solve_hit_run_dir(run_dir, rank=1)
    assert materialized.rank == 1
    assert materialized.rendered_plot_path is None
    assert materialized.materialized_run_dir == "outputs/released_solve_helper/analysis/materialized_hits/hit_01"
    assert released_solve_hit_json_path(hit_run_dir).exists()
    assert released_solve_hit_plot_context_path(hit_run_dir).exists()


def test_build_released_solve_report_preserves_route_policy_and_counts(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)
    request = _local_request()
    output = ReleasedSolveOutputConfig(
        run_dir=Path("outputs/released_solve"),
        materialize_top_k=1,
        render_format="pdf",
        emit_renders=False,
    )
    search_report = run_released_snapback_target_search(request=request, workspace_root=fixture.workspace_root)
    resolved_catalogs = resolve_released_catalogs(
        nick_sources=request.nick_sources,
        release_sources=request.release_sources,
        workspace_root=fixture.workspace_root,
    )
    run_dir = fixture.workspace_root / output.run_dir
    ensure_released_solve_run_dirs(run_dir)
    materialized_hit = materialize_released_solve_hit(
        hit=search_report.exact_hits[0],
        rank=1,
        run_dir=run_dir,
        workspace_root=fixture.workspace_root,
        output=output,
    )

    report = build_released_solve_report(
        search_report=search_report,
        request=request,
        output=output,
        resolved_catalogs=resolved_catalogs,
        workspace_root=fixture.workspace_root,
        run_dir=run_dir,
        materialized_hits=[materialized_hit],
        selected_hit_kind="exact",
    )

    assert report.status == "exact_hits_materialized"
    assert report.workspace_root == str(fixture.workspace_root.resolve())
    assert report.metadata.selected_hit_kind == "exact"
    assert report.metadata.materialized_hit_count == 1
    assert report.metadata.available_exact_hit_count == search_report.metadata.pre_truncation_exact_hit_count
    assert report.metadata.allowed_route_families == list(request.search.allowed_route_families)
    assert report.metadata.allowed_active_strands == list(request.search.allowed_active_strands)
