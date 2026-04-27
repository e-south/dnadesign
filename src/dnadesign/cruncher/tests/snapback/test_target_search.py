"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/snapback/test_target_search.py

Target-first snapback catalog search tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.snapback_target_search_workflow import run_snapback_target_search
from dnadesign.cruncher.snapback.models import CatalogSources
from dnadesign.cruncher.snapback.target_models import SnapbackTargetGeometry


def test_run_snapback_target_search_finds_exact_origin_hit_and_later_near_hits(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces" / "demo_snapback"
    workspace_root.mkdir(parents=True, exist_ok=True)

    report = run_snapback_target_search(
        catalog=CatalogSources(preset="neb_nicking_v1", additional_presets=["thermo_nicking_v1"]),
        workspace_root=workspace_root,
        target=SnapbackTargetGeometry(
            nick_boundary_from_left=0,
            paired_bp=3,
            cap_nt=3,
            require_site_sequence_preserved=True,
        ),
        normalize_to_top_strand_nick=True,
        max_results=8,
    )

    assert report.status == "exact_hits_found"
    assert report.metadata.exact_hit_count >= 1
    assert report.exact_hits[0].variant_id == "Nb.BsrDI"
    assert report.exact_hits[0].nick_boundary_from_left == 0
    assert report.exact_hits[0].paired_bp == 3
    assert report.exact_hits[0].cap_nt == 3
    assert report.exact_hits[0].input_length_nt == 6
    assert report.exact_hits[0].site_mutation_count == 0
    assert report.exact_hits[0].intended_site_sequence == "CATTGC"
    assert {hit.variant_id for hit in report.exact_hits} == {"Nb.BsrDI", "Nb.BtsI", "Nt.CviPII"}

    near_boundaries = {(hit.variant_id, hit.nick_boundary_from_left) for hit in report.near_hits}
    assert ("Nt.Bpu10I", 2) in near_boundaries

    feasibility = {(row.variant_id, row.orientation): row for row in report.feasibility}
    assert feasibility[("Nt.CviPII", "forward")].exact_boundary_hit_possible is True
    assert feasibility[("Nt.BspQI", "forward")].earliest_feasible_boundary == 8
    assert "NEGATIVE_SITE_START_AT_TARGET_BOUNDARY" in feasibility[("Nt.BspQI", "forward")].exact_boundary_blockers
