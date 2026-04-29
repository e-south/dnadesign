"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_snapback_cli_requests.py

Typed request-builder tests for Snapback CLI/app boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.cruncher.app.snapback_cli_requests import (
    build_released_solve_invocation,
    build_released_target_search_invocation,
    build_snapback_target_search_invocation,
)


def test_build_snapback_target_search_invocation_supplies_default_presets() -> None:
    invocation = build_snapback_target_search_invocation(
        preset=None,
        additional_preset=[],
        additional_path=[],
        workspace_root=Path("."),
        nick_boundary=0,
        paired_bp=3,
        cap_nt=3,
        max_results=8,
        normalize_to_top_strand_nick=True,
    )

    assert invocation.catalog.preset == "neb_nicking_v1"
    assert invocation.catalog.additional_presets == ["thermo_nicking_v1"]
    assert invocation.target.nick_boundary_from_left == 0
    assert invocation.target.paired_bp == 3
    assert invocation.target.cap_nt == 3


def test_build_released_target_search_invocation_requires_explicit_sources() -> None:
    with pytest.raises(ValueError, match="requires at least one explicit nickase source"):
        build_released_target_search_invocation(
            nick_preset=None,
            nick_additional_preset=[],
            nick_additional_path=[],
            release_preset="type_iis_release_v1",
            release_additional_preset=[],
            release_additional_path=[],
            workspace_root=Path("."),
            nick_boundary=0,
            paired_bp=3,
            cap_nt=3,
            max_results=8,
            near_boundary_search_limit=8,
            release_variant_id=[],
            allow_demo_hits=False,
            allow_frequent_cutter_nickases=False,
            allow_top_active_routes=False,
            allow_precut_footprint_outside_active_product=False,
        )


def test_build_released_target_search_invocation_sets_retained_active_route_policy() -> None:
    invocation = build_released_target_search_invocation(
        nick_preset="neb_nicking_v1",
        nick_additional_preset=["thermo_nicking_v1"],
        nick_additional_path=[],
        release_preset="type_iis_release_v1",
        release_additional_preset=[],
        release_additional_path=[],
        workspace_root=Path("."),
        nick_boundary=0,
        paired_bp=3,
        cap_nt=3,
        max_results=8,
        near_boundary_search_limit=8,
        release_variant_id=["BspQI"],
        allow_demo_hits=False,
        allow_frequent_cutter_nickases=False,
        allow_top_active_routes=True,
        allow_precut_footprint_outside_active_product=True,
    )

    assert invocation.request.search.allowed_active_strands == ["top", "bottom"]
    assert invocation.request.search.allowed_route_families == [
        "bottom_active_from_top_nick",
        "top_active_from_bottom_nick",
    ]
    assert invocation.request.search.route_policy_final_geometry_source == "retained_active_strand"
    assert invocation.request.search.allowed_release_variant_ids == ["BspQI"]
    assert invocation.request.search.allow_precut_footprint_outside_active_product is True
    assert invocation.request.search.disallowed_nickase_warning_codes == ["FREQUENT_CUTTER"]


def test_build_released_solve_invocation_raises_max_results_to_materialize_top_k() -> None:
    invocation = build_released_solve_invocation(
        nick_preset="neb_nicking_v1",
        nick_additional_preset=["thermo_nicking_v1"],
        nick_additional_path=[],
        release_preset="type_iis_release_v1",
        release_additional_preset=[],
        release_additional_path=[],
        workspace_root=Path("."),
        nick_boundary=0,
        paired_bp=3,
        cap_nt=3,
        max_results=1,
        near_boundary_search_limit=8,
        materialize_top_k=4,
        release_variant_id=[],
        run_dir=Path("outputs/released_solve"),
        render_format="pdf",
        emit_renders=False,
        allow_demo_hits=False,
        allow_frequent_cutter_nickases=True,
        allow_top_active_routes=False,
        allow_precut_footprint_outside_active_product=False,
    )

    assert invocation.request.search.max_results == 4
    assert invocation.request.search.route_policy_final_geometry_source == "exposed_bottom_strand"
    assert invocation.request.search.disallowed_nickase_warning_codes == []
    assert invocation.output.materialize_top_k == 4
