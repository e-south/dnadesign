"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/released_snapback/test_screen.py

Screen-level tests for the released-product Snapback study objective.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.snapback_screen_workflow import (
    build_snapback_screen_request,
    run_snapback_screen,
)


def test_snapback_screen_request_defaults_encode_origin_zero_retained_active_semantics() -> None:
    request = build_snapback_screen_request()

    assert request.target.nick_boundary_from_left == 0
    assert request.target.paired_bp == 3
    assert request.target.cap_nt == 3
    assert request.nick_sources.preset == "neb_nicking_v1"
    assert request.nick_sources.additional_presets == ["thermo_nicking_v1"]
    assert request.release_sources.preset == "type_iis_release_v1"
    assert request.search.allowed_release_variant_ids == ["BspQI"]
    assert request.search.allowed_active_strands == ["top", "bottom"]
    assert request.search.allowed_route_families == [
        "bottom_active_from_top_nick",
        "top_active_from_bottom_nick",
    ]
    assert request.search.allow_precut_footprint_outside_active_product is True


def test_snapback_screen_real_presets_emit_exact_mechanism_ledger(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces" / "de033"
    workspace_root.mkdir(parents=True, exist_ok=True)

    report = run_snapback_screen(
        request=build_snapback_screen_request(max_results=16),
        workspace_root=workspace_root,
    )

    assert report.status == "exact_hits_found"
    assert report.target_topology.logical_origin == 0
    assert report.target_topology.stem_bp == 3
    assert report.target_topology.cap_nt == 3
    assert report.target_topology.retained_product_strands == ["top", "bottom"]
    assert report.target_topology.allow_oriented_vendor_footprints is True
    assert report.target_topology.allow_degenerate_motif_assignment is True
    assert report.target_topology.allow_release_trim_after_foldback_return is True

    assert {entry.release_variant_id for entry in report.mechanism_ledger} == {"BspQI"}
    expected_ids = {"Nt.BstNBI", "Nt.AlwI", "Nt.BsmAI", "Nb.BsrDI", "Nb.BtsI"}
    ledger_by_id = {entry.nickase_variant_id: entry for entry in report.mechanism_ledger}
    assert expected_ids == set(ledger_by_id)

    for variant_id in expected_ids:
        entry = ledger_by_id[variant_id]
        assert entry.hit_kind == "exact"
        assert entry.logical_origin == 0
        assert entry.logical_stem_bp == 3
        assert entry.cap_nt == 3
        assert entry.foldback_mismatch_count == 0
        assert entry.release_terminal_boundary >= entry.logical_foldback_return_span.end
        assert entry.mechanism_class in {
            "degenerate_footprint_snapback",
            "fixed_footprint_plus_release_trim",
            "mixed_footprint_payload",
        }
        assert entry.provenance_counts
        assert entry.frame_transforms

    assert ledger_by_id["Nt.BsmAI"].release_variant_id == "BspQI"
    assert ledger_by_id["Nt.BsmAI"].provenance_counts["degenerate_motif_base"] >= 1
    assert ledger_by_id["Nt.BstNBI"].release_variant_id == "BspQI"
    assert ledger_by_id["Nb.BsrDI"].retained_product_strand == "bottom"
    assert ledger_by_id["Nb.BsrDI"].physical_nicked_strand == "top"
    assert ledger_by_id["Nb.BtsI"].retained_product_strand == "bottom"
    assert ledger_by_id["Nb.BtsI"].physical_nicked_strand == "top"
