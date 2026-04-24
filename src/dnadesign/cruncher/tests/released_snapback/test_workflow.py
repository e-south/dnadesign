"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/released_snapback/test_workflow.py

Bundle and show-path tests for released-product snapback workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import pytest
import yaml

import dnadesign.cruncher.snapback.released_target_search as released_target_search
from dnadesign.cruncher.app.snapback_released_show import released_show_payload
from dnadesign.cruncher.app.snapback_released_solve_workflow import run_released_snapback_solve
from dnadesign.cruncher.app.snapback_released_target_search_workflow import run_released_snapback_target_search
from dnadesign.cruncher.app.snapback_released_workflow import (
    run_released_snapback_design,
    validate_released_snapback_spec,
)
from dnadesign.cruncher.nickases.catalog import load_merged_nickase_catalog
from dnadesign.cruncher.nickases.selection import matching_nickase_warning_codes
from dnadesign.cruncher.release_enzymes.catalog import load_merged_release_enzyme_catalog
from dnadesign.cruncher.snapback.errors import SnapbackSpecError
from dnadesign.cruncher.snapback.models import CatalogSources
from dnadesign.cruncher.snapback.publication_support import complement_sequence
from dnadesign.cruncher.snapback.released_hit_plot import (
    _ROW_BOTTOM_Y,
    _ROW_TOP_Y,
    _SITE_FOOTPRINT_VERTICAL_PAD,
    _site_footprint_bounds,
    build_released_hit_plot_context,
    render_released_hit_plot,
)
from dnadesign.cruncher.snapback.released_models import (
    ReleaseCatalogSources,
    ReleasedFinalTargetGeometry,
    ReleasedSolveOutputConfig,
    ReleasedTargetSearchConfig,
    SingleNickReleasedTargetSearchRequest,
)
from dnadesign.cruncher.tests.released_snapback.builders import write_released_workspace


def test_released_hit_plot_site_footprint_bounds_track_the_duplex_band() -> None:
    fill_y0, fill_y1 = _site_footprint_bounds()

    assert fill_y0 == pytest.approx(_ROW_BOTTOM_Y - _SITE_FOOTPRINT_VERTICAL_PAD)
    assert fill_y1 == pytest.approx(_ROW_TOP_Y + _SITE_FOOTPRINT_VERTICAL_PAD)
    assert fill_y0 > (_ROW_BOTTOM_Y - 0.08)
    assert fill_y1 < (_ROW_TOP_Y + 0.08)


def test_released_design_writes_bundle_and_released_show_revalidates_it(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)

    run_dir, report = run_released_snapback_design(fixture.spec_path)

    assert report.status == "satisfied"
    assert (run_dir / "meta" / "released_snapback_manifest.json").exists()
    assert (run_dir / "meta" / "released_snapback_status.json").exists()
    assert (run_dir / "analysis" / "report.json").exists()
    assert (run_dir / "analysis" / "released_product_projection.json").exists()
    assert (run_dir / "analysis" / "pre_nick_site.json").exists()
    assert (run_dir / "analysis" / "release_site.json").exists()
    assert (run_dir / "export" / "released_design_summary.csv").exists()

    payload = released_show_payload(run_dir)

    assert payload["kind"] == "released_explicit"
    assert payload["status"] == "satisfied"
    projection_payload = json.loads(
        (run_dir / "analysis" / "released_product_projection.json").read_text(encoding="utf-8")
    )
    assert projection_payload["release_top_cut_precursor"] == 10
    assert projection_payload["release_bottom_cut_precursor"] == 9


def test_released_design_rejects_left_of_origin_outside_site_exact_bundle(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "de033"
    spec_path = workspace / "configs" / "snapback" / "de033.released.snapback.yaml"
    nick_catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    release_catalog_path = workspace / "inputs" / "release_enzymes" / "local.release.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    release_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nt.BsmAI",
                            "specificity_id": "BsmAI",
                            "motif_top_5to3": "GTCTC",
                            "top_cut_offset": 6,
                            "selection": {"outside_site": True},
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    release_catalog_path.write_text(
        yaml.safe_dump(
            {
                "release_enzymes": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "variant_id": "Re.Exact",
                            "display_name": "Re.Exact",
                            "recognition_sequence": "CCAA",
                            "top_cut_offset": 1,
                            "bottom_cut_offset": 0,
                            "class_label": "other_ds_re",
                            "commercial_confidence": "primary_vendor_current",
                            "source_catalog_id": "local_release",
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    spec_path.write_text(
        yaml.safe_dump(
            {
                "released_snapback": {
                    "schema_version": 1,
                    "kind": "single_nick_released_snapback_v1",
                    "name": "de033_left_prefix",
                },
                "input": {
                    "precursor_top_strand": "GTCTCAAACGTTGTTCCAA",
                },
                "nick_stage": {
                    "nickase_variant_id": "Nt.BsmAI",
                    "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
                    "intended_site_sequence": "GTCTC",
                },
                "release_stage": {
                    "release_variant_id": "Re.Exact",
                    "catalog": {"additional_paths": ["inputs/release_enzymes/local.release.yaml"]},
                    "intended_site_sequence": "CCAA",
                    "retained_side": "upstream",
                    "stage_order": "nick_then_release",
                },
                "final_target": {
                    "nick_boundary_from_left": 0,
                    "paired_bp": 3,
                    "cap_nt": 3,
                },
                "constraints": {
                    "allow_post_release_loss_of_nickase_site": True,
                    "allow_post_release_loss_of_release_site": True,
                    "require_release_site_downstream_of_nick": True,
                    "require_complete_downstream_fragment_separation": True,
                },
                "output": {"run_dir": "outputs/released_design"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_dir, report = run_released_snapback_design(spec_path)

    assert report.status == "invalid_precursor"
    assert any(issue.code == "PRE_NICK_SITE_LEFT_OF_ORIGIN" for issue in report.issues)
    payload = released_show_payload(run_dir)
    assert payload["status"] == "invalid_precursor"


def test_released_solve_materializes_hits_and_emits_per_hit_plots(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)
    workspace_root = fixture.workspace_root

    run_dir, report = run_released_snapback_solve(
        request=SingleNickReleasedTargetSearchRequest(
            target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
            nick_sources=CatalogSources(additional_paths=[Path("inputs/nickases/local.nickases.yaml")]),
            release_sources=ReleaseCatalogSources(additional_paths=[Path("inputs/release_enzymes/local.release.yaml")]),
            search=ReleasedTargetSearchConfig(max_results=2, near_boundary_search_limit=2),
        ),
        output=ReleasedSolveOutputConfig(
            run_dir=Path("outputs/released_solve"),
            materialize_top_k=2,
            render_format="pdf",
            emit_renders=True,
        ),
        workspace_root=workspace_root,
        force_overwrite=True,
    )

    assert report.status == "exact_hits_materialized"
    assert report.metadata.materialized_hit_count == 1
    assert report.metadata.selected_hit_kind == "exact"
    assert report.metadata.evaluated_pair_count > 0
    assert report.issues == []
    assert (run_dir / "meta" / "released_solve_manifest.json").exists()
    assert (run_dir / "meta" / "released_solve_status.json").exists()
    assert (run_dir / "analysis" / "solve_report.json").exists()
    assert (run_dir / "export" / "table__hits.csv").exists()
    for hit in report.hits:
        hit_run_dir = workspace_root / hit.materialized_run_dir
        assert hit.render_job_path is None
        assert hit.rendered_plot_path is not None
        assert hit_run_dir.exists()
        assert (workspace_root / hit.rendered_plot_path).exists()
        assert (workspace_root / hit.rendered_plot_path).read_bytes().startswith(b"%PDF")
        assert (hit_run_dir / "analysis" / "target_search_hit.json").exists()
        assert (hit_run_dir / "analysis" / "released_hit_plot_context.json").exists()
        assert (hit_run_dir / "analysis" / "released_product_projection.json").exists()
        assert (hit_run_dir / "analysis" / "pre_nick_site.json").exists()
        assert (hit_run_dir / "analysis" / "release_site.json").exists()
    first_context = json.loads(
        (
            workspace_root / report.hits[0].materialized_run_dir / "analysis" / "released_hit_plot_context.json"
        ).read_text(encoding="utf-8")
    )
    assert first_context["foldback"]["foldback_sequence"] == "CAA"
    assert first_context["foldback"]["foldback_partner_sequence"] == "AAC"


def test_released_design_rejects_frequent_cutter_nickase_by_default(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)
    nick_payload = yaml.safe_load(fixture.nick_catalog_path.read_text(encoding="utf-8"))
    nick_payload["nickases"]["entries"][0]["selection"] = {"warning_codes": ["FREQUENT_CUTTER"]}
    fixture.nick_catalog_path.write_text(yaml.safe_dump(nick_payload, sort_keys=False), encoding="utf-8")

    report = validate_released_snapback_spec(fixture.spec_path)

    assert report.status == "invalid_catalog"
    assert report.issues[0].code == "DISALLOWED_NICKASE_WARNING_CODE"
    assert report.metadata.disallowed_nickase_warning_codes == ["FREQUENT_CUTTER"]


def test_released_design_rejects_unknown_nick_stage_key(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)
    spec_payload = yaml.safe_load(fixture.spec_path.read_text(encoding="utf-8"))
    spec_payload["nick_stage"]["unexpected_unknown_key"] = True
    fixture.spec_path.write_text(yaml.safe_dump(spec_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(SnapbackSpecError, match="unexpected_unknown_key"):
        validate_released_snapback_spec(fixture.spec_path)


def test_released_design_fails_closed_on_ambiguous_precursor_origin(tmp_path: Path) -> None:
    fixture = write_released_workspace(
        tmp_path,
        precursor_top_strand="AACGTTGAACGTTGTTCCAA",
    )

    report = validate_released_snapback_spec(fixture.spec_path)

    assert report.status == "invalid_precursor"
    assert any(issue.code == "PRECURSOR_ORIGIN_AMBIGUOUS" for issue in report.issues)
    assert report.projection is None
    assert report.candidate is None


def test_checked_in_de033_released_design_fixture_stays_invalid() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    spec_path = (
        repo_root
        / "src"
        / "dnadesign"
        / "cruncher"
        / "workspaces"
        / "de033"
        / "configs"
        / "snapback"
        / "de033.released.snapback.yaml"
    )

    report = validate_released_snapback_spec(spec_path)

    assert report.status == "invalid_precursor"
    assert any(issue.code == "PRE_NICK_SITE_LEFT_OF_ORIGIN" for issue in report.issues)


def test_released_solve_real_presets_materializes_exact_hits_with_bottom_strand_context(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces" / "de033"
    workspace_root.mkdir(parents=True, exist_ok=True)
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="neb_nicking_v1", additional_presets=["thermo_nicking_v1"]),
        release_sources=ReleaseCatalogSources(preset="type_iis_release_v1"),
        search=ReleasedTargetSearchConfig(max_results=8, near_boundary_search_limit=8),
    )

    search_report = run_released_snapback_target_search(
        request=request,
        workspace_root=workspace_root,
    )
    nick_catalog, _nick_paths = load_merged_nickase_catalog(
        preset_id="neb_nicking_v1",
        additional_preset_ids=["thermo_nicking_v1"],
        additional_paths=[],
        workspace_root=workspace_root,
    )
    release_catalog, _release_paths = load_merged_release_enzyme_catalog(
        preset_id="type_iis_release_v1",
        additional_paths=[],
        workspace_root=workspace_root,
    )
    disallowed_nick_placement_count = len(
        [
            placement
            for placement in released_target_search._nick_placements(
                nick_catalog,
                physical_nicked_strand="top",
            )
            if matching_nickase_warning_codes(
                placement.entry,
                warning_codes=request.search.disallowed_nickase_warning_codes,
            )
        ]
    )
    release_placement_count = len(
        released_target_search._release_placements(
            release_catalog,
            target=request.target,
        )
    )
    assert search_report.status == "near_hits_only"
    assert search_report.metadata.pre_truncation_exact_hit_count == 0
    assert search_report.metadata.disallowed_nickase_warning_codes == ["FREQUENT_CUTTER"]
    assert (
        search_report.metadata.blocker_counts["DISALLOWED_NICKASE_WARNING_CODE"]
        == disallowed_nick_placement_count * release_placement_count
    )
    assert not search_report.exact_hits
    assert search_report.near_hits
    assert all(
        hit.pre_nick_site.local_start is not None and hit.pre_nick_site.local_start >= 0
        for hit in search_report.near_hits
    )

    run_dir, solve_report = run_released_snapback_solve(
        request=request,
        output=ReleasedSolveOutputConfig(
            run_dir=Path("outputs/released_solve"),
            materialize_top_k=8,
            render_format="pdf",
            emit_renders=False,
        ),
        workspace_root=workspace_root,
        force_overwrite=True,
    )

    assert run_dir.exists()
    assert solve_report.status == "near_hits_materialized"
    assert solve_report.metadata.materialized_hit_count == min(8, len(search_report.near_hits))
    assert solve_report.metadata.available_exact_hit_count == 0
    assert solve_report.metadata.selected_hit_kind == "nearest"
    assert solve_report.hits[0].nickase_variant_id != "Nt.BspQI"
    assert solve_report.hits[0].release_variant_id == "BsaI-HFv2"
    assert solve_report.hits[0].target_search_hit.sacrificial_downstream_tail_nt == 7
    plot_context = build_released_hit_plot_context(solve_report.hits[0].target_search_hit)
    assert plot_context["precursor"]["nick_site"]["local_start"] >= 0
    assert plot_context["precursor"]["nick_site"]["local_end"] >= 0
    assert (
        plot_context["precursor"]["nick_boundary"]
        == solve_report.hits[0].target_search_hit.pre_nick_event.boundary_context
    )
    assert plot_context["precursor"]["nicked_strand"] == solve_report.hits[0].target_search_hit.physical_nicked_strand
    assert plot_context["released_product"]["retained_partner_span"]["start"] >= 0
    assert plot_context["released_product"]["active_product_span"]["start"] >= 0
    assert (
        plot_context["released_product"]["nicked_strand"]
        == solve_report.hits[0].target_search_hit.physical_nicked_strand
    )
    assert plot_context["released_product"]["duplex_overlap_span"] == {"start": 0, "end": 2}
    assert plot_context["released_product"]["duplex_top_sequence"] == "CC"
    assert plot_context["released_product"]["duplex_bottom_sequence"] == "GG"
    assert plot_context["released_product"]["duplex_mismatch_positions"] == []
    assert plot_context["released_product"]["top_only_overhang_span"] is None
    assert plot_context["released_product"]["bottom_only_overhang_span"] == {
        "start": 2,
        "end": plot_context["released_product"]["bottom_row"]["span"]["end"],
    }
    assert plot_context["foldback"]["foldback_mismatch_positions"] == []
    assert plot_context["foldback"]["nicked_strand"] == solve_report.hits[0].target_search_hit.physical_nicked_strand
    assert (
        plot_context["labels"]["orientation_note"]
        == "Rows stay on physical top/bottom lanes; foldback keeps the active row at origin."
    )
    assert plot_context["labels"]["active_start_terminal"] == "3'"
    assert plot_context["labels"]["active_end_terminal"] == "5'"
    assert plot_context["released_product"]["top_row"]["role"] == "retained_partner"
    assert plot_context["released_product"]["top_row"]["strand"] == "top"
    assert plot_context["released_product"]["top_row"]["sequence"] == "CC"
    assert plot_context["released_product"]["top_row"]["start_terminal"] == "5'"
    assert plot_context["released_product"]["top_row"]["end_terminal"] == "3'"
    assert plot_context["released_product"]["bottom_row"]["role"] == "active_product"
    assert plot_context["released_product"]["bottom_row"]["strand"] == "bottom"
    assert plot_context["released_product"]["bottom_row"]["sequence"] == "GGATTCGTAAT"
    assert plot_context["precursor"]["retained_partner_span"] == {
        "start": 0,
        "end": plot_context["precursor"]["nick_boundary"],
    }
    assert plot_context["precursor"]["active_product_span"] == {
        "start": (
            solve_report.hits[0].target_search_hit.projection.nick_coordinate_precursor
            - solve_report.hits[0].target_search_hit.projection.rebased_nick_boundary
        ),
        "end": (
            solve_report.hits[0].target_search_hit.projection.nick_coordinate_precursor
            - solve_report.hits[0].target_search_hit.projection.rebased_nick_boundary
            + solve_report.hits[0].target_search_hit.projection.active_product_length_nt
        ),
    }
    assert plot_context["foldback"]["top_row"]["role"] == "foldback_return"
    assert plot_context["foldback"]["bottom_row"]["role"] == "active_stem"
    first_context = json.loads(
        (
            workspace_root / solve_report.hits[0].materialized_run_dir / "analysis" / "released_hit_plot_context.json"
        ).read_text(encoding="utf-8")
    )
    assert first_context["precursor"]["nick_site"]["local_start"] >= 0
    assert first_context["precursor"]["nick_site"]["local_end"] >= 0
    assert (
        first_context["precursor"]["nick_boundary"]
        == solve_report.hits[0].target_search_hit.pre_nick_event.boundary_context
    )
    assert first_context["precursor"]["nicked_strand"] == solve_report.hits[0].target_search_hit.physical_nicked_strand
    assert first_context["released_product"]["retained_partner_span"]["start"] >= 0
    assert first_context["released_product"]["active_product_span"]["start"] >= 0
    assert first_context["released_product"]["duplex_overlap_span"] == {"start": 0, "end": 2}
    assert first_context["released_product"]["duplex_top_sequence"] == "CC"
    assert first_context["released_product"]["duplex_bottom_sequence"] == "GG"
    assert first_context["released_product"]["duplex_mismatch_positions"] == []
    assert first_context["released_product"]["top_only_overhang_span"] is None
    assert first_context["released_product"]["bottom_only_overhang_span"] == {
        "start": 2,
        "end": first_context["released_product"]["bottom_row"]["span"]["end"],
    }
    assert first_context["released_product"]["nick_boundary"] >= 0
    assert (
        first_context["released_product"]["nicked_strand"]
        == solve_report.hits[0].target_search_hit.physical_nicked_strand
    )
    assert (
        first_context["released_product"]["retained_partner_span"]["end"]
        == first_context["released_product"]["nick_boundary"]
    )
    assert first_context["released_product"]["nickase_site_survives_post_release"] is False
    assert first_context["labels"]["active_start_terminal"] == "3'"
    assert first_context["labels"]["active_end_terminal"] == "5'"
    assert first_context["released_product"]["top_row"]["role"] == "retained_partner"
    assert first_context["released_product"]["top_row"]["sequence"] == "CC"
    assert first_context["released_product"]["bottom_row"]["role"] == "active_product"
    assert first_context["released_product"]["bottom_row"]["sequence"] == "GGATTCGTAAT"
    assert first_context["precursor"]["retained_partner_span"] == {
        "start": 0,
        "end": first_context["precursor"]["nick_boundary"],
    }
    assert first_context["foldback"]["top_row"]["role"] == "foldback_return"
    assert first_context["foldback"]["bottom_row"]["role"] == "active_stem"
    assert first_context["foldback"]["nicked_strand"] == solve_report.hits[0].target_search_hit.physical_nicked_strand


def test_released_solve_real_presets_materializes_retained_active_hits_with_route_metadata(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspaces" / "de033"
    workspace_root.mkdir(parents=True, exist_ok=True)
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="neb_nicking_v1", additional_presets=["thermo_nicking_v1"]),
        release_sources=ReleaseCatalogSources(preset="type_iis_release_v1"),
        search=ReleasedTargetSearchConfig(
            max_results=16,
            near_boundary_search_limit=8,
            allow_precut_footprint_outside_active_product=True,
            allowed_active_strands=["top", "bottom"],
            allowed_route_families=["bottom_active_from_top_nick", "top_active_from_bottom_nick"],
        ),
    )

    search_report = run_released_snapback_target_search(
        request=request,
        workspace_root=workspace_root,
    )

    assert search_report.status == "exact_hits_found"
    assert search_report.metadata.route_policy_final_geometry_source == "retained_active_strand"
    exact_hits_by_id = {hit.nickase_variant_id: hit for hit in search_report.exact_hits}
    assert {"Nt.BsmAI", "Nt.BstNBI", "Nt.AlwI", "Nb.BsrDI", "Nb.BtsI"}.issubset(exact_hits_by_id)

    run_dir, solve_report = run_released_snapback_solve(
        request=request,
        output=ReleasedSolveOutputConfig(
            run_dir=Path("outputs/released_solve"),
            materialize_top_k=16,
            render_format="pdf",
            emit_renders=False,
        ),
        workspace_root=workspace_root,
        force_overwrite=True,
    )

    assert run_dir.exists()
    assert solve_report.status == "exact_hits_materialized"
    assert solve_report.metadata.route_policy_final_geometry_source == "retained_active_strand"
    assert solve_report.metadata.allowed_active_strands == ["top", "bottom"]
    assert solve_report.metadata.allowed_route_families == [
        "bottom_active_from_top_nick",
        "top_active_from_bottom_nick",
    ]
    assert any(hit.target_search_hit.active_strand == "top" for hit in solve_report.hits)
    top_active_hits = [hit for hit in solve_report.hits if hit.target_search_hit.active_strand == "top"]
    assert any(hit.nickase_variant_id == "Nt.BstNBI" for hit in top_active_hits)
    top_active_overhang_hit = next(
        hit
        for hit in top_active_hits
        if hit.target_search_hit.projection.active_product_length_nt
        > hit.target_search_hit.projection.retained_partner_length_nt
    )
    assert any(
        base.source_constraint == "degenerate_motif_base"
        for hit in top_active_hits
        for base in hit.target_search_hit.projection.active_product_provenance
    )
    top_active_context = build_released_hit_plot_context(top_active_overhang_hit.target_search_hit)
    top_active_coordinate_offset = (
        top_active_overhang_hit.target_search_hit.projection.nick_coordinate_precursor
        - top_active_overhang_hit.target_search_hit.projection.rebased_nick_boundary
    )
    assert (
        top_active_context["precursor"]["nick_boundary"]
        == top_active_overhang_hit.target_search_hit.pre_nick_event.boundary_context
    )
    assert (
        top_active_context["precursor"]["nicked_strand"]
        == top_active_overhang_hit.target_search_hit.physical_nicked_strand
    )
    assert top_active_context["labels"]["active_start_terminal"] == "5'"
    assert top_active_context["labels"]["active_end_terminal"] == "3'"
    assert top_active_context["precursor"]["retained_partner_span"] == {
        "start": 0,
        "end": top_active_context["precursor"]["nick_boundary"],
    }
    assert top_active_context["precursor"]["active_product_span"] == {
        "start": (
            top_active_overhang_hit.target_search_hit.projection.nick_coordinate_precursor
            - top_active_overhang_hit.target_search_hit.projection.rebased_nick_boundary
        ),
        "end": (
            top_active_overhang_hit.target_search_hit.projection.nick_coordinate_precursor
            - top_active_overhang_hit.target_search_hit.projection.rebased_nick_boundary
            + top_active_overhang_hit.target_search_hit.projection.active_product_length_nt
        ),
    }
    assert top_active_context["released_product"]["top_row"]["role"] == "active_product"
    assert top_active_context["released_product"]["top_row"]["strand"] == "top"
    assert (
        top_active_context["released_product"]["nicked_strand"]
        == top_active_overhang_hit.target_search_hit.physical_nicked_strand
    )
    assert top_active_context["released_product"]["bottom_row"]["role"] == "retained_partner"
    assert top_active_context["released_product"]["bottom_row"]["strand"] == "bottom"
    assert (
        top_active_context["released_product"]["bottom_row"]["sequence"]
        == top_active_overhang_hit.target_search_hit.projection.retained_partner_sequence
    )
    assert top_active_context["released_product"]["top_row"]["label"] == "Exposed Top"
    assert top_active_context["released_product"]["bottom_row"]["label"] == "Bottom"
    assert top_active_context["released_product"]["bottom_row"]["span"] == {
        "start": (-top_active_coordinate_offset),
        "end": (
            top_active_overhang_hit.target_search_hit.projection.retained_partner_length_nt
            - top_active_coordinate_offset
        ),
    }
    assert top_active_context["released_product"]["bottom_row"]["start_terminal"] == "3'"
    assert top_active_context["released_product"]["bottom_row"]["end_terminal"] == "5'"
    assert top_active_context["released_product"]["top_row"]["sequence"] == (
        top_active_overhang_hit.target_search_hit.precursor_top_strand[:top_active_coordinate_offset]
        + top_active_overhang_hit.target_search_hit.projection.active_product_sequence
    )
    assert top_active_context["released_product"]["top_row"]["span"] == {
        "start": -top_active_coordinate_offset,
        "end": top_active_overhang_hit.target_search_hit.projection.active_product_length_nt,
    }
    assert top_active_context["released_product"]["top_only_overhang_span"] == {
        "start": 0,
        "end": top_active_context["released_product"]["top_row"]["span"]["end"],
    }
    assert top_active_context["released_product"]["bottom_only_overhang_span"] is None
    assert top_active_context["released_product"]["duplex_overlap_span"] == {
        "start": -top_active_coordinate_offset,
        "end": 0,
    }
    assert (
        top_active_context["released_product"]["duplex_top_sequence"]
        == (top_active_overhang_hit.target_search_hit.precursor_top_strand[:top_active_coordinate_offset])
    )
    assert (
        top_active_context["released_product"]["duplex_bottom_sequence"]
        == (
            complement_sequence(top_active_overhang_hit.target_search_hit.precursor_top_strand)[
                :top_active_coordinate_offset
            ]
        )
    )
    assert top_active_context["released_product"]["duplex_mismatch_positions"] == []
    assert top_active_context["foldback"]["top_row"]["role"] == "active_stem"
    assert top_active_context["foldback"]["bottom_row"]["role"] == "foldback_return"
    assert (
        top_active_context["foldback"]["nicked_strand"]
        == top_active_overhang_hit.target_search_hit.physical_nicked_strand
    )
    assert top_active_context["foldback"]["top_row"]["label"] == "Stem"
    assert top_active_context["foldback"]["bottom_row"]["label"] == "Foldback Stem"
    assert top_active_context["foldback"]["top_row"]["span"] == {
        "start": -top_active_coordinate_offset,
        "end": top_active_overhang_hit.target_search_hit.final_candidate.paired_bp,
    }
    assert top_active_context["foldback"]["bottom_row"]["span"] == {
        "start": -top_active_coordinate_offset,
        "end": top_active_overhang_hit.target_search_hit.final_candidate.paired_bp,
    }
    assert top_active_context["foldback"]["top_row"]["sequence"] == (
        top_active_overhang_hit.target_search_hit.precursor_top_strand[:top_active_coordinate_offset]
        + top_active_context["foldback"]["stem_sequence"]
    )
    assert top_active_context["foldback"]["bottom_row"]["sequence"] == (
        complement_sequence(top_active_overhang_hit.target_search_hit.precursor_top_strand)[
            :top_active_coordinate_offset
        ]
        + top_active_context["foldback"]["foldback_partner_sequence"]
    )
    invalid_offset_hit = top_active_overhang_hit.target_search_hit.model_copy(
        update={
            "projection": top_active_overhang_hit.target_search_hit.projection.model_copy(
                update={
                    "rebased_nick_boundary": (
                        top_active_overhang_hit.target_search_hit.projection.nick_coordinate_precursor + 1
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="nonnegative precursor nick offset"):
        build_released_hit_plot_context(invalid_offset_hit)

    rendered_top_active_path = workspace_root / "top_active_triptych.png"
    rendered_top_active_context = render_released_hit_plot(
        top_active_overhang_hit.target_search_hit, rendered_top_active_path
    )
    rendered_bytes = rendered_top_active_path.read_bytes()
    assert rendered_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", rendered_bytes[16:24])
    assert width > height
    assert rendered_top_active_context["released_product"]["top_row"]["role"] == "active_product"
    assert (
        rendered_top_active_context["released_product"]["nicked_strand"]
        == top_active_overhang_hit.target_search_hit.physical_nicked_strand
    )
    assert (
        rendered_top_active_context["released_product"]["bottom_row"]["sequence"]
        == top_active_overhang_hit.target_search_hit.projection.retained_partner_sequence
    )
    assert rendered_top_active_context["released_product"]["top_row"]["label"] == "Exposed Top"
    assert rendered_top_active_context["released_product"]["bottom_row"]["label"] == "Bottom"
    assert rendered_top_active_context["released_product"]["bottom_row"]["span"] == {
        "start": (-top_active_coordinate_offset),
        "end": (
            top_active_overhang_hit.target_search_hit.projection.retained_partner_length_nt
            - top_active_coordinate_offset
        ),
    }
    assert rendered_top_active_context["released_product"]["top_row"]["sequence"] == (
        top_active_overhang_hit.target_search_hit.precursor_top_strand[:top_active_coordinate_offset]
        + top_active_overhang_hit.target_search_hit.projection.active_product_sequence
    )
    assert rendered_top_active_context["released_product"]["top_only_overhang_span"] == {
        "start": 0,
        "end": rendered_top_active_context["released_product"]["top_row"]["span"]["end"],
    }
    assert rendered_top_active_context["released_product"]["bottom_only_overhang_span"] is None
    assert rendered_top_active_context["released_product"]["duplex_overlap_span"] == {
        "start": -top_active_coordinate_offset,
        "end": 0,
    }
    assert rendered_top_active_context["foldback"]["top_row"]["label"] == "Stem"
    assert rendered_top_active_context["foldback"]["bottom_row"]["label"] == "Foldback Stem"
    assert (
        rendered_top_active_context["foldback"]["nicked_strand"]
        == top_active_overhang_hit.target_search_hit.physical_nicked_strand
    )
    assert rendered_top_active_context["foldback"]["top_row"]["sequence"] == (
        top_active_overhang_hit.target_search_hit.precursor_top_strand[:top_active_coordinate_offset]
        + rendered_top_active_context["foldback"]["stem_sequence"]
    )
    assert rendered_top_active_context["foldback"]["bottom_row"]["sequence"] == (
        complement_sequence(top_active_overhang_hit.target_search_hit.precursor_top_strand)[
            :top_active_coordinate_offset
        ]
        + rendered_top_active_context["foldback"]["foldback_partner_sequence"]
    )

    with (run_dir / "export" / "table__hits.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["final_geometry_source"] in {"exposed_bottom_strand", "retained_active_strand"}
    assert rows[0]["route_family"]
    assert rows[0]["active_strand"] in {"top", "bottom"}
    assert rows[0]["retained_partner_strand"] in {"top", "bottom"}
    assert rows[0]["physical_nicked_strand"] in {"top", "bottom"}
    assert rows[0]["active_product_input_length_nt"]
    assert rows[0]["active_product_length_nt"]
    assert rows[0]["retained_partner_length_nt"]
