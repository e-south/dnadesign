"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/released_snapback/test_target_search.py

Target-search tests for released-product snapback workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import dnadesign.cruncher.snapback.released_target_search as released_target_search
from dnadesign.cruncher.app.snapback_released_target_search_workflow import run_released_snapback_target_search
from dnadesign.cruncher.nickases.catalog import load_merged_nickase_catalog
from dnadesign.cruncher.nickases.models import NickaseCatalog, NickaseCatalogEntry
from dnadesign.cruncher.nickases.selection import matching_nickase_warning_codes
from dnadesign.cruncher.release_enzymes.catalog import load_merged_release_enzyme_catalog
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeCatalog, ReleaseEnzymeEntry
from dnadesign.cruncher.snapback.models import CatalogNormalizationInfo, CatalogSources
from dnadesign.cruncher.snapback.released_models import (
    ReleaseCatalogNormalizationInfo,
    ReleaseCatalogSources,
    ReleasedFinalCandidate,
    ReleasedFinalTargetGeometry,
    ReleasedProductProjection,
    ReleasedTargetSearchConfig,
    ReleasedTargetSearchHit,
    SingleNickReleasedTargetSearchRequest,
)
from dnadesign.cruncher.snapback.released_search.placement_models import (
    NickPlacement,
    ReleasePlacement,
)


def _write_nick_catalog(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nx.Exact7",
                            "specificity_id": "Nx.Exact7",
                            "motif_top_5to3": "AACGTTG",
                            "top_cut_offset": 0,
                        },
                        {
                            "id": "Nx.ExactAlt7",
                            "specificity_id": "Nx.ExactAlt7",
                            "motif_top_5to3": "AAAGTTT",
                            "top_cut_offset": 0,
                        },
                        {
                            "id": "Nx.Near7",
                            "specificity_id": "Nx.Near7",
                            "motif_top_5to3": "TAACGTT",
                            "top_cut_offset": 1,
                        },
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_release_catalog(path: Path) -> None:
    path.write_text(
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
                        },
                        {
                            "variant_id": "Re.Overlap",
                            "display_name": "Re.Overlap",
                            "recognition_sequence": "GGGG",
                            "top_cut_offset": 12,
                            "bottom_cut_offset": 13,
                            "class_label": "other_ds_re",
                            "commercial_confidence": "primary_vendor_current",
                            "source_catalog_id": "local_release",
                        },
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_build_precursor_sequence_rejects_left_of_origin_outside_site_nickase() -> None:
    target = ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3)
    nick_placement = NickPlacement(
        entry=NickaseCatalogEntry(
            id="Nt.BsmAI",
            specificity_id="BsmAI",
            motif_top_5to3="GTCTC",
            top_cut_offset=6,
        ),
        orientation="forward",
        motif="GTCTC",
        site_start_at_boundary_zero=-6,
    )
    release_placement = ReleasePlacement(
        entry=ReleaseEnzymeEntry(
            variant_id="Re.Test",
            display_name="Re.Test",
            recognition_sequence="CCAA",
            top_cut_offset=1,
            bottom_cut_offset=0,
            class_label="other_ds_re",
            commercial_confidence="primary_vendor_current",
            source_catalog_id="test_release",
        ),
        orientation="forward",
        motif="CCAA",
        site_shift_from_boundary=9,
        top_cut_shift_from_boundary=10,
        bottom_cut_shift_from_boundary=9,
    )

    built = released_target_search._build_precursor_sequence(
        boundary=0,
        target=target,
        nick_placement=nick_placement,
        release_placement=release_placement,
    )

    assert built.precursor is None
    assert built.blocker_code == "PRE_NICK_SITE_LEFT_OF_ORIGIN"


def test_nick_placements_allow_left_of_origin_only_for_contiguous_leading_degenerate_prefix() -> None:
    catalog = NickaseCatalog(
        preset_id="local",
        preset_ids=["local"],
        entries=[
            NickaseCatalogEntry(
                id="Nx.DegeneratePrefix",
                specificity_id="Nx.DegeneratePrefix",
                motif_top_5to3="NNAA",
                top_cut_offset=4,
            )
        ],
    )

    placements = released_target_search._nick_placements(catalog, physical_nicked_strand="top")

    assert len(placements) == 1
    placement = placements[0]
    assert placement.orientation == "forward"
    assert placement.site_start_at_boundary_zero == -4
    assert placement.left_of_origin_slack_nt == 2
    assert placement.earliest_allowed_boundary() == 2
    assert placement.allows_left_of_origin_prefix(2) is True
    assert placement.allows_left_of_origin_prefix(1) is False


def test_nick_placements_apply_left_of_origin_slack_in_reverse_complemented_bottom_nicker_view() -> None:
    catalog = NickaseCatalog(
        preset_id="local",
        preset_ids=["local"],
        entries=[
            NickaseCatalogEntry(
                id="Nb.ReverseDegeneratePrefix",
                specificity_id="Nb.ReverseDegeneratePrefix",
                motif_top_5to3="GGNN",
                bottom_cut_offset=0,
            )
        ],
    )

    placements = released_target_search._nick_placements(catalog, physical_nicked_strand="top")

    assert len(placements) == 1
    placement = placements[0]
    assert placement.orientation == "reverse"
    assert placement.motif == "NNCC"
    assert placement.site_start_at_boundary_zero == -4
    assert placement.left_of_origin_slack_nt == 2
    assert placement.earliest_allowed_boundary() == 2


def test_build_precursor_sequence_allows_left_of_origin_when_truncated_prefix_is_all_degenerate() -> None:
    target = ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3)
    nick_placement = NickPlacement(
        entry=NickaseCatalogEntry(
            id="Nx.DegeneratePrefix",
            specificity_id="Nx.DegeneratePrefix",
            motif_top_5to3="NNAA",
            top_cut_offset=4,
        ),
        orientation="forward",
        motif="NNAA",
        site_start_at_boundary_zero=-4,
        left_of_origin_slack_nt=2,
    )
    release_placement = ReleasePlacement(
        entry=ReleaseEnzymeEntry(
            variant_id="Re.Test",
            display_name="Re.Test",
            recognition_sequence="CCAA",
            top_cut_offset=1,
            bottom_cut_offset=0,
            class_label="other_ds_re",
            commercial_confidence="primary_vendor_current",
            source_catalog_id="test_release",
        ),
        orientation="forward",
        motif="CCAA",
        site_shift_from_boundary=9,
        top_cut_shift_from_boundary=10,
        bottom_cut_shift_from_boundary=9,
    )

    built = released_target_search._build_precursor_sequence(
        boundary=2,
        target=target,
        nick_placement=nick_placement,
        release_placement=release_placement,
    )

    assert built.precursor is not None
    assert built.blocker_code is None
    assert built.precursor.coordinate_offset == 0
    assert built.precursor.top_strand.startswith("AA")


def test_builtin_neb_candidates_do_not_gain_origin_slack_without_leading_degenerate_prefix(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspaces" / "released_snapback"
    workspace.mkdir(parents=True, exist_ok=True)
    catalog, _resolved_paths = load_merged_nickase_catalog(
        preset_id="neb_nicking_v1",
        additional_paths=[],
        workspace_root=workspace,
    )

    placements = {
        placement.entry.id: placement
        for placement in released_target_search._nick_placements(
            catalog,
            physical_nicked_strand="top",
        )
        if placement.entry.id in {"Nb.BsrDI", "Nb.BtsI", "Nt.AlwI", "Nt.BsmAI", "Nt.BspQI", "Nt.BstNBI"}
    }

    assert {key for key in placements} == {"Nb.BsrDI", "Nb.BtsI", "Nt.AlwI", "Nt.BsmAI", "Nt.BspQI", "Nt.BstNBI"}
    assert placements["Nb.BsrDI"].orientation == "reverse"
    assert placements["Nb.BtsI"].orientation == "reverse"
    assert placements["Nt.AlwI"].orientation == "forward"
    assert placements["Nt.BsmAI"].orientation == "forward"
    assert placements["Nt.BspQI"].orientation == "forward"
    assert placements["Nt.BstNBI"].orientation == "forward"
    assert all(placement.left_of_origin_slack_nt == 0 for placement in placements.values())
    assert placements["Nb.BsrDI"].earliest_allowed_boundary() == 0
    assert placements["Nb.BtsI"].earliest_allowed_boundary() == 0
    assert placements["Nt.BsmAI"].earliest_allowed_boundary() == 6
    assert placements["Nt.BspQI"].earliest_allowed_boundary() == 8
    assert placements["Nt.AlwI"].earliest_allowed_boundary() == 9
    assert placements["Nt.BstNBI"].earliest_allowed_boundary() == 9

    vendor_placements = {
        placement.entry.id: placement
        for placement in released_target_search._nick_placements(
            catalog,
            physical_nicked_strand="top",
            use_vendor_diagram=True,
        )
        if placement.entry.id in {"Nb.BsrDI", "Nb.BtsI"}
    }
    assert vendor_placements["Nb.BsrDI"].motif == "NNCATTGC"
    assert vendor_placements["Nb.BtsI"].motif == "NNCACTGC"
    assert vendor_placements["Nb.BsrDI"].site_start_at_boundary_zero == -2
    assert vendor_placements["Nb.BtsI"].site_start_at_boundary_zero == -2
    assert vendor_placements["Nb.BsrDI"].left_of_origin_slack_nt == 2
    assert vendor_placements["Nb.BtsI"].left_of_origin_slack_nt == 2
    assert vendor_placements["Nb.BsrDI"].earliest_allowed_boundary() == 0
    assert vendor_placements["Nb.BtsI"].earliest_allowed_boundary() == 0


def test_released_target_search_reports_exact_hits_near_hits_blockers_and_pre_post_truncation_counts(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspaces" / "demo_released"
    nick_catalog_path = workspace_root / "inputs" / "nickases" / "local.nickases.yaml"
    release_catalog_path = workspace_root / "inputs" / "release_enzymes" / "local.release.yaml"
    nick_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    release_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    _write_nick_catalog(nick_catalog_path)
    _write_release_catalog(release_catalog_path)

    report = run_released_snapback_target_search(
        request=SingleNickReleasedTargetSearchRequest(
            target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
            nick_sources=CatalogSources(additional_paths=[Path("inputs/nickases/local.nickases.yaml")]),
            release_sources=ReleaseCatalogSources(additional_paths=[Path("inputs/release_enzymes/local.release.yaml")]),
            search=ReleasedTargetSearchConfig(max_results=1, near_boundary_search_limit=3),
        ),
        workspace_root=workspace_root,
    )

    assert report.status == "exact_hits_found"
    assert report.metadata.pre_truncation_exact_hit_count == 2
    assert report.metadata.post_truncation_exact_hit_count == 1
    assert report.metadata.pre_truncation_near_hit_count == 9
    assert report.metadata.post_truncation_near_hit_count == 1
    assert report.exact_hits[0].nickase_variant_id in {"Nx.Exact7", "Nx.ExactAlt7", "Nx.Near7"}
    assert report.exact_hits[0].release_variant_id in {"Re.Exact", "Re.Overlap"}
    assert report.exact_hits[0].nick_boundary_from_left == 0
    assert report.exact_hits[0].active_product_input_length_nt == 6
    assert report.near_hits[0].hit_kind == "nearest"
    assert report.near_hits[0].nick_boundary_from_left == 1
    assert report.near_hits[0].upstream_retained_duplex_bp == 1
    assert report.near_hits[0].effective_stem_bp == 4
    assert report.metadata.blocker_counts["RELEASE_OVERLAPS_REQUIRED_TARGET_REGION"] >= 1


def test_released_target_search_can_materialize_top_active_routes_with_vendor_footprints(
    tmp_path: Path,
) -> None:
    report = released_target_search.search_released_target_hits(
        request=SingleNickReleasedTargetSearchRequest(
            target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
            nick_sources=CatalogSources(preset="local"),
            release_sources=ReleaseCatalogSources(preset="local"),
            search=ReleasedTargetSearchConfig(
                max_results=4,
                near_boundary_search_limit=0,
                disallowed_nickase_warning_codes=[],
                allow_precut_footprint_outside_active_product=True,
                allowed_active_strands=["top", "bottom"],
                allowed_route_families=["top_active_from_bottom_nick"],
            ),
        ),
        nick_catalog=NickaseCatalog(
            preset_id="local",
            preset_ids=["local"],
            entries=[
                NickaseCatalogEntry(
                    id="Nt.BsmAI",
                    specificity_id="Nt.BsmAI",
                    motif_top_5to3="GTCTC",
                    vendor_diagram_top_5to3="GTCTCNN",
                    top_cut_offset=6,
                )
            ],
        ),
        release_catalog=ReleaseEnzymeCatalog(
            preset_id="local",
            preset_ids=["local"],
            entries=[
                ReleaseEnzymeEntry(
                    variant_id="Re.Exact",
                    display_name="Re.Exact",
                    recognition_sequence="CCAA",
                    top_cut_offset=1,
                    bottom_cut_offset=0,
                    class_label="other_ds_re",
                    commercial_confidence="primary_vendor_current",
                    source_catalog_id="local_release",
                )
            ],
        ),
        workspace_root=tmp_path,
        nick_catalog_source="local",
        release_catalog_source="local",
    )

    assert report.status == "exact_hits_found"
    assert report.metadata.route_policy_final_geometry_source == "retained_active_strand"
    assert report.metadata.allowed_active_strands == ["top", "bottom"]
    assert report.metadata.allowed_route_families == ["top_active_from_bottom_nick"]
    assert report.exact_hits
    hit = report.exact_hits[0]
    assert hit.route_family == "top_active_from_bottom_nick"
    assert hit.active_strand == "top"
    assert hit.physical_nicked_strand == "bottom"
    assert hit.pre_nick_site.orientation == "reverse"
    assert hit.projection.final_geometry_source == "retained_active_strand"
    assert hit.projection.active_product_length_nt == 9
    assert any(base.source_constraint == "degenerate_motif_base" for base in hit.projection.active_product_provenance)


def test_released_target_search_real_presets_find_expected_retained_active_routes(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspaces" / "released_snapback"
    workspace_root.mkdir(parents=True, exist_ok=True)

    report = run_released_snapback_target_search(
        request=SingleNickReleasedTargetSearchRequest(
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
        ),
        workspace_root=workspace_root,
    )

    assert report.status == "exact_hits_found"
    assert report.metadata.route_policy_final_geometry_source == "retained_active_strand"
    exact_hits_by_id = {hit.nickase_variant_id: hit for hit in report.exact_hits}
    assert {"Nt.BsmAI", "Nt.BstNBI", "Nt.AlwI", "Nb.BsrDI", "Nb.BtsI"}.issubset(exact_hits_by_id)
    assert exact_hits_by_id["Nt.BstNBI"].route_family == "top_active_from_bottom_nick"
    assert exact_hits_by_id["Nt.BstNBI"].active_strand == "top"
    assert exact_hits_by_id["Nt.BstNBI"].physical_nicked_strand == "bottom"
    assert any(
        base.source_constraint == "degenerate_motif_base"
        for base in exact_hits_by_id["Nt.BstNBI"].projection.active_product_provenance
    )
    assert exact_hits_by_id["Nt.AlwI"].route_family == "top_active_from_bottom_nick"
    assert exact_hits_by_id["Nt.AlwI"].active_strand == "top"
    assert exact_hits_by_id["Nt.AlwI"].physical_nicked_strand == "bottom"
    assert exact_hits_by_id["Nb.BsrDI"].route_family == "bottom_active_from_top_nick"
    assert exact_hits_by_id["Nb.BsrDI"].active_strand == "bottom"
    assert exact_hits_by_id["Nb.BsrDI"].physical_nicked_strand == "top"
    assert exact_hits_by_id["Nb.BsrDI"].pre_nick_site.local_start == -2
    assert exact_hits_by_id["Nb.BtsI"].route_family == "bottom_active_from_top_nick"
    assert exact_hits_by_id["Nb.BtsI"].active_strand == "bottom"
    assert exact_hits_by_id["Nb.BtsI"].physical_nicked_strand == "top"
    assert exact_hits_by_id["Nb.BtsI"].pre_nick_site.local_start == -2


def test_released_target_search_can_pin_release_variant_for_bspqi_policy(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspaces" / "released_snapback"
    workspace_root.mkdir(parents=True, exist_ok=True)

    report = run_released_snapback_target_search(
        request=SingleNickReleasedTargetSearchRequest(
            target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
            nick_sources=CatalogSources(preset="neb_nicking_v1", additional_presets=["thermo_nicking_v1"]),
            release_sources=ReleaseCatalogSources(preset="type_iis_release_v1"),
            search=ReleasedTargetSearchConfig(
                max_results=16,
                near_boundary_search_limit=8,
                allow_precut_footprint_outside_active_product=True,
                allowed_active_strands=["top", "bottom"],
                allowed_route_families=["bottom_active_from_top_nick", "top_active_from_bottom_nick"],
                allowed_release_variant_ids=["BspQI"],
            ),
        ),
        workspace_root=workspace_root,
    )

    assert report.status == "exact_hits_found"
    assert report.metadata.allowed_release_variant_ids == ["BspQI"]
    assert report.exact_hits
    assert {hit.release_variant_id for hit in report.exact_hits} == {"BspQI"}
    assert {hit.nickase_variant_id for hit in report.exact_hits} == {
        "Nt.BstNBI",
        "Nt.AlwI",
        "Nt.BsmAI",
        "Nb.BsrDI",
        "Nb.BtsI",
    }
    assert all(
        hit.pre_nick_site.local_start is None or hit.pre_nick_site.local_start >= -2 for hit in report.exact_hits
    )
    assert {
        hit.nickase_variant_id: hit.pre_nick_site.local_start
        for hit in report.exact_hits
        if hit.nickase_variant_id in {"Nb.BsrDI", "Nb.BtsI"}
    } == {"Nb.BsrDI": -2, "Nb.BtsI": -2}
    assert all(hit.pre_nick_site.local_end is None or hit.pre_nick_site.local_end > 0 for hit in report.exact_hits)
    assert "RELEASE_VARIANT_FILTERED" in report.metadata.blocker_counts


def test_rank_hits_ignores_gc_when_other_exact_rank_inputs_match() -> None:
    def _exact_hit(*, variant_id: str, designed_sequence: str) -> ReleasedTargetSearchHit:
        return ReleasedTargetSearchHit.model_construct(
            rank=1,
            hit_kind="exact",
            nickase_variant_id=variant_id,
            release_variant_id="Re.Exact",
            intended_nick_site_orientation="forward",
            intended_nick_site_sequence="AAAA",
            release_site_orientation="forward",
            release_site_sequence="TTTT",
            nick_boundary_from_left=0,
            active_product_input_length_nt=6,
            active_product_length_nt=9,
            precursor_length_nt=12,
            sacrificial_downstream_tail_nt=3,
            extra_nick_event_count=0,
            extra_target_strand_nick_count=0,
            precursor_top_strand="AAAATTTTAAAA",
            nickase=CatalogNormalizationInfo.model_construct(
                variant_id=variant_id,
                specificity_id=variant_id,
                motif_top_5to3="AAAA",
                motif_len=4,
                nicked_strand="top",
                active_cut_offset=0,
            ),
            release_enzyme=ReleaseCatalogNormalizationInfo.model_construct(
                variant_id="Re.Exact",
                display_name="Re.Exact",
                recognition_sequence="TTTT",
                recognition_len=4,
                top_cut_offset=0,
                bottom_cut_offset=0,
                class_label="other_ds_re",
                outside_site=False,
                commercial_confidence="primary_vendor_current",
                source_catalog_id="test_release",
            ),
            projection=ReleasedProductProjection.model_construct(
                final_geometry_source="exposed_bottom_strand",
                precursor_top_strand="AAAATTTTAAAA",
                precursor_length=12,
                nick_coordinate_precursor=0,
                release_top_cut_precursor=9,
                release_bottom_cut_precursor=9,
                retained_partner_sequence="AAAATTTTA",
                retained_partner_length_nt=9,
                active_product_sequence=designed_sequence,
                active_product_span=(0, 9),
                active_product_length_nt=9,
                rebased_nick_boundary=0,
                nickase_site_survives_post_release=True,
                release_site_survives_post_release=False,
            ),
            final_candidate=ReleasedFinalCandidate.model_construct(
                final_geometry_source="exposed_bottom_strand",
                designed_sequence=designed_sequence,
                input_sequence=designed_sequence[:6],
                foldback_arm=designed_sequence[6:],
                nick_boundary_from_left=0,
                paired_bp=3,
                cap_nt=3,
                source_cap_nt=3,
                cap_extension_nt=0,
                active_product_length_nt=9,
                active_product_input_length_nt=6,
                mismatch_count=0,
                mismatch_positions=[],
                terminal_ligatable_duplex_bp=3,
                max_uninterrupted_duplex_bp=3,
                extra_nick_event_count=0,
                extra_target_strand_nick_count=0,
                gc_fraction_added=0.0,
                max_homopolymer_run_added=4,
                projected_origin_event=None,
                extra_target_strand_nicks=[],
                extra_nick_events=[],
                post_nick_sequence=designed_sequence,
                nickase_site_survives_post_release=True,
                release_site_survives_post_release=False,
            ),
        )

    low_gc = _exact_hit(variant_id="Nx.ALowGC", designed_sequence="ATTCGTAAT")
    zero_gc = _exact_hit(variant_id="Nx.BZeroGC", designed_sequence="TTTTTTAAA")

    ranked = released_target_search._rank_hits(
        [zero_gc, low_gc],
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        exact=True,
    )

    assert [hit.nickase_variant_id for hit in ranked] == ["Nx.ALowGC", "Nx.BZeroGC"]


def test_rank_hits_deduplicates_exact_hits_by_post_nick_stem_and_cap() -> None:
    def _exact_hit(*, variant_id: str, designed_sequence: str) -> ReleasedTargetSearchHit:
        return ReleasedTargetSearchHit.model_construct(
            rank=1,
            hit_kind="exact",
            nickase_variant_id=variant_id,
            release_variant_id="Re.Exact",
            intended_nick_site_orientation="forward",
            intended_nick_site_sequence="AAAA",
            release_site_orientation="forward",
            release_site_sequence="TTTT",
            nick_boundary_from_left=0,
            active_product_input_length_nt=6,
            active_product_length_nt=9,
            precursor_length_nt=12,
            sacrificial_downstream_tail_nt=3,
            extra_nick_event_count=0,
            extra_target_strand_nick_count=0,
            precursor_top_strand="AAAATTTTAAAA",
            nickase=CatalogNormalizationInfo.model_construct(
                variant_id=variant_id,
                specificity_id=variant_id,
                motif_top_5to3="AAAA",
                motif_len=4,
                nicked_strand="top",
                active_cut_offset=0,
            ),
            release_enzyme=ReleaseCatalogNormalizationInfo.model_construct(
                variant_id="Re.Exact",
                display_name="Re.Exact",
                recognition_sequence="TTTT",
                recognition_len=4,
                top_cut_offset=0,
                bottom_cut_offset=0,
                class_label="other_ds_re",
                outside_site=False,
                commercial_confidence="primary_vendor_current",
                source_catalog_id="test_release",
            ),
            projection=ReleasedProductProjection.model_construct(
                final_geometry_source="exposed_bottom_strand",
                precursor_top_strand="AAAATTTTAAAA",
                precursor_length=12,
                nick_coordinate_precursor=0,
                release_top_cut_precursor=9,
                release_bottom_cut_precursor=9,
                retained_partner_sequence="AAAATTTTA",
                retained_partner_length_nt=9,
                active_product_sequence=designed_sequence,
                active_product_span=(0, 9),
                active_product_length_nt=9,
                rebased_nick_boundary=0,
                nickase_site_survives_post_release=True,
                release_site_survives_post_release=False,
            ),
            final_candidate=ReleasedFinalCandidate.model_construct(
                final_geometry_source="exposed_bottom_strand",
                designed_sequence=designed_sequence,
                input_sequence=designed_sequence[:6],
                foldback_arm=designed_sequence[6:],
                nick_boundary_from_left=0,
                paired_bp=3,
                cap_nt=3,
                source_cap_nt=3,
                cap_extension_nt=0,
                active_product_length_nt=9,
                active_product_input_length_nt=6,
                mismatch_count=0,
                mismatch_positions=[],
                terminal_ligatable_duplex_bp=3,
                max_uninterrupted_duplex_bp=3,
                extra_nick_event_count=0,
                extra_target_strand_nick_count=0,
                gc_fraction_added=0.0,
                max_homopolymer_run_added=4,
                projected_origin_event=None,
                extra_target_strand_nicks=[],
                extra_nick_events=[],
                post_nick_sequence=designed_sequence,
                nickase_site_survives_post_release=True,
                release_site_survives_post_release=False,
            ),
        )

    stem_cap_a = _exact_hit(variant_id="Nx.A", designed_sequence="TTTTTTAAA")
    stem_cap_a_duplicate = _exact_hit(variant_id="Nx.B", designed_sequence="TTTTTTCCC")
    stem_cap_b = _exact_hit(variant_id="Nx.C", designed_sequence="ATTCGTAAT")
    policy = released_target_search.ReleasedRankingPolicy(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3)
    )

    assert policy.dedupe_key(stem_cap_a) == policy.dedupe_key(stem_cap_a_duplicate)
    assert policy.dedupe_key(stem_cap_a) != policy.dedupe_key(stem_cap_b)

    ranked = policy.rank_hits([stem_cap_b, stem_cap_a_duplicate, stem_cap_a], exact=True)

    assert [hit.nickase_variant_id for hit in ranked] == ["Nx.A", "Nx.C"]


def test_search_pair_collects_all_near_hits_within_bounded_window(monkeypatch: pytest.MonkeyPatch) -> None:
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=2, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="test_nick"),
        release_sources=ReleaseCatalogSources(preset="test_release"),
        search=ReleasedTargetSearchConfig(near_boundary_search_limit=2),
    )
    nick_placement = NickPlacement(
        entry=NickaseCatalogEntry(
            id="Nx.Test",
            specificity_id="Nx.Test",
            motif_top_5to3="AAAA",
            top_cut_offset=0,
        ),
        orientation="forward",
        motif="AAAA",
        site_start_at_boundary_zero=0,
    )
    release_placement = ReleasePlacement(
        entry=ReleaseEnzymeEntry(
            variant_id="Re.Test",
            display_name="Re.Test",
            recognition_sequence="TTTT",
            top_cut_offset=0,
            bottom_cut_offset=0,
            class_label="other_ds_re",
            commercial_confidence="primary_vendor_current",
            source_catalog_id="test_release",
        ),
        orientation="forward",
        motif="TTTT",
        site_shift_from_boundary=0,
        top_cut_shift_from_boundary=9,
        bottom_cut_shift_from_boundary=9,
    )

    def fake_build_precursor_sequence(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            precursor=SimpleNamespace(top_strand="A" * 12, coordinate_offset=0),
            blocker_code=None,
        )

    def fake_evaluate_released_precursor(*, target: ReleasedFinalTargetGeometry, **_: object) -> SimpleNamespace:
        status = "satisfied" if target.nick_boundary_from_left in {1, 3, 4} else "unsatisfied"
        return SimpleNamespace(
            status=status,
            issues=[],
            candidate=object(),
            projection=object(),
            pre_nick_match=object(),
            release_match=object(),
        )

    def fake_hit_from_evaluation(*, boundary: int, hit_kind: str, **_: object) -> tuple[int, str]:
        return (boundary, hit_kind)

    monkeypatch.setattr(released_target_search, "_build_precursor_sequence", fake_build_precursor_sequence)
    monkeypatch.setattr(released_target_search, "evaluate_released_precursor", fake_evaluate_released_precursor)
    monkeypatch.setattr(released_target_search, "_hit_from_evaluation", fake_hit_from_evaluation)

    exact_hit, near_hits = released_target_search._search_pair(
        request=request,
        nick_placement=nick_placement,
        release_placement=release_placement,
        blocker_counts={},
    )

    assert exact_hit is None
    assert near_hits == [(1, "nearest"), (3, "nearest"), (4, "nearest")]


def test_search_pair_keeps_same_pair_near_hits_when_exact_hit_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=2, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="test_nick"),
        release_sources=ReleaseCatalogSources(preset="test_release"),
        search=ReleasedTargetSearchConfig(near_boundary_search_limit=2),
    )
    nick_placement = NickPlacement(
        entry=NickaseCatalogEntry(
            id="Nx.Test",
            specificity_id="Nx.Test",
            motif_top_5to3="AAAA",
            top_cut_offset=0,
        ),
        orientation="forward",
        motif="AAAA",
        site_start_at_boundary_zero=0,
    )
    release_placement = ReleasePlacement(
        entry=ReleaseEnzymeEntry(
            variant_id="Re.Test",
            display_name="Re.Test",
            recognition_sequence="TTTT",
            top_cut_offset=0,
            bottom_cut_offset=0,
            class_label="other_ds_re",
            commercial_confidence="primary_vendor_current",
            source_catalog_id="test_release",
        ),
        orientation="forward",
        motif="TTTT",
        site_shift_from_boundary=0,
        top_cut_shift_from_boundary=9,
        bottom_cut_shift_from_boundary=9,
    )

    def fake_build_precursor_sequence(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            precursor=SimpleNamespace(top_strand="A" * 12, coordinate_offset=0),
            blocker_code=None,
        )

    def fake_evaluate_released_precursor(*, target: ReleasedFinalTargetGeometry, **_: object) -> SimpleNamespace:
        status = "satisfied" if target.nick_boundary_from_left in {2, 3, 4} else "unsatisfied"
        return SimpleNamespace(
            status=status,
            issues=[],
            candidate=object(),
            projection=object(),
            pre_nick_match=object(),
            release_match=object(),
        )

    def fake_hit_from_evaluation(*, boundary: int, hit_kind: str, **_: object) -> tuple[int, str]:
        return (boundary, hit_kind)

    monkeypatch.setattr(released_target_search, "_build_precursor_sequence", fake_build_precursor_sequence)
    monkeypatch.setattr(released_target_search, "evaluate_released_precursor", fake_evaluate_released_precursor)
    monkeypatch.setattr(released_target_search, "_hit_from_evaluation", fake_hit_from_evaluation)

    exact_hit, near_hits = released_target_search._search_pair(
        request=request,
        nick_placement=nick_placement,
        release_placement=release_placement,
        blocker_counts={},
    )

    assert exact_hit == (2, "exact")
    assert near_hits == [(3, "nearest"), (4, "nearest")]


def test_search_released_target_hits_keeps_near_hits_when_pair_has_exact_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=2, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="test_nick"),
        release_sources=ReleaseCatalogSources(preset="test_release"),
        search=ReleasedTargetSearchConfig(max_results=8, near_boundary_search_limit=2),
    )
    exact_hit = ReleasedTargetSearchHit.model_construct(
        rank=1,
        hit_kind="exact",
        nickase_variant_id="Nx.Exact",
        release_variant_id="Re.Exact",
        intended_nick_site_orientation="forward",
        intended_nick_site_sequence="AAAA",
        release_site_orientation="forward",
        release_site_sequence="TTTT",
        nick_boundary_from_left=2,
        active_product_input_length_nt=8,
        active_product_length_nt=11,
        precursor_length_nt=12,
        sacrificial_downstream_tail_nt=1,
        extra_nick_event_count=0,
        extra_target_strand_nick_count=0,
        precursor_top_strand="AAAAAAAAAAAA",
        nickase=CatalogNormalizationInfo.model_construct(
            variant_id="Nx.Exact",
            specificity_id="Nx.Exact",
            motif_top_5to3="AAAA",
            motif_len=4,
            nicked_strand="top",
            active_cut_offset=0,
        ),
        release_enzyme=ReleaseCatalogNormalizationInfo.model_construct(
            variant_id="Re.Exact",
            display_name="Re.Exact",
            recognition_sequence="TTTT",
            recognition_len=4,
            top_cut_offset=0,
            bottom_cut_offset=0,
            class_label="other_ds_re",
            outside_site=False,
            commercial_confidence="primary_vendor_current",
            source_catalog_id="test_release",
        ),
        projection=ReleasedProductProjection.model_construct(
            final_geometry_source="exposed_bottom_strand",
            precursor_top_strand="AAAAAAAAAAAA",
            precursor_length=12,
            nick_coordinate_precursor=2,
            release_top_cut_precursor=11,
            release_bottom_cut_precursor=11,
            retained_partner_sequence="AAAAAAAATTT",
            retained_partner_length_nt=11,
            active_product_sequence="TTTTTTTTTTT",
            active_product_span=(0, 11),
            active_product_length_nt=11,
            rebased_nick_boundary=2,
            nickase_site_survives_post_release=True,
            release_site_survives_post_release=False,
        ),
        final_candidate=ReleasedFinalCandidate.model_construct(
            final_geometry_source="exposed_bottom_strand",
            designed_sequence="AAAAAAAATTT",
            input_sequence="AAAAAAAA",
            foldback_arm="TTT",
            nick_boundary_from_left=2,
            paired_bp=3,
            cap_nt=3,
            source_cap_nt=3,
            cap_extension_nt=0,
            active_product_length_nt=11,
            active_product_input_length_nt=8,
            mismatch_count=0,
            mismatch_positions=[],
            terminal_ligatable_duplex_bp=3,
            max_uninterrupted_duplex_bp=3,
            extra_nick_event_count=0,
            extra_target_strand_nick_count=0,
            gc_fraction_added=0.0,
            max_homopolymer_run_added=8,
            projected_origin_event=None,
            extra_target_strand_nicks=[],
            extra_nick_events=[],
            post_nick_sequence="AAAAAAAATTT",
            nickase_site_survives_post_release=True,
            release_site_survives_post_release=False,
        ),
    )
    near_hit = exact_hit.model_copy(update={"rank": 1, "hit_kind": "nearest", "nick_boundary_from_left": 3})

    nick_placement = NickPlacement(
        entry=NickaseCatalogEntry(
            id="Nx.Test",
            specificity_id="Nx.Test",
            motif_top_5to3="AAAA",
            top_cut_offset=0,
        ),
        orientation="forward",
        motif="AAAA",
        site_start_at_boundary_zero=0,
    )
    release_placement = ReleasePlacement(
        entry=ReleaseEnzymeEntry(
            variant_id="Re.Test",
            display_name="Re.Test",
            recognition_sequence="TTTT",
            top_cut_offset=0,
            bottom_cut_offset=0,
            class_label="other_ds_re",
            commercial_confidence="primary_vendor_current",
            source_catalog_id="test_release",
        ),
        orientation="forward",
        motif="TTTT",
        site_shift_from_boundary=0,
        top_cut_shift_from_boundary=9,
        bottom_cut_shift_from_boundary=9,
    )

    monkeypatch.setattr(released_target_search, "_nick_placements", lambda *_args, **_kwargs: [nick_placement])
    monkeypatch.setattr(released_target_search, "_release_placements", lambda *_args, **_kwargs: [release_placement])
    monkeypatch.setattr(released_target_search, "_search_pair", lambda **_kwargs: (exact_hit, [near_hit]))
    monkeypatch.setattr(released_target_search, "_rank_hits", lambda hits, **_kwargs: hits)

    report = released_target_search.search_released_target_hits(
        request=request,
        nick_catalog=object(),  # type: ignore[arg-type]
        release_catalog=object(),  # type: ignore[arg-type]
        workspace_root=Path("/tmp/workspace"),
        nick_catalog_source="test_nick",
        release_catalog_source="test_release",
    )

    assert report.status == "exact_hits_found"
    assert report.metadata.pre_truncation_exact_hit_count == 1
    assert report.metadata.pre_truncation_near_hit_count == 1
    assert len(report.exact_hits) == 1
    assert len(report.near_hits) == 1


def test_released_target_search_against_real_presets_evaluates_the_full_cross_product(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces" / "released_snapback"
    workspace_root.mkdir(parents=True, exist_ok=True)
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="neb_nicking_v1", additional_presets=["thermo_nicking_v1"]),
        release_sources=ReleaseCatalogSources(preset="type_iis_release_v1"),
    )

    report = run_released_snapback_target_search(
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
    nick_placements = released_target_search._nick_placements(
        nick_catalog,
        physical_nicked_strand="top",
    )
    release_placements = released_target_search._release_placements(
        release_catalog,
        target=request.target,
    )
    allowed_nick_placements = [
        placement
        for placement in nick_placements
        if not matching_nickase_warning_codes(
            placement.entry,
            warning_codes=request.search.disallowed_nickase_warning_codes,
        )
    ]
    expected_pairs = len(allowed_nick_placements) * len(release_placements)
    disallowed_nick_placement_count = len(nick_placements) - len(allowed_nick_placements)

    assert report.status == "exact_hits_found"
    assert report.metadata.evaluated_pair_count == expected_pairs
    assert report.metadata.pre_truncation_exact_hit_count == 2
    assert len(report.exact_hits) == 2
    assert {hit.nickase_variant_id for hit in report.exact_hits} == {"Nb.BsrDI", "Nb.BtsI"}
    assert report.metadata.blocker_counts["DISALLOWED_NICKASE_WARNING_CODE"] == disallowed_nick_placement_count * len(
        release_placements
    )
    assert report.metadata.pre_truncation_near_hit_count >= 1
    assert len(report.near_hits) >= 1
    assert all(
        hit.pre_nick_site.local_start is not None and hit.pre_nick_site.local_start >= 0 for hit in report.near_hits
    )
    assert all(hit.nickase_variant_id != "Nt.CviPII" for hit in report.near_hits)
    assert all(hit.nickase.metadata.get("demo_only") is not True for hit in report.near_hits)
    assert all(hit.release_enzyme.source_catalog_id == "type_iis_release_v1" for hit in report.near_hits)


def test_search_released_target_hits_suppresses_demo_only_pairs_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="test_nick"),
        release_sources=ReleaseCatalogSources(preset="test_release"),
    )
    demo_nick = NickPlacement(
        entry=NickaseCatalogEntry(
            id="Nx.Demo",
            specificity_id="Nx.Demo",
            motif_top_5to3="AAAA",
            top_cut_offset=0,
            metadata={"demo_only": True},
        ),
        orientation="forward",
        motif="AAAA",
        site_start_at_boundary_zero=0,
    )
    demo_release = ReleasePlacement(
        entry=ReleaseEnzymeEntry(
            variant_id="Re.Demo",
            display_name="Re.Demo",
            recognition_sequence="TTTT",
            top_cut_offset=0,
            bottom_cut_offset=0,
            class_label="other_ds_re",
            commercial_confidence="primary_vendor_current",
            source_catalog_id="local_release",
            metadata={"demo_only": True},
        ),
        orientation="forward",
        motif="TTTT",
        site_shift_from_boundary=0,
        top_cut_shift_from_boundary=9,
        bottom_cut_shift_from_boundary=9,
    )

    monkeypatch.setattr(released_target_search, "_nick_placements", lambda *_args, **_kwargs: [demo_nick])
    monkeypatch.setattr(released_target_search, "_release_placements", lambda *_args, **_kwargs: [demo_release])
    monkeypatch.setattr(
        released_target_search,
        "_search_pair",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("demo-only pairs must be skipped before evaluation")),
    )

    report = released_target_search.search_released_target_hits(
        request=request,
        nick_catalog=object(),  # type: ignore[arg-type]
        release_catalog=object(),  # type: ignore[arg-type]
        workspace_root=Path("/tmp/workspace"),
        nick_catalog_source="test_nick",
        release_catalog_source="test_release",
    )

    assert report.status == "no_hits"
    assert report.metadata.evaluated_pair_count == 0
    assert report.metadata.blocker_counts == {"DEMO_ONLY_PAIR_SUPPRESSED": 1}


def test_search_released_target_hits_allows_demo_only_pairs_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="test_nick"),
        release_sources=ReleaseCatalogSources(preset="test_release"),
        search=ReleasedTargetSearchConfig(allow_demo_hits=True),
    )
    demo_nick = NickPlacement(
        entry=NickaseCatalogEntry(
            id="Nx.Demo",
            specificity_id="Nx.Demo",
            motif_top_5to3="AAAA",
            top_cut_offset=0,
            metadata={"demo_only": True},
        ),
        orientation="forward",
        motif="AAAA",
        site_start_at_boundary_zero=0,
    )
    demo_release = ReleasePlacement(
        entry=ReleaseEnzymeEntry(
            variant_id="Re.Demo",
            display_name="Re.Demo",
            recognition_sequence="TTTT",
            top_cut_offset=0,
            bottom_cut_offset=0,
            class_label="other_ds_re",
            commercial_confidence="primary_vendor_current",
            source_catalog_id="local_release",
            metadata={"demo_only": True},
        ),
        orientation="forward",
        motif="TTTT",
        site_shift_from_boundary=0,
        top_cut_shift_from_boundary=9,
        bottom_cut_shift_from_boundary=9,
    )
    exact_hit = ReleasedTargetSearchHit.model_construct(
        rank=1,
        hit_kind="exact",
        nickase_variant_id="Nx.Demo",
        release_variant_id="Re.Demo",
        intended_nick_site_orientation="forward",
        intended_nick_site_sequence="AAAA",
        release_site_orientation="forward",
        release_site_sequence="TTTT",
        nick_boundary_from_left=0,
        active_product_input_length_nt=6,
        active_product_length_nt=9,
        precursor_length_nt=12,
        sacrificial_downstream_tail_nt=3,
        extra_nick_event_count=0,
        extra_target_strand_nick_count=0,
        precursor_top_strand="AAAATTTTAAAA",
        nickase=CatalogNormalizationInfo.model_construct(
            variant_id="Nx.Demo",
            specificity_id="Nx.Demo",
            motif_top_5to3="AAAA",
            motif_len=4,
            nicked_strand="top",
            active_cut_offset=0,
        ),
        release_enzyme=ReleaseCatalogNormalizationInfo.model_construct(
            variant_id="Re.Demo",
            display_name="Re.Demo",
            recognition_sequence="TTTT",
            recognition_len=4,
            top_cut_offset=0,
            bottom_cut_offset=0,
            class_label="other_ds_re",
            outside_site=False,
            commercial_confidence="primary_vendor_current",
            source_catalog_id="test_release",
        ),
        projection=ReleasedProductProjection.model_construct(
            final_geometry_source="exposed_bottom_strand",
            precursor_top_strand="AAAATTTTAAAA",
            precursor_length=12,
            nick_coordinate_precursor=0,
            release_top_cut_precursor=9,
            release_bottom_cut_precursor=9,
            retained_partner_sequence="AAAATTTTA",
            retained_partner_length_nt=9,
            active_product_sequence="TTTTAAAAT",
            active_product_span=(0, 9),
            active_product_length_nt=9,
            rebased_nick_boundary=0,
            nickase_site_survives_post_release=True,
            release_site_survives_post_release=False,
        ),
        final_candidate=ReleasedFinalCandidate.model_construct(
            final_geometry_source="exposed_bottom_strand",
            designed_sequence="AAAATTTTA",
            input_sequence="AAAATT",
            foldback_arm="TTA",
            nick_boundary_from_left=0,
            paired_bp=3,
            cap_nt=3,
            source_cap_nt=3,
            cap_extension_nt=0,
            active_product_length_nt=9,
            active_product_input_length_nt=6,
            mismatch_count=0,
            mismatch_positions=[],
            terminal_ligatable_duplex_bp=3,
            max_uninterrupted_duplex_bp=3,
            extra_nick_event_count=0,
            extra_target_strand_nick_count=0,
            gc_fraction_added=0.0,
            max_homopolymer_run_added=4,
            projected_origin_event=None,
            extra_target_strand_nicks=[],
            extra_nick_events=[],
            post_nick_sequence="AAAATTTTA",
            nickase_site_survives_post_release=True,
            release_site_survives_post_release=False,
        ),
    )

    monkeypatch.setattr(released_target_search, "_nick_placements", lambda *_args, **_kwargs: [demo_nick])
    monkeypatch.setattr(released_target_search, "_release_placements", lambda *_args, **_kwargs: [demo_release])
    monkeypatch.setattr(released_target_search, "_search_pair", lambda **_kwargs: (exact_hit, []))
    monkeypatch.setattr(released_target_search, "_rank_hits", lambda hits, **_kwargs: hits)

    report = released_target_search.search_released_target_hits(
        request=request,
        nick_catalog=object(),  # type: ignore[arg-type]
        release_catalog=object(),  # type: ignore[arg-type]
        workspace_root=Path("/tmp/workspace"),
        nick_catalog_source="test_nick",
        release_catalog_source="test_release",
    )

    assert report.status == "exact_hits_found"
    assert report.metadata.evaluated_pair_count == 1
    assert report.metadata.blocker_counts == {}
    assert report.exact_hits[0].nickase_variant_id == "Nx.Demo"


def test_search_released_target_hits_suppresses_disallowed_warning_code_pairs_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="test_nick"),
        release_sources=ReleaseCatalogSources(preset="test_release"),
    )
    blocked_nick = NickPlacement(
        entry=NickaseCatalogEntry(
            id="Nx.Blocked",
            specificity_id="Nx.Blocked",
            motif_top_5to3="AAAA",
            top_cut_offset=0,
            selection={"warning_codes": ["FREQUENT_CUTTER"]},
        ),
        orientation="forward",
        motif="AAAA",
        site_start_at_boundary_zero=0,
    )
    release = ReleasePlacement(
        entry=ReleaseEnzymeEntry(
            variant_id="Re.Test",
            display_name="Re.Test",
            recognition_sequence="TTTT",
            top_cut_offset=0,
            bottom_cut_offset=0,
            class_label="other_ds_re",
            commercial_confidence="primary_vendor_current",
            source_catalog_id="local_release",
        ),
        orientation="forward",
        motif="TTTT",
        site_shift_from_boundary=0,
        top_cut_shift_from_boundary=9,
        bottom_cut_shift_from_boundary=9,
    )

    monkeypatch.setattr(released_target_search, "_nick_placements", lambda *_args, **_kwargs: [blocked_nick])
    monkeypatch.setattr(released_target_search, "_release_placements", lambda *_args, **_kwargs: [release])
    monkeypatch.setattr(
        released_target_search,
        "_search_pair",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("disallowed warning-code pairs must be skipped before evaluation")
        ),
    )

    report = released_target_search.search_released_target_hits(
        request=request,
        nick_catalog=object(),  # type: ignore[arg-type]
        release_catalog=object(),  # type: ignore[arg-type]
        workspace_root=Path("/tmp/workspace"),
        nick_catalog_source="test_nick",
        release_catalog_source="test_release",
    )

    assert report.status == "no_hits"
    assert report.metadata.evaluated_pair_count == 0
    assert report.metadata.blocker_counts == {"DISALLOWED_NICKASE_WARNING_CODE": 1}


def test_search_released_target_hits_allows_warning_coded_nickase_when_policy_is_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="test_nick"),
        release_sources=ReleaseCatalogSources(preset="test_release"),
        search=ReleasedTargetSearchConfig(disallowed_nickase_warning_codes=[]),
    )
    warning_nick = NickPlacement(
        entry=NickaseCatalogEntry(
            id="Nx.Warning",
            specificity_id="Nx.Warning",
            motif_top_5to3="AAAA",
            top_cut_offset=0,
            selection={"warning_codes": ["FREQUENT_CUTTER"]},
        ),
        orientation="forward",
        motif="AAAA",
        site_start_at_boundary_zero=0,
    )
    release = ReleasePlacement(
        entry=ReleaseEnzymeEntry(
            variant_id="Re.Test",
            display_name="Re.Test",
            recognition_sequence="TTTT",
            top_cut_offset=0,
            bottom_cut_offset=0,
            class_label="other_ds_re",
            commercial_confidence="primary_vendor_current",
            source_catalog_id="local_release",
        ),
        orientation="forward",
        motif="TTTT",
        site_shift_from_boundary=0,
        top_cut_shift_from_boundary=9,
        bottom_cut_shift_from_boundary=9,
    )
    exact_hit = ReleasedTargetSearchHit.model_construct(
        rank=1,
        hit_kind="exact",
        nickase_variant_id="Nx.Warning",
        release_variant_id="Re.Test",
        intended_nick_site_orientation="forward",
        intended_nick_site_sequence="AAAA",
        release_site_orientation="forward",
        release_site_sequence="TTTT",
        nick_boundary_from_left=0,
        active_product_input_length_nt=6,
        active_product_length_nt=9,
        precursor_length_nt=12,
        sacrificial_downstream_tail_nt=3,
        extra_nick_event_count=0,
        extra_target_strand_nick_count=0,
        precursor_top_strand="AAAATTTTAAAA",
        nickase=CatalogNormalizationInfo.model_construct(
            variant_id="Nx.Warning",
            specificity_id="Nx.Warning",
            motif_top_5to3="AAAA",
            motif_len=4,
            nicked_strand="top",
            active_cut_offset=0,
            selection={"warning_codes": ["FREQUENT_CUTTER"]},
        ),
        release_enzyme=ReleaseCatalogNormalizationInfo.model_construct(
            variant_id="Re.Test",
            display_name="Re.Test",
            recognition_sequence="TTTT",
            recognition_len=4,
            top_cut_offset=0,
            bottom_cut_offset=0,
            class_label="other_ds_re",
            outside_site=False,
            commercial_confidence="primary_vendor_current",
            source_catalog_id="test_release",
        ),
        projection=ReleasedProductProjection.model_construct(
            final_geometry_source="exposed_bottom_strand",
            precursor_top_strand="AAAATTTTAAAA",
            precursor_length=12,
            nick_coordinate_precursor=0,
            release_top_cut_precursor=9,
            release_bottom_cut_precursor=9,
            retained_partner_sequence="AAAATTTTA",
            retained_partner_length_nt=9,
            active_product_sequence="TTTTAAAAT",
            active_product_span=(0, 9),
            active_product_length_nt=9,
            rebased_nick_boundary=0,
            nickase_site_survives_post_release=True,
            release_site_survives_post_release=False,
        ),
        final_candidate=ReleasedFinalCandidate.model_construct(
            final_geometry_source="exposed_bottom_strand",
            designed_sequence="TTTTAAAAT",
            input_sequence="TTTTAA",
            foldback_arm="AAT",
            nick_boundary_from_left=0,
            paired_bp=3,
            cap_nt=3,
            source_cap_nt=3,
            cap_extension_nt=0,
            active_product_length_nt=9,
            active_product_input_length_nt=6,
            mismatch_count=0,
            mismatch_positions=[],
            terminal_ligatable_duplex_bp=3,
            max_uninterrupted_duplex_bp=3,
            extra_nick_event_count=0,
            extra_target_strand_nick_count=0,
            gc_fraction_added=0.0,
            max_homopolymer_run_added=4,
            projected_origin_event=None,
            extra_target_strand_nicks=[],
            extra_nick_events=[],
            post_nick_sequence="TTTTAAAAT",
            nickase_site_survives_post_release=True,
            release_site_survives_post_release=False,
        ),
    )

    monkeypatch.setattr(released_target_search, "_nick_placements", lambda *_args, **_kwargs: [warning_nick])
    monkeypatch.setattr(released_target_search, "_release_placements", lambda *_args, **_kwargs: [release])
    monkeypatch.setattr(released_target_search, "_search_pair", lambda **_kwargs: (exact_hit, []))
    monkeypatch.setattr(released_target_search, "_rank_hits", lambda hits, **_kwargs: hits)

    report = released_target_search.search_released_target_hits(
        request=request,
        nick_catalog=object(),  # type: ignore[arg-type]
        release_catalog=object(),  # type: ignore[arg-type]
        workspace_root=Path("/tmp/workspace"),
        nick_catalog_source="test_nick",
        release_catalog_source="test_release",
    )

    assert report.status == "exact_hits_found"
    assert report.metadata.evaluated_pair_count == 1
    assert report.metadata.blocker_counts == {}
    assert report.exact_hits[0].nickase_variant_id == "Nx.Warning"
