"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/snapback/test_target_search_helpers.py

Focused helper tests for preserved-site target search seams.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.nickases.catalog import load_merged_nickase_catalog
from dnadesign.cruncher.snapback.models import CatalogNormalizationInfo
from dnadesign.cruncher.snapback.preserved_search.placements import (
    build_feasibility_row,
    iter_target_strand_placements,
)
from dnadesign.cruncher.snapback.preserved_search.ranking import rank_hits
from dnadesign.cruncher.snapback.target_models import (
    SnapbackTargetGeometry,
    SnapbackTargetSearchHit,
)


def test_iter_target_strand_placements_surfaces_exact_and_near_feasibility(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces" / "demo_snapback"
    workspace_root.mkdir(parents=True, exist_ok=True)
    catalog, _ = load_merged_nickase_catalog(
        preset_id="neb_nicking_v1",
        additional_preset_ids=["thermo_nicking_v1"],
        additional_paths=[],
        workspace_root=workspace_root,
    )

    placements = iter_target_strand_placements(
        catalog_entries=catalog.entries,
        target=SnapbackTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        normalize_to_top_strand_nick=True,
    )
    feasibility = {(row.variant_id, row.orientation): row for row in map(build_feasibility_row, placements)}

    assert feasibility[("Nt.CviPII", "forward")].exact_boundary_hit_possible is True
    assert feasibility[("Nt.BspQI", "forward")].earliest_feasible_boundary == 8
    assert "NEGATIVE_SITE_START_AT_TARGET_BOUNDARY" in feasibility[("Nt.BspQI", "forward")].exact_boundary_blockers


def test_rank_hits_prefers_fewer_extra_target_nicks_for_exact_hits() -> None:
    def _hit(*, variant_id: str, extra_target_strand_nick_count: int) -> SnapbackTargetSearchHit:
        return SnapbackTargetSearchHit.model_construct(
            rank=1,
            hit_kind="exact",
            variant_id=variant_id,
            intended_site_orientation="forward",
            intended_site_sequence="CCA",
            nick_boundary_from_left=0,
            site_start=0,
            site_end=3,
            input_sequence="CCAGGG",
            designed_sequence="CCAGGGTTT",
            input_length_nt=6,
            designed_length_nt=9,
            paired_bp=3,
            cap_nt=3,
            source_cap_nt=3,
            cap_extension_nt=0,
            site_mutation_count=0,
            extra_nick_event_count=0,
            extra_target_strand_nick_count=extra_target_strand_nick_count,
            nickase=CatalogNormalizationInfo.model_construct(
                variant_id=variant_id,
                specificity_id=variant_id,
                motif_top_5to3="CCA",
                motif_len=3,
                nicked_strand="top",
                active_cut_offset=0,
            ),
            explicit_report=None,
        )

    ranked = rank_hits(
        [
            _hit(variant_id="Nt.MoreTargetNicks", extra_target_strand_nick_count=1),
            _hit(variant_id="Nt.FewerTargetNicks", extra_target_strand_nick_count=0),
        ],
        target=SnapbackTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        exact=True,
    )

    assert [hit.variant_id for hit in ranked] == ["Nt.FewerTargetNicks", "Nt.MoreTargetNicks"]
