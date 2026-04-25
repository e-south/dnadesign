"""
Placement enumeration helpers for preserved-site target search.
"""

from __future__ import annotations

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry
from dnadesign.cruncher.nickases.scanning import (
    display_motif_for_orientation,
    enumerate_boundary_placements,
)
from dnadesign.cruncher.nickases.selection import snapback_entry_priority_key
from dnadesign.cruncher.snapback.preserved_search.placement_models import Placement
from dnadesign.cruncher.snapback.target_models import (
    SnapbackTargetFeasibilityRow,
    SnapbackTargetGeometry,
)


def placement_rank_key(placement: Placement) -> tuple[object, ...]:
    outside_site = placement.entry.selection.outside_site if placement.entry.selection is not None else None
    outside_rank = 0 if outside_site is True else 1 if outside_site is False else 2
    earliest_boundary = (
        placement.earliest_feasible_boundary if placement.earliest_feasible_boundary is not None else 10**9
    )
    return (
        0 if placement.exact_boundary_hit_possible else 1,
        earliest_boundary,
        outside_rank,
        snapback_entry_priority_key(placement.entry),
        placement.orientation,
        placement.motif,
        placement.entry.id,
    )


def build_placement(
    *,
    entry: NickaseCatalogEntry,
    orientation: str,
    site_start_at_target_boundary: int,
    target: SnapbackTargetGeometry,
) -> Placement:
    motif = display_motif_for_orientation(entry, orientation=orientation)
    boundary_offset = target.nick_boundary_from_left - site_start_at_target_boundary
    site_end_at_target_boundary = site_start_at_target_boundary + len(motif)
    max_input_length_at_target = target.nick_boundary_from_left + target.paired_bp + target.cap_nt
    exact_input_length_nt: int | None = None
    exact_boundary_blockers: list[str] = []
    if site_start_at_target_boundary < 0:
        exact_boundary_blockers.append("NEGATIVE_SITE_START_AT_TARGET_BOUNDARY")
    if site_end_at_target_boundary > max_input_length_at_target:
        exact_boundary_blockers.append("SITE_EXCEEDS_MAX_INPUT_AT_TARGET_BOUNDARY")
    if not exact_boundary_blockers:
        exact_input_length_nt = max(target.nick_boundary_from_left + target.paired_bp, site_end_at_target_boundary)

    boundary_invariant_site_extent = len(motif) - boundary_offset
    earliest_feasible_boundary: int | None = None
    earliest_input_length_nt: int | None = None
    if boundary_invariant_site_extent <= target.paired_bp + target.cap_nt:
        earliest_feasible_boundary = max(0, boundary_offset)
        site_start = earliest_feasible_boundary - boundary_offset
        site_end = site_start + len(motif)
        earliest_input_length_nt = max(earliest_feasible_boundary + target.paired_bp, site_end)

    return Placement(
        entry=entry,
        orientation=orientation,
        motif=motif,
        site_start_at_target_boundary=site_start_at_target_boundary,
        boundary_offset=boundary_offset,
        exact_input_length_nt=exact_input_length_nt,
        earliest_feasible_boundary=earliest_feasible_boundary,
        earliest_input_length_nt=earliest_input_length_nt,
        exact_boundary_blockers=tuple(exact_boundary_blockers),
    )


def iter_target_strand_placements(
    *,
    catalog_entries: list[NickaseCatalogEntry],
    target: SnapbackTargetGeometry,
    normalize_to_top_strand_nick: bool,
) -> list[Placement]:
    required_strand = "primary" if normalize_to_top_strand_nick else None
    placements: list[Placement] = []
    for entry in catalog_entries:
        for orientation, site_start_at_target_boundary in enumerate_boundary_placements(
            entry,
            boundary=target.nick_boundary_from_left,
            required_strand=required_strand,
        ):
            placements.append(
                build_placement(
                    entry=entry,
                    orientation=orientation,
                    site_start_at_target_boundary=site_start_at_target_boundary,
                    target=target,
                )
            )
    return sorted(placements, key=placement_rank_key)


def build_feasibility_row(placement: Placement) -> SnapbackTargetFeasibilityRow:
    return SnapbackTargetFeasibilityRow(
        variant_id=placement.entry.id,
        orientation=placement.orientation,
        motif_top_5to3=placement.motif,
        motif_len=len(placement.motif),
        site_start_at_target_boundary=placement.site_start_at_target_boundary,
        site_end_at_target_boundary=placement.site_start_at_target_boundary + len(placement.motif),
        boundary_offset=placement.boundary_offset,
        outside_site=placement.entry.outside_site,
        exact_boundary_hit_possible=placement.exact_boundary_hit_possible,
        exact_boundary_blockers=list(placement.exact_boundary_blockers),
        any_boundary_hit_possible=placement.any_boundary_hit_possible,
        earliest_feasible_boundary=placement.earliest_feasible_boundary,
        exact_input_length_nt=placement.exact_input_length_nt,
        earliest_input_length_nt=placement.earliest_input_length_nt,
    )
