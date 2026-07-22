"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/released_search/release_placements.py

Release-enzyme placement generation for released-product target-search.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeCatalog, ReleaseEnzymeEntry
from dnadesign.cruncher.release_enzymes.scanning import derive_release_cut
from dnadesign.cruncher.release_enzymes.scanning import display_motif_for_orientation as display_release_motif
from dnadesign.cruncher.release_enzymes.selection import release_entry_priority_key
from dnadesign.cruncher.snapback.released_route_policy import ReleasedActiveStrand
from dnadesign.cruncher.snapback.released_search.placement_models import ReleasePlacement
from dnadesign.cruncher.snapback.released_spec_models import ReleasedFinalTargetGeometry


def release_entry_is_demo_only(entry: ReleaseEnzymeEntry) -> bool:
    demo_only = entry.metadata.get("demo_only")
    return isinstance(demo_only, bool) and demo_only


def release_placements(
    catalog: ReleaseEnzymeCatalog,
    *,
    target: ReleasedFinalTargetGeometry,
    active_strand: ReleasedActiveStrand = "bottom",
) -> list[ReleasePlacement]:
    placements: list[ReleasePlacement] = []
    active_product_length_offset = (2 * target.paired_bp) + target.cap_nt
    for entry in catalog.entries:
        for orientation in ("forward", "reverse"):
            motif = display_release_motif(entry, orientation=orientation)
            cut = derive_release_cut(entry=entry, start=0, orientation=orientation)
            active_cut_boundary = cut.top_cut_boundary if active_strand == "top" else cut.bottom_cut_boundary
            placements.append(
                ReleasePlacement(
                    entry=entry,
                    orientation=orientation,
                    motif=motif,
                    active_strand=active_strand,
                    site_shift_from_boundary=active_product_length_offset - active_cut_boundary,
                    top_cut_shift_from_boundary=active_product_length_offset
                    + (cut.top_cut_boundary - active_cut_boundary),
                    bottom_cut_shift_from_boundary=active_product_length_offset
                    + (cut.bottom_cut_boundary - active_cut_boundary),
                )
            )
    return sorted(
        placements,
        key=lambda placement: (
            release_entry_priority_key(placement.entry),
            placement.orientation,
            placement.motif,
            placement.active_strand,
            placement.entry.variant_id,
        ),
    )


__all__ = ["release_entry_is_demo_only", "release_placements"]
