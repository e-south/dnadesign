"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/released_search/nick_placements.py

Nickase placement generation for released-product target-search.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.nickases.models import NickaseCatalog, NickaseCatalogEntry
from dnadesign.cruncher.nickases.scanning import (
    display_footprint_for_orientation,
    display_motif_for_orientation,
    enumerate_boundary_placements,
    leading_fully_degenerate_prefix_nt,
)
from dnadesign.cruncher.nickases.selection import matching_nickase_warning_codes, snapback_entry_priority_key
from dnadesign.cruncher.snapback.released_route_policy import ReleasedActiveStrand
from dnadesign.cruncher.snapback.released_search.placement_models import NickPlacement


def nickase_entry_is_demo_only(entry: NickaseCatalogEntry) -> bool:
    demo_only = entry.metadata.get("demo_only")
    if isinstance(demo_only, bool):
        return demo_only
    return entry.source == "local_demo"


def nickase_entry_has_disallowed_warning_code(entry: NickaseCatalogEntry, *, warning_codes: list[str]) -> bool:
    return bool(matching_nickase_warning_codes(entry, warning_codes=warning_codes))


def nick_placements(
    catalog: NickaseCatalog,
    *,
    physical_nicked_strand: ReleasedActiveStrand | None = None,
    use_vendor_diagram: bool = False,
) -> list[NickPlacement]:
    placements: list[NickPlacement] = []
    required_strand = None
    if physical_nicked_strand == "top":
        required_strand = "primary"
    elif physical_nicked_strand == "bottom":
        required_strand = "complement"
    for entry in catalog.entries:
        for orientation, site_start in enumerate_boundary_placements(
            entry,
            boundary=0,
            required_strand=required_strand,
            use_vendor_diagram=use_vendor_diagram,
        ):
            placements.append(
                NickPlacement(
                    entry=entry,
                    orientation=orientation,
                    motif=(
                        display_footprint_for_orientation(entry, orientation=orientation)
                        if use_vendor_diagram
                        else display_motif_for_orientation(entry, orientation=orientation)
                    ),
                    site_start_at_boundary_zero=site_start,
                    left_of_origin_slack_nt=leading_fully_degenerate_prefix_nt(
                        entry,
                        orientation=orientation,
                        use_vendor_diagram=use_vendor_diagram,
                    ),
                )
            )
    return sorted(
        placements,
        key=lambda placement: (
            snapback_entry_priority_key(placement.entry),
            placement.orientation,
            placement.motif,
            placement.entry.id,
        ),
    )


__all__ = [
    "nick_placements",
    "nickase_entry_has_disallowed_warning_code",
    "nickase_entry_is_demo_only",
]
