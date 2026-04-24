"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_projection_common.py

Shared helpers for released-product precursor projection.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.nickases.models import NickEvent, iupac_bases_for_symbol
from dnadesign.cruncher.nickases.scanning import (
    EvaluatedMatch,
    display_footprint_for_orientation,
    display_motif_for_orientation,
)
from dnadesign.cruncher.snapback.models import SnapbackIssue
from dnadesign.cruncher.snapback.released_route_policy import ReleasedActiveStrand


def build_issue(code: str, message: str, **details: object) -> SnapbackIssue:
    return SnapbackIssue(code=code, message=message, details=details)


def physical_nicked_strand_from_event(event: NickEvent) -> ReleasedActiveStrand:
    return "top" if event.strand == "primary" else "bottom"


def site_symbols_for_match(match: EvaluatedMatch) -> str:
    site_span_len = match.site.end - match.site.start
    if site_span_len == match.variant.resolved_vendor_diagram_len:
        return display_footprint_for_orientation(match.variant, orientation=match.site.orientation)
    return display_motif_for_orientation(match.variant, orientation=match.site.orientation)


def provenance_source_constraint(
    *,
    nick_match: EvaluatedMatch,
    precursor_index: int,
) -> str:
    if not (nick_match.site.start <= precursor_index < nick_match.site.end):
        return "user_sequence"
    site_symbol = site_symbols_for_match(nick_match)[precursor_index - nick_match.site.start]
    if len(iupac_bases_for_symbol(site_symbol)) > 1:
        return "degenerate_motif_base"
    return "fixed_motif_base"


def build_projected_origin_event(original: NickEvent, *, rebased_boundary: int) -> NickEvent:
    return NickEvent(
        variant_id=original.variant_id,
        specificity_id=original.specificity_id,
        strand=original.strand,
        boundary=rebased_boundary,
        boundary_context=rebased_boundary,
        source_site_start=max(0, rebased_boundary),
        source_site_end=max(0, rebased_boundary + 1),
        source_site_orientation=original.source_site_orientation,
    )
