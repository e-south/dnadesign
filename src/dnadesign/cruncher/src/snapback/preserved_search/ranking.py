"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/preserved_search/ranking.py

Ranking helpers for preserved-site target search.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.snapback.target_models import (
    SnapbackTargetGeometry,
    SnapbackTargetSearchHit,
)

_SNAPBACK_TIER_RANK = {
    "tier1": 0,
    "tier2": 1,
    "tier3": 2,
    None: 3,
}
_COMMERCIAL_CONFIDENCE_RANK = {
    "primary_vendor_current": 0,
    "secondary_vendor_current": 1,
    "produced_on_demand": 2,
    "literature_only": 3,
    None: 4,
}


def catalog_info_priority_key(hit: SnapbackTargetSearchHit) -> tuple[object, ...]:
    selection = hit.nickase.selection
    warning_codes = selection.warning_codes if selection is not None else []
    return (
        _SNAPBACK_TIER_RANK[selection.snapback_tier if selection is not None else None],
        0 if selection is not None and selection.outside_site is True else 1 if selection is not None else 2,
        -(hit.nickase.motif_len or len(hit.nickase.motif_top_5to3)),
        _COMMERCIAL_CONFIDENCE_RANK[selection.commercial_confidence if selection is not None else None],
        len(warning_codes),
        hit.variant_id,
    )


def exact_hit_rank_key(hit: SnapbackTargetSearchHit) -> tuple[object, ...]:
    outside_site = hit.nickase.selection.outside_site if hit.nickase.selection is not None else None
    outside_rank = 0 if outside_site is True else 1 if outside_site is False else 2
    return (
        hit.extra_target_strand_nick_count,
        hit.extra_nick_event_count,
        hit.input_length_nt,
        outside_rank,
        catalog_info_priority_key(hit),
        hit.intended_site_sequence,
        hit.input_sequence,
        hit.variant_id,
    )


def near_hit_rank_key(
    hit: SnapbackTargetSearchHit,
    *,
    target: SnapbackTargetGeometry,
) -> tuple[object, ...]:
    exact_key = exact_hit_rank_key(hit)
    return (
        abs(hit.nick_boundary_from_left - target.nick_boundary_from_left),
        hit.nick_boundary_from_left,
        *exact_key,
    )


def rank_hits(
    hits: list[SnapbackTargetSearchHit],
    *,
    target: SnapbackTargetGeometry,
    exact: bool,
) -> list[SnapbackTargetSearchHit]:
    ordered = sorted(
        hits,
        key=(exact_hit_rank_key if exact else lambda hit: near_hit_rank_key(hit, target=target)),
    )
    return [hit.model_copy(update={"rank": index}) for index, hit in enumerate(ordered, start=1)]
