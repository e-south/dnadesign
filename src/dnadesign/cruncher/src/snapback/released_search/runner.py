"""
Top-level orchestration for released-product target-search.
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.nickases.models import NickaseCatalog
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeCatalog
from dnadesign.cruncher.snapback.released_route_policy import (
    route_family_active_strand,
    route_family_physical_nicked_strand,
)
from dnadesign.cruncher.snapback.released_search.evaluator_adapter import search_pair
from dnadesign.cruncher.snapback.released_search.nick_placements import (
    nick_placements,
    nickase_entry_has_disallowed_warning_code,
    nickase_entry_is_demo_only,
)
from dnadesign.cruncher.snapback.released_search.ranking import rank_hits
from dnadesign.cruncher.snapback.released_search.release_placements import (
    release_entry_is_demo_only,
    release_placements,
)
from dnadesign.cruncher.snapback.released_search.reporting import blocker, build_search_report
from dnadesign.cruncher.snapback.released_search_models import (
    ReleasedTargetSearchHit,
    ReleasedTargetSearchReport,
    SingleNickReleasedTargetSearchRequest,
)


def search_released_target_hits(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    nick_catalog: NickaseCatalog,
    release_catalog: ReleaseEnzymeCatalog,
    workspace_root: Path,
    nick_catalog_source: str,
    release_catalog_source: str,
    nick_placements_fn=nick_placements,
    release_placements_fn=release_placements,
    search_pair_fn=search_pair,
    rank_hits_fn=rank_hits,
    nickase_entry_is_demo_only_fn=nickase_entry_is_demo_only,
    release_entry_is_demo_only_fn=release_entry_is_demo_only,
    nickase_entry_has_disallowed_warning_code_fn=nickase_entry_has_disallowed_warning_code,
    blocker_fn=blocker,
) -> ReleasedTargetSearchReport:
    blocker_counts: dict[str, int] = {}
    exact_hits: list[ReleasedTargetSearchHit] = []
    near_hits: list[ReleasedTargetSearchHit] = []
    evaluated_pair_count = 0
    for route_family in request.search.allowed_route_families:
        active_strand = route_family_active_strand(route_family)
        if active_strand not in request.search.allowed_active_strands:
            blocker_fn(blocker_counts, "ACTIVE_STRAND_UNSUPPORTED")
            continue
        route_nick_placements = nick_placements_fn(
            nick_catalog,
            physical_nicked_strand=route_family_physical_nicked_strand(route_family),
            use_vendor_diagram=request.search.allow_precut_footprint_outside_active_product,
        )
        if not route_nick_placements:
            blocker_fn(blocker_counts, "NO_NICKASE_PLACEMENT")
            continue
        route_release_placements = release_placements_fn(
            release_catalog,
            target=request.target,
            active_strand=active_strand,
        )
        allowed_release_variant_ids = set(request.search.allowed_release_variant_ids)
        if allowed_release_variant_ids:
            filtered_release_placements = [
                placement
                for placement in route_release_placements
                if placement.entry.variant_id in allowed_release_variant_ids
            ]
            for _index in range(len(route_release_placements) - len(filtered_release_placements)):
                blocker_fn(blocker_counts, "RELEASE_VARIANT_FILTERED")
            route_release_placements = filtered_release_placements
        for nick_placement in route_nick_placements:
            for release_placement in route_release_placements:
                if not request.search.allow_demo_hits and (
                    nickase_entry_is_demo_only_fn(nick_placement.entry)
                    or release_entry_is_demo_only_fn(release_placement.entry)
                ):
                    blocker_fn(blocker_counts, "DEMO_ONLY_PAIR_SUPPRESSED")
                    continue
                if nickase_entry_has_disallowed_warning_code_fn(
                    nick_placement.entry,
                    warning_codes=request.search.disallowed_nickase_warning_codes,
                ):
                    blocker_fn(blocker_counts, "DISALLOWED_NICKASE_WARNING_CODE")
                    continue
                evaluated_pair_count += 1
                exact_hit, pair_near_hits = search_pair_fn(
                    request=request,
                    route_family=route_family,
                    nick_placement=nick_placement,
                    release_placement=release_placement,
                    blocker_counts=blocker_counts,
                )
                if exact_hit is not None:
                    exact_hits.append(exact_hit)
                if pair_near_hits:
                    near_hits.extend(pair_near_hits)
    ranked_exact_hits = rank_hits_fn(exact_hits, target=request.target, exact=True)
    ranked_near_hits = rank_hits_fn(near_hits, target=request.target, exact=False)
    return build_search_report(
        request=request,
        workspace_root=workspace_root,
        nick_catalog_source=nick_catalog_source,
        release_catalog_source=release_catalog_source,
        blocker_counts=blocker_counts,
        evaluated_pair_count=evaluated_pair_count,
        ranked_exact_hits=ranked_exact_hits,
        ranked_near_hits=ranked_near_hits,
    )


__all__ = ["search_released_target_hits"]
