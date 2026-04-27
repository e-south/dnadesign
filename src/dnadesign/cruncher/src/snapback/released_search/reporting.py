"""
Report shaping for released-product target-search.
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.snapback.released_search_models import (
    ReleasedTargetSearchHit,
    ReleasedTargetSearchMetadata,
    ReleasedTargetSearchReport,
    SingleNickReleasedTargetSearchRequest,
)


def blocker(counts: dict[str, int], code: str) -> None:
    counts[code] = counts.get(code, 0) + 1


def report_status(
    *,
    exact_hits: list[ReleasedTargetSearchHit],
    near_hits: list[ReleasedTargetSearchHit],
) -> str:
    if exact_hits:
        return "exact_hits_found"
    if near_hits:
        return "near_hits_only"
    return "no_hits"


def build_search_report(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    workspace_root: Path,
    nick_catalog_source: str,
    release_catalog_source: str,
    blocker_counts: dict[str, int],
    evaluated_pair_count: int,
    ranked_exact_hits: list[ReleasedTargetSearchHit],
    ranked_near_hits: list[ReleasedTargetSearchHit],
) -> ReleasedTargetSearchReport:
    exact_hits = ranked_exact_hits[: request.search.max_results]
    near_hits = ranked_near_hits[: request.search.max_results]
    return ReleasedTargetSearchReport(
        status=report_status(exact_hits=exact_hits, near_hits=near_hits),
        workspace_root=str(workspace_root),
        metadata=ReleasedTargetSearchMetadata(
            route_policy_final_geometry_source=request.search.route_policy_final_geometry_source,
            target=request.target,
            nick_catalog_source=nick_catalog_source,
            release_catalog_source=release_catalog_source,
            disallowed_nickase_warning_codes=list(request.search.disallowed_nickase_warning_codes),
            allowed_release_variant_ids=list(request.search.allowed_release_variant_ids),
            allowed_active_strands=list(request.search.allowed_active_strands),
            allowed_route_families=list(request.search.allowed_route_families),
            evaluated_pair_count=evaluated_pair_count,
            pre_truncation_exact_hit_count=len(ranked_exact_hits),
            post_truncation_exact_hit_count=len(exact_hits),
            pre_truncation_near_hit_count=len(ranked_near_hits),
            post_truncation_near_hit_count=len(near_hits),
            blocker_counts=blocker_counts,
        ),
        exact_hits=exact_hits,
        near_hits=near_hits,
    )


__all__ = [
    "blocker",
    "build_search_report",
    "report_status",
]
