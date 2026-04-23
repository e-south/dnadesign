"""
Evaluation adapter for released-product target-search pairs.
"""

from __future__ import annotations

from dnadesign.cruncher.snapback.released_projection import evaluate_released_precursor
from dnadesign.cruncher.snapback.released_projection_models import (
    build_release_catalog_info,
    build_released_nickase_catalog_info,
)
from dnadesign.cruncher.snapback.released_route_policy import ReleasedRouteFamily
from dnadesign.cruncher.snapback.released_search.placement_models import NickPlacement, ReleasePlacement
from dnadesign.cruncher.snapback.released_search.precursor_builder import build_precursor_sequence
from dnadesign.cruncher.snapback.released_search.reporting import blocker
from dnadesign.cruncher.snapback.released_search_models import (
    ReleasedTargetSearchHit,
    SingleNickReleasedTargetSearchRequest,
)
from dnadesign.cruncher.snapback.released_spec_models import (
    ReleasedFinalTargetGeometry,
    ReleasedSnapbackConstraintsSpec,
)


def hit_from_evaluation(
    *,
    boundary: int,
    hit_kind: str,
    precursor_top_strand: str,
    nick_placement: NickPlacement,
    release_placement: ReleasePlacement,
    evaluation: object,
) -> ReleasedTargetSearchHit | None:
    if (
        getattr(evaluation, "candidate", None) is None
        or getattr(evaluation, "projection", None) is None
        or getattr(evaluation, "pre_nick_match", None) is None
        or getattr(evaluation, "release_match", None) is None
    ):
        return None
    sacrificial_downstream_tail_nt = len(precursor_top_strand) - max(
        evaluation.release_match.cut.top_cut_boundary,
        evaluation.release_match.cut.bottom_cut_boundary,
    )
    return ReleasedTargetSearchHit(
        rank=1,
        hit_kind=hit_kind,  # type: ignore[arg-type]
        route_family=evaluation.projection.route_family,
        active_strand=evaluation.projection.active_strand,
        physical_nicked_strand=evaluation.projection.physical_nicked_strand,
        nickase_variant_id=nick_placement.entry.id,
        release_variant_id=release_placement.entry.variant_id,
        intended_nick_site_orientation=nick_placement.orientation,  # type: ignore[arg-type]
        intended_nick_site_sequence=evaluation.pre_nick_match.site.matched_span_sequence,
        release_site_orientation=release_placement.orientation,  # type: ignore[arg-type]
        release_site_sequence=evaluation.release_match.site.matched_span_sequence,
        nick_boundary_from_left=boundary,
        active_product_input_length_nt=evaluation.candidate.active_product_input_length_nt,
        active_product_length_nt=evaluation.candidate.active_product_length_nt,
        precursor_length_nt=len(precursor_top_strand),
        sacrificial_downstream_tail_nt=sacrificial_downstream_tail_nt,
        extra_nick_event_count=evaluation.candidate.extra_nick_event_count,
        extra_target_strand_nick_count=evaluation.candidate.extra_target_strand_nick_count,
        precursor_top_strand=precursor_top_strand,
        pre_nick_site=evaluation.pre_nick_match.site,
        pre_nick_event=evaluation.pre_nick_match.nick,
        release_site=evaluation.release_match.site,
        release_event=evaluation.release_match.cut,
        nickase=build_released_nickase_catalog_info(nick_placement.entry),
        release_enzyme=build_release_catalog_info(release_placement.entry),
        projection=evaluation.projection,
        final_candidate=evaluation.candidate,
    )


def search_pair(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    route_family: ReleasedRouteFamily = "bottom_active_from_top_nick",
    nick_placement: NickPlacement,
    release_placement: ReleasePlacement,
    blocker_counts: dict[str, int],
    build_precursor_sequence_fn=build_precursor_sequence,
    evaluate_released_precursor_fn=evaluate_released_precursor,
    hit_from_evaluation_fn=hit_from_evaluation,
    blocker_fn=blocker,
) -> tuple[ReleasedTargetSearchHit | None, list[ReleasedTargetSearchHit]]:
    target = request.target
    if request.search.retained_side != "upstream" or request.search.stage_order != "nick_then_release":
        raise ValueError(
            "released-product target-search only supports retained_side=upstream and stage_order=nick_then_release."
        )
    if not release_placement.starts_downstream_of_boundary():
        blocker_fn(blocker_counts, "RELEASE_OVERLAPS_REQUIRED_TARGET_REGION")
        return None, []
    target_boundary = target.nick_boundary_from_left
    boundaries = [
        (target_boundary, "exact"),
        *[
            (boundary_value, "nearest")
            for boundary_value in range(
                max(0, target_boundary - request.search.near_boundary_search_limit),
                target_boundary + request.search.near_boundary_search_limit + 1,
            )
            if boundary_value != target_boundary
        ],
    ]
    exact_hit: ReleasedTargetSearchHit | None = None
    near_hits: list[ReleasedTargetSearchHit] = []
    for boundary, hit_kind in boundaries:
        built_precursor = build_precursor_sequence_fn(
            boundary=boundary,
            target=target,
            nick_placement=nick_placement,
            release_placement=release_placement,
            allow_precut_footprint_outside_active_product=request.search.allow_precut_footprint_outside_active_product,
        )
        if built_precursor.precursor is None:
            blocker_fn(blocker_counts, built_precursor.blocker_code or "FOOTPRINT_NOT_CONSTRUCTABLE")
            continue
        precursor = built_precursor.precursor
        local_target = ReleasedFinalTargetGeometry(
            nick_boundary_from_left=boundary,
            paired_bp=target.paired_bp,
            cap_nt=target.cap_nt,
        )
        evaluation = evaluate_released_precursor_fn(
            precursor_top_strand=precursor.top_strand,
            nick_entry=nick_placement.entry,
            release_entry=release_placement.entry,
            target=local_target,
            constraints=ReleasedSnapbackConstraintsSpec(
                allow_post_release_loss_of_nickase_site=request.search.allow_post_release_loss_of_nickase_site,
                allow_post_release_loss_of_release_site=True,
                require_release_site_downstream_of_nick=True,
                require_complete_downstream_fragment_separation=True,
            ),
            precursor_coordinate_offset=precursor.coordinate_offset,
            route_family=route_family,
            allow_precut_footprint_outside_active_product=request.search.allow_precut_footprint_outside_active_product,
        )
        if evaluation.status == "satisfied":
            hit = hit_from_evaluation_fn(
                boundary=boundary,
                hit_kind=hit_kind,
                precursor_top_strand=precursor.top_strand,
                nick_placement=nick_placement,
                release_placement=release_placement,
                evaluation=evaluation,
            )
            if hit_kind == "exact":
                exact_hit = hit
                continue
            if hit is not None:
                near_hits.append(hit)
            continue
        for issue in evaluation.issues:
            if issue.code == "RELEASE_DOES_NOT_SEPARATE_DOWNSTREAM_FRAGMENT":
                blocker_fn(blocker_counts, "RELEASE_DOES_NOT_SEPARATE_DOWNSTREAM_FRAGMENT")
            elif evaluation.status == "post_release_projection_failed":
                blocker_fn(blocker_counts, "POST_RELEASE_PROJECTION_INVALID")
            elif issue.code == "ACTIVE_PRODUCT_TOO_SHORT":
                blocker_fn(blocker_counts, "ACTIVE_PRODUCT_TOO_SHORT")
            else:
                blocker_fn(blocker_counts, "SNAPBACK_PAIRING_UNSAT")
    return exact_hit, near_hits


__all__ = ["hit_from_evaluation", "search_pair"]

