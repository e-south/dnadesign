"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/released_projection_selection.py

Match selection and precursor-boundary validation helpers for released-product.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.nickases.scanning import EvaluatedMatch
from dnadesign.cruncher.release_enzymes.scanning import ReleaseEvaluatedMatch
from dnadesign.cruncher.snapback.models import SnapbackIssue
from dnadesign.cruncher.snapback.released_projection_common import (
    build_issue,
    physical_nicked_strand_from_event,
)
from dnadesign.cruncher.snapback.released_route_policy import (
    ReleasedActiveStrand,
    ReleasedRouteFamily,
    route_family_physical_nicked_strand,
)
from dnadesign.cruncher.snapback.released_spec_models import ReleasedFinalTargetGeometry


def select_nick_match(
    *,
    matches: list[EvaluatedMatch],
    target: ReleasedFinalTargetGeometry,
    intended_site_sequence: str | None,
    route_family: ReleasedRouteFamily,
) -> tuple[EvaluatedMatch | None, list[SnapbackIssue]]:
    selected = [match for match in matches if match.nick.boundary == target.nick_boundary_from_left]
    expected_physical_nicked_strand = route_family_physical_nicked_strand(route_family)
    selected = [
        match for match in selected if physical_nicked_strand_from_event(match.nick) == expected_physical_nicked_strand
    ]
    if intended_site_sequence is not None:
        selected = [match for match in selected if match.site.matched_span_sequence == intended_site_sequence]
    if not selected:
        return (
            None,
            [
                build_issue(
                    "PRE_NICK_SITE_NOT_FOUND",
                    "No pre-nick site matched the requested boundary and route-family nicked strand on the precursor.",
                    nick_boundary_from_left=target.nick_boundary_from_left,
                    route_family=route_family,
                )
            ],
        )
    if len(selected) > 1:
        return (
            None,
            [
                build_issue(
                    "PRE_NICK_SITE_AMBIGUOUS",
                    "Multiple pre-nick sites matched the requested boundary on the precursor.",
                    match_count=len(selected),
                )
            ],
        )
    return selected[0], []


def select_release_match(
    *,
    matches: list[ReleaseEvaluatedMatch],
    target_active_product_length: int,
    coordinate_offset: int,
    active_strand: ReleasedActiveStrand,
    intended_site_sequence: str | None,
) -> tuple[ReleaseEvaluatedMatch | None, list[SnapbackIssue]]:
    selected = [
        match
        for match in matches
        if (
            (match.cut.top_cut_boundary if active_strand == "top" else match.cut.bottom_cut_boundary)
            - coordinate_offset
        )
        == target_active_product_length
    ]
    if intended_site_sequence is not None:
        selected = [match for match in selected if match.site.matched_span_sequence == intended_site_sequence]
    if not selected:
        return (
            None,
            [
                build_issue(
                    "NO_RELEASE_MATCH_FOR_TARGET_LENGTH",
                    "No release site produced the retained active-strand length required by the final target.",
                    active_strand=active_strand,
                    required_active_product_length=target_active_product_length,
                )
            ],
        )
    if len(selected) > 1:
        return (
            None,
            [
                build_issue(
                    "RELEASE_SITE_AMBIGUOUS",
                    "Multiple release sites produced the same retained active-strand length.",
                    match_count=len(selected),
                    active_strand=active_strand,
                    required_active_product_length=target_active_product_length,
                )
            ],
        )
    return selected[0], []


def validate_pre_nick_site_origin_boundary(nick_match: EvaluatedMatch) -> list[SnapbackIssue]:
    if (
        nick_match.site.local_start is not None
        and nick_match.site.local_end is not None
        and nick_match.site.local_start < 0
    ):
        if nick_match.site.local_end > 0:
            return [
                build_issue(
                    "PRE_NICK_SITE_OVERLAPS_ACTIVE_STRAND",
                    (
                        "The released-product lane rejects pre-nick recognition sites "
                        "that straddle the active-strand origin."
                    ),
                    local_start=nick_match.site.local_start,
                    local_end=nick_match.site.local_end,
                    variant_id=nick_match.variant.id,
                )
            ]
        return [
            build_issue(
                "PRE_NICK_SITE_LEFT_OF_ORIGIN",
                "The released-product lane rejects pre-nick recognition sites that begin left of logical origin 0.",
                local_start=nick_match.site.local_start,
                local_end=nick_match.site.local_end,
                variant_id=nick_match.variant.id,
            )
        ]
    return []
