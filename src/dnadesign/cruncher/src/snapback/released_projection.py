"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_projection.py

Released-product precursor projection and explicit evaluator reuse.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry
from dnadesign.cruncher.nickases.scanning import EvaluatedMatch
from dnadesign.cruncher.nickases.scanning import (
    enumerate_site_instances as enumerate_nickase_site_instances,
)
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeEntry
from dnadesign.cruncher.release_enzymes.scanning import (
    ReleaseEvaluatedMatch,
)
from dnadesign.cruncher.release_enzymes.scanning import (
    enumerate_site_instances as enumerate_release_site_instances,
)
from dnadesign.cruncher.snapback.models import SnapbackIssue
from dnadesign.cruncher.snapback.released_projection_candidate import (
    build_projected_snapback_candidate,
    candidate_failure_status,
    evaluate_projected_candidate,
)
from dnadesign.cruncher.snapback.released_projection_models import (
    ReleasedFinalCandidate,
    ReleasedProductProjection,
)
from dnadesign.cruncher.snapback.released_projection_projection import (
    project_released_product,
    released_duplex_overlap_pairing_issues,
)
from dnadesign.cruncher.snapback.released_projection_selection import (
    select_nick_match,
    select_release_match,
    validate_pre_nick_site_origin_boundary,
)
from dnadesign.cruncher.snapback.released_route_policy import (
    ReleasedRouteFamily,
    route_family_active_strand,
)
from dnadesign.cruncher.snapback.released_spec_models import (
    ReleasedFinalTargetGeometry,
    ReleasedSnapbackConstraintsSpec,
)


@dataclass(frozen=True)
class ReleasedEvaluationResult:
    status: str
    issues: list[SnapbackIssue]
    pre_nick_match: EvaluatedMatch | None
    release_match: ReleaseEvaluatedMatch | None
    projection: ReleasedProductProjection | None
    candidate: ReleasedFinalCandidate | None


def _result(
    *,
    status: str,
    issues: list[SnapbackIssue],
    pre_nick_match: EvaluatedMatch | None,
    release_match: ReleaseEvaluatedMatch | None,
    projection: ReleasedProductProjection | None,
    candidate: ReleasedFinalCandidate | None,
) -> ReleasedEvaluationResult:
    return ReleasedEvaluationResult(
        status=status,
        issues=issues,
        pre_nick_match=pre_nick_match,
        release_match=release_match,
        projection=projection,
        candidate=candidate,
    )


def evaluate_released_precursor(
    *,
    precursor_top_strand: str,
    nick_entry: NickaseCatalogEntry,
    release_entry: ReleaseEnzymeEntry,
    target: ReleasedFinalTargetGeometry,
    constraints: ReleasedSnapbackConstraintsSpec,
    nick_intended_site_sequence: str | None = None,
    release_intended_site_sequence: str | None = None,
    precursor_coordinate_offset: int = 0,
    route_family: ReleasedRouteFamily = "bottom_active_from_top_nick",
    allow_precut_footprint_outside_active_product: bool = False,
) -> ReleasedEvaluationResult:
    nick_matches = enumerate_nickase_site_instances(
        precursor_top_strand,
        coordinate_offset=precursor_coordinate_offset,
        entry=nick_entry,
        use_vendor_diagram=allow_precut_footprint_outside_active_product,
    )
    nick_match, nick_issues = select_nick_match(
        matches=nick_matches,
        target=target,
        intended_site_sequence=nick_intended_site_sequence,
        route_family=route_family,
    )
    if nick_match is None:
        return _result(
            status="invalid_precursor",
            issues=nick_issues,
            pre_nick_match=None,
            release_match=None,
            projection=None,
            candidate=None,
        )
    if not allow_precut_footprint_outside_active_product:
        nick_boundary_issues = validate_pre_nick_site_origin_boundary(nick_match)
        if nick_boundary_issues:
            return _result(
                status="invalid_precursor",
                issues=nick_boundary_issues,
                pre_nick_match=nick_match,
                release_match=None,
                projection=None,
                candidate=None,
            )
    target_active_product_length = target.nick_boundary_from_left + (2 * target.paired_bp) + target.cap_nt
    release_matches = enumerate_release_site_instances(
        precursor_top_strand,
        coordinate_offset=precursor_coordinate_offset,
        entry=release_entry,
    )
    release_match, release_issues = select_release_match(
        matches=release_matches,
        target_active_product_length=target_active_product_length,
        coordinate_offset=precursor_coordinate_offset,
        active_strand=route_family_active_strand(route_family),
        intended_site_sequence=release_intended_site_sequence,
    )
    if release_match is None:
        return _result(
            status="no_release_path",
            issues=release_issues,
            pre_nick_match=nick_match,
            release_match=None,
            projection=None,
            candidate=None,
        )
    projection, projection_issues = project_released_product(
        precursor_top_strand=precursor_top_strand,
        nick_match=nick_match,
        release_match=release_match,
        constraints=constraints,
        coordinate_offset=precursor_coordinate_offset,
        route_family=route_family,
    )
    if projection is None:
        return _result(
            status="no_release_path",
            issues=projection_issues,
            pre_nick_match=nick_match,
            release_match=release_match,
            projection=None,
            candidate=None,
        )
    candidate, candidate_issues = evaluate_projected_candidate(
        projection=projection,
        original_nick_match=nick_match,
        nick_entry=nick_entry,
        target=target,
    )
    if candidate is None:
        return _result(
            status=candidate_failure_status(candidate_issues),
            issues=candidate_issues,
            pre_nick_match=nick_match,
            release_match=release_match,
            projection=projection,
            candidate=None,
        )
    return _result(
        status="satisfied",
        issues=[],
        pre_nick_match=nick_match,
        release_match=release_match,
        projection=projection,
        candidate=candidate,
    )


_released_duplex_overlap_pairing_issues = released_duplex_overlap_pairing_issues


__all__ = ["ReleasedEvaluationResult", "build_projected_snapback_candidate", "evaluate_released_precursor"]
