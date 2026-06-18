"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/released_projection_projection.py

Released-product post-cut projection helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.nickases.scanning import EvaluatedMatch
from dnadesign.cruncher.nickases.scanning import (
    enumerate_site_instances as enumerate_nickase_site_instances,
)
from dnadesign.cruncher.release_enzymes.scanning import ReleaseEvaluatedMatch
from dnadesign.cruncher.release_enzymes.scanning import (
    enumerate_site_instances as enumerate_release_site_instances,
)
from dnadesign.cruncher.snapback.models import SnapbackIssue
from dnadesign.cruncher.snapback.publication_support import complement_sequence
from dnadesign.cruncher.snapback.released_projection_common import (
    build_issue,
    physical_nicked_strand_from_event,
    provenance_source_constraint,
)
from dnadesign.cruncher.snapback.released_projection_models import (
    ReleasedProductBaseProvenance,
    ReleasedProductProjection,
)
from dnadesign.cruncher.snapback.released_route_policy import (
    ReleasedActiveStrand,
    ReleasedRouteFamily,
    route_family_active_strand,
    route_family_final_geometry_source,
)
from dnadesign.cruncher.snapback.released_spec_models import ReleasedSnapbackConstraintsSpec

_COMPLEMENT_BASE = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
}


def released_duplex_overlap_pairing_issues(
    *,
    retained_partner_sequence: str,
    active_product_sequence: str,
    coordinate_offset: int,
    release_top_cut_precursor: int,
    release_bottom_cut_precursor: int,
    active_strand: ReleasedActiveStrand = "bottom",
) -> list[SnapbackIssue]:
    overlap_length = max(0, min(release_top_cut_precursor, release_bottom_cut_precursor) - coordinate_offset)
    if overlap_length == 0:
        return []
    retained_overlap = retained_partner_sequence[coordinate_offset : coordinate_offset + overlap_length]
    active_overlap = active_product_sequence[:overlap_length]
    mismatch_positions = [
        index
        for index, (retained_base, active_base) in enumerate(zip(retained_overlap, active_overlap, strict=True))
        if _COMPLEMENT_BASE[active_base] != retained_base
    ]
    if not mismatch_positions:
        return []
    return [
        build_issue(
            "POST_RELEASE_DUPLEX_PAIRING_INVALID",
            (
                "The retained partner fragment and active product are not "
                "Watson-Crick paired across the surviving duplex overlap."
            ),
            active_strand=active_strand,
            overlap_length=overlap_length,
            mismatch_positions=mismatch_positions,
            retained_partner_overlap=retained_overlap,
            active_product_overlap=active_overlap,
        )
    ]


def project_released_product(
    *,
    precursor_top_strand: str,
    nick_match: EvaluatedMatch,
    release_match: ReleaseEvaluatedMatch,
    constraints: ReleasedSnapbackConstraintsSpec,
    coordinate_offset: int,
    route_family: ReleasedRouteFamily,
) -> tuple[ReleasedProductProjection | None, list[SnapbackIssue]]:
    precursor_length = len(precursor_top_strand)
    nick_coordinate_precursor = nick_match.nick.boundary_context
    active_strand = route_family_active_strand(route_family)
    retained_partner_strand: ReleasedActiveStrand = "top" if active_strand == "bottom" else "bottom"
    active_cut_precursor = (
        release_match.cut.top_cut_boundary if active_strand == "top" else release_match.cut.bottom_cut_boundary
    )
    issues: list[SnapbackIssue] = []
    if constraints.require_release_site_downstream_of_nick and release_match.site.start < nick_coordinate_precursor:
        issues.append(
            build_issue(
                "RELEASE_SITE_NOT_DOWNSTREAM_OF_NICK",
                "The release recognition site must start downstream of the intended nick boundary.",
                release_site_start=release_match.site.start,
                nick_boundary=nick_coordinate_precursor,
            )
        )
    if release_match.cut.top_cut_boundary < 0 or release_match.cut.bottom_cut_boundary < 0:
        issues.append(
            build_issue(
                "RELEASE_CUT_OUTSIDE_PRECURSOR",
                "Release cut coordinates must be non-negative in precursor space.",
                release_top_cut=release_match.cut.top_cut_boundary,
                release_bottom_cut=release_match.cut.bottom_cut_boundary,
            )
        )
    if (
        release_match.cut.top_cut_boundary > precursor_length
        or release_match.cut.bottom_cut_boundary > precursor_length
    ):
        issues.append(
            build_issue(
                "RELEASE_CUT_OUTSIDE_PRECURSOR",
                "Release cut coordinates must stay inside precursor length.",
                release_top_cut=release_match.cut.top_cut_boundary,
                release_bottom_cut=release_match.cut.bottom_cut_boundary,
                precursor_length=precursor_length,
            )
        )
    if constraints.require_complete_downstream_fragment_separation and (
        release_match.cut.top_cut_boundary >= precursor_length
        or release_match.cut.bottom_cut_boundary >= precursor_length
    ):
        issues.append(
            build_issue(
                "RELEASE_DOES_NOT_SEPARATE_DOWNSTREAM_FRAGMENT",
                "The release cut must leave a physically separate downstream fragment on both strands.",
                release_top_cut=release_match.cut.top_cut_boundary,
                release_bottom_cut=release_match.cut.bottom_cut_boundary,
                precursor_length=precursor_length,
            )
        )
    if issues:
        return None, issues

    retained_partner_sequence = (
        precursor_top_strand[:nick_coordinate_precursor]
        if retained_partner_strand == "top"
        else complement_sequence(precursor_top_strand[:nick_coordinate_precursor])
    )
    retained_partner_scan_sequence = precursor_top_strand[:nick_coordinate_precursor]
    active_product_sequence = (
        precursor_top_strand[coordinate_offset:active_cut_precursor]
        if active_strand == "top"
        else complement_sequence(precursor_top_strand[coordinate_offset:active_cut_precursor])
    )
    duplex_pairing_issues = released_duplex_overlap_pairing_issues(
        retained_partner_sequence=retained_partner_sequence,
        active_product_sequence=active_product_sequence,
        coordinate_offset=coordinate_offset,
        release_top_cut_precursor=nick_coordinate_precursor,
        release_bottom_cut_precursor=active_cut_precursor,
        active_strand=active_strand,
    )
    if duplex_pairing_issues:
        return None, duplex_pairing_issues
    retained_nickase_matches = enumerate_nickase_site_instances(
        retained_partner_scan_sequence,
        coordinate_offset=0,
        entry=nick_match.variant,
        use_vendor_diagram=True,
    )
    nickase_site_survives = any(
        match.nick.boundary == nick_match.nick.boundary
        and physical_nicked_strand_from_event(match.nick) == physical_nicked_strand_from_event(nick_match.nick)
        and match.site.orientation == nick_match.site.orientation
        for match in retained_nickase_matches
    )
    retained_release_matches = enumerate_release_site_instances(
        retained_partner_scan_sequence,
        coordinate_offset=0,
        entry=release_match.variant,
    )
    release_site_survives = any(
        match.key() == release_match.key()
        and match.site.matched_span_sequence == release_match.site.matched_span_sequence
        for match in retained_release_matches
    )
    if not constraints.allow_post_release_loss_of_nickase_site and not nickase_site_survives:
        return (
            None,
            [
                build_issue(
                    "POST_RELEASE_NICKASE_SITE_LOST",
                    "The released-product spec requires the nickase site to survive post-release.",
                    variant_id=nick_match.variant.id,
                )
            ],
        )
    if not constraints.allow_post_release_loss_of_release_site and not release_site_survives:
        return (
            None,
            [
                build_issue(
                    "POST_RELEASE_RELEASE_SITE_LOST",
                    "The released-product spec requires the release site to survive post-release.",
                    variant_id=release_match.variant.variant_id,
                )
            ],
        )
    active_product_provenance = [
        ReleasedProductBaseProvenance(
            active_index=index,
            precursor_strand=active_strand,
            precursor_index=coordinate_offset + index,
            source_constraint=provenance_source_constraint(
                nick_match=nick_match,
                precursor_index=coordinate_offset + index,
            ),
        )
        for index in range(len(active_product_sequence))
    ]
    return (
        ReleasedProductProjection(
            final_geometry_source=route_family_final_geometry_source(route_family),
            route_family=route_family,
            physical_nicked_strand=physical_nicked_strand_from_event(nick_match.nick),
            active_strand=active_strand,
            retained_partner_strand=retained_partner_strand,
            precursor_top_strand=precursor_top_strand,
            precursor_length=precursor_length,
            nick_coordinate_precursor=nick_coordinate_precursor,
            release_top_cut_precursor=release_match.cut.top_cut_boundary,
            release_bottom_cut_precursor=release_match.cut.bottom_cut_boundary,
            retained_partner_sequence=retained_partner_sequence,
            retained_partner_length_nt=len(retained_partner_sequence),
            active_product_sequence=active_product_sequence,
            active_product_span=(0, active_cut_precursor - coordinate_offset),
            active_product_length_nt=len(active_product_sequence),
            active_product_provenance=active_product_provenance,
            rebased_nick_boundary=nick_match.nick.boundary,
            nickase_site_survives_post_release=nickase_site_survives,
            release_site_survives_post_release=release_site_survives,
        ),
        [],
    )
