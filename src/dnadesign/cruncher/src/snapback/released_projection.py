"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_projection.py

Released-product precursor projection and explicit evaluator reuse.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry, NickEvent, iupac_bases_for_symbol
from dnadesign.cruncher.nickases.scanning import (
    EvaluatedMatch,
    display_footprint_for_orientation,
    display_motif_for_orientation,
)
from dnadesign.cruncher.nickases.scanning import (
    build_evaluated_match as build_nick_evaluated_match,
)
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
from dnadesign.cruncher.snapback.models import CoordinateSpan, SnapbackCandidateDesign, SnapbackIssue
from dnadesign.cruncher.snapback.planner import evaluate_snapback_candidate
from dnadesign.cruncher.snapback.publication_support import complement_sequence
from dnadesign.cruncher.snapback.released_projection_models import (
    ReleasedFinalCandidate,
    ReleasedProductBaseProvenance,
    ReleasedProductProjection,
)
from dnadesign.cruncher.snapback.released_route_policy import (
    ReleasedActiveStrand,
    ReleasedRouteFamily,
    route_family_active_strand,
    route_family_final_geometry_source,
    route_family_physical_nicked_strand,
)
from dnadesign.cruncher.snapback.released_spec_models import (
    ReleasedFinalTargetGeometry,
    ReleasedSnapbackConstraintsSpec,
)

_COMPLEMENT_BASE = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
}


def _issue(code: str, message: str, **details: object) -> SnapbackIssue:
    return SnapbackIssue(code=code, message=message, details=details)


def _physical_nicked_strand_from_event(event: NickEvent) -> ReleasedActiveStrand:
    return "top" if event.strand == "primary" else "bottom"


def _site_symbols_for_match(match: EvaluatedMatch) -> str:
    site_span_len = match.site.end - match.site.start
    if site_span_len == match.variant.resolved_vendor_diagram_len:
        return display_footprint_for_orientation(match.variant, orientation=match.site.orientation)
    return display_motif_for_orientation(match.variant, orientation=match.site.orientation)


def _provenance_source_constraint(
    *,
    nick_match: EvaluatedMatch,
    precursor_index: int,
) -> str:
    if not (nick_match.site.start <= precursor_index < nick_match.site.end):
        return "user_sequence"
    site_symbol = _site_symbols_for_match(nick_match)[precursor_index - nick_match.site.start]
    if len(iupac_bases_for_symbol(site_symbol)) > 1:
        return "degenerate_motif_base"
    return "fixed_motif_base"


@dataclass(frozen=True)
class ReleasedEvaluationResult:
    status: str
    issues: list[SnapbackIssue]
    pre_nick_match: EvaluatedMatch | None
    release_match: ReleaseEvaluatedMatch | None
    projection: ReleasedProductProjection | None
    candidate: ReleasedFinalCandidate | None


def _select_nick_match(
    *,
    matches: list[EvaluatedMatch],
    target: ReleasedFinalTargetGeometry,
    intended_site_sequence: str | None,
    route_family: ReleasedRouteFamily,
) -> tuple[EvaluatedMatch | None, list[SnapbackIssue]]:
    selected = [match for match in matches if match.nick.boundary == target.nick_boundary_from_left]
    expected_physical_nicked_strand = route_family_physical_nicked_strand(route_family)
    selected = [
        match for match in selected if _physical_nicked_strand_from_event(match.nick) == expected_physical_nicked_strand
    ]
    if intended_site_sequence is not None:
        selected = [match for match in selected if match.site.matched_span_sequence == intended_site_sequence]
    if not selected:
        return (
            None,
            [
                _issue(
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
                _issue(
                    "PRE_NICK_SITE_AMBIGUOUS",
                    "Multiple pre-nick sites matched the requested boundary on the precursor.",
                    match_count=len(selected),
                )
            ],
        )
    return selected[0], []


def _select_release_match(
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
                _issue(
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
                _issue(
                    "RELEASE_SITE_AMBIGUOUS",
                    "Multiple release sites produced the same retained active-strand length.",
                    match_count=len(selected),
                    active_strand=active_strand,
                    required_active_product_length=target_active_product_length,
                )
            ],
        )
    return selected[0], []


def _validate_pre_nick_site_origin_boundary(nick_match: EvaluatedMatch) -> list[SnapbackIssue]:
    if (
        nick_match.site.local_start is not None
        and nick_match.site.local_end is not None
        and nick_match.site.local_start < 0
    ):
        if nick_match.site.local_end > 0:
            return [
                _issue(
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
            _issue(
                "PRE_NICK_SITE_LEFT_OF_ORIGIN",
                "The released-product lane rejects pre-nick recognition sites that begin left of logical origin 0.",
                local_start=nick_match.site.local_start,
                local_end=nick_match.site.local_end,
                variant_id=nick_match.variant.id,
            )
        ]
    return []


def _build_projected_origin_event(original: NickEvent, *, rebased_boundary: int) -> NickEvent:
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


def _released_duplex_overlap_pairing_issues(
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
        _issue(
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


def _project_released_product(
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
            _issue(
                "RELEASE_SITE_NOT_DOWNSTREAM_OF_NICK",
                "The release recognition site must start downstream of the intended nick boundary.",
                release_site_start=release_match.site.start,
                nick_boundary=nick_coordinate_precursor,
            )
        )
    if release_match.cut.top_cut_boundary < 0 or release_match.cut.bottom_cut_boundary < 0:
        issues.append(
            _issue(
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
            _issue(
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
            _issue(
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
    duplex_pairing_issues = _released_duplex_overlap_pairing_issues(
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
        and _physical_nicked_strand_from_event(match.nick) == _physical_nicked_strand_from_event(nick_match.nick)
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
                _issue(
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
                _issue(
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
            source_constraint=_provenance_source_constraint(
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
            physical_nicked_strand=_physical_nicked_strand_from_event(nick_match.nick),
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


def _precursor_extra_nick_events(
    *,
    precursor_top_strand: str,
    nick_entry: NickaseCatalogEntry,
    original_nick_match: EvaluatedMatch,
) -> tuple[list[NickEvent], list[NickEvent]]:
    coordinate_offset = original_nick_match.nick.boundary_context - original_nick_match.nick.boundary
    all_matches = enumerate_nickase_site_instances(
        precursor_top_strand,
        coordinate_offset=coordinate_offset,
        entry=nick_entry,
        use_vendor_diagram=True,
    )
    selected_key = original_nick_match.key()
    extra_nick_events = [match.nick for match in all_matches if match.key() != selected_key]
    extra_target_strand_nicks = [
        event for event in extra_nick_events if event.strand == original_nick_match.nick.strand
    ]
    return extra_nick_events, extra_target_strand_nicks


def _build_projected_selected_match(
    *,
    entry: NickaseCatalogEntry,
    input_sequence: str,
    nick_boundary: int,
    intended_strand: str,
) -> EvaluatedMatch:
    base = input_sequence[nick_boundary]
    if intended_strand == "primary":
        projected_entry = entry.model_copy(
            update={
                "motif_top_5to3": base,
                "motif_len": 1,
                "top_cut_offset": 0,
                "bottom_cut_offset": None,
            }
        )
        return build_nick_evaluated_match(
            entry=projected_entry,
            start=nick_boundary,
            orientation="forward",
            coordinate_offset=0,
            matched_span_sequence=base,
        )
    projected_entry = entry.model_copy(
        update={
            "motif_top_5to3": base,
            "motif_len": 1,
            "top_cut_offset": None,
            "bottom_cut_offset": 0,
        }
    )
    return build_nick_evaluated_match(
        entry=projected_entry,
        start=nick_boundary,
        orientation="forward",
        coordinate_offset=0,
        matched_span_sequence=base,
    )


def build_projected_snapback_candidate(
    *,
    projection: ReleasedProductProjection,
    nick_entry: NickaseCatalogEntry,
    target: ReleasedFinalTargetGeometry,
    intended_strand: str,
) -> tuple[SnapbackCandidateDesign | None, list[SnapbackIssue]]:
    active_product_length = projection.active_product_length_nt
    input_length = active_product_length - target.paired_bp
    expected_input_length = target.nick_boundary_from_left + target.paired_bp + target.cap_nt
    if input_length != expected_input_length:
        return (
            None,
            [
                _issue(
                    "ACTIVE_PRODUCT_DOES_NOT_PROJECT_TO_FINAL_TARGET",
                    "The retained active-strand length does not project to the requested final target geometry.",
                    active_strand=projection.active_strand,
                    active_product_length=active_product_length,
                    input_length=input_length,
                    required_input_length=expected_input_length,
                )
            ],
        )
    if input_length <= target.nick_boundary_from_left or active_product_length <= input_length:
        return (
            None,
            [
                _issue(
                    "ACTIVE_PRODUCT_TOO_SHORT",
                    "The retained active strand could not be partitioned into input sequence and foldback arm.",
                    active_strand=projection.active_strand,
                    active_product_length=active_product_length,
                    input_length=input_length,
                    nick_boundary=target.nick_boundary_from_left,
                )
            ],
        )
    input_sequence = projection.active_product_sequence[:input_length]
    foldback_arm = projection.active_product_sequence[input_length:]
    selected_match = _build_projected_selected_match(
        entry=nick_entry,
        input_sequence=input_sequence,
        nick_boundary=projection.rebased_nick_boundary,
        intended_strand=intended_strand,
    )
    all_matches: list[EvaluatedMatch] = [selected_match]

    return evaluate_snapback_candidate(
        input_sequence=input_sequence,
        protected_region=CoordinateSpan(
            start=selected_match.site.start,
            end=selected_match.site.end,
        ),
        pre_nick_duplex_window=CoordinateSpan(start=0, end=input_length),
        retained_homology_window=CoordinateSpan(
            start=projection.rebased_nick_boundary,
            end=projection.rebased_nick_boundary + target.paired_bp,
        ),
        cap_sequence="",
        foldback_arm=foldback_arm,
        homology_max_mismatches=0,
        terminal_ligatable_duplex_min=target.paired_bp,
        terminal_ligatable_duplex_max=target.paired_bp,
        max_uninterrupted_duplex_bp=target.paired_bp,
        max_added_nt=len(foldback_arm),
        gc_bounds=None,
        max_homopolymer_run_allowed=None,
        intended_match=selected_match,
        site_mutation_count=0,
        all_matches=all_matches,
        forbid_additional_target_strand_nicks=False,
        forbid_any_additional_nicks=False,
    )


def _evaluate_projected_candidate(
    *,
    projection: ReleasedProductProjection,
    original_nick_match: EvaluatedMatch,
    nick_entry: NickaseCatalogEntry,
    target: ReleasedFinalTargetGeometry,
) -> tuple[ReleasedFinalCandidate | None, list[SnapbackIssue]]:
    candidate, issues = build_projected_snapback_candidate(
        projection=projection,
        nick_entry=nick_entry,
        target=target,
        intended_strand=original_nick_match.nick.strand,
    )
    if issues or candidate is None:
        return None, issues
    extra_nick_events, extra_target_strand_nicks = _precursor_extra_nick_events(
        precursor_top_strand=projection.precursor_top_strand,
        nick_entry=nick_entry,
        original_nick_match=original_nick_match,
    )
    projected_origin_event = _build_projected_origin_event(
        original_nick_match.nick,
        rebased_boundary=projection.rebased_nick_boundary,
    )
    return (
        ReleasedFinalCandidate(
            final_geometry_source=projection.final_geometry_source,
            route_family=projection.route_family,
            physical_nicked_strand=projection.physical_nicked_strand,
            active_strand=projection.active_strand,
            designed_sequence=projection.active_product_sequence,
            input_sequence=candidate.input_sequence,
            foldback_arm=candidate.foldback_arm,
            nick_boundary_from_left=candidate.nick_boundary_from_left,
            paired_bp=candidate.paired_bp,
            cap_nt=candidate.cap_nt,
            source_cap_nt=len(candidate.source_cap_sequence),
            cap_extension_nt=candidate.cap_extension_nt,
            active_product_length_nt=projection.active_product_length_nt,
            active_product_input_length_nt=len(candidate.input_sequence),
            mismatch_count=candidate.mismatch_count,
            mismatch_positions=list(candidate.mismatch_positions),
            terminal_ligatable_duplex_bp=candidate.terminal_ligatable_duplex_bp,
            max_uninterrupted_duplex_bp=candidate.max_uninterrupted_duplex_bp,
            extra_nick_event_count=len(extra_nick_events),
            extra_target_strand_nick_count=len(extra_target_strand_nicks),
            gc_fraction_added=candidate.gc_fraction_added,
            max_homopolymer_run_added=candidate.max_homopolymer_run_added,
            projected_origin_event=projected_origin_event,
            extra_target_strand_nicks=extra_target_strand_nicks,
            extra_nick_events=extra_nick_events,
            post_nick_sequence=candidate.post_nick_sequence,
            nickase_site_survives_post_release=projection.nickase_site_survives_post_release,
            release_site_survives_post_release=projection.release_site_survives_post_release,
            active_product_provenance=list(projection.active_product_provenance),
        ),
        [],
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
    nick_match, nick_issues = _select_nick_match(
        matches=nick_matches,
        target=target,
        intended_site_sequence=nick_intended_site_sequence,
        route_family=route_family,
    )
    if nick_match is None:
        return ReleasedEvaluationResult(
            status="invalid_precursor",
            issues=nick_issues,
            pre_nick_match=None,
            release_match=None,
            projection=None,
            candidate=None,
        )
    if not allow_precut_footprint_outside_active_product:
        nick_boundary_issues = _validate_pre_nick_site_origin_boundary(nick_match)
        if nick_boundary_issues:
            return ReleasedEvaluationResult(
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
    release_match, release_issues = _select_release_match(
        matches=release_matches,
        target_active_product_length=target_active_product_length,
        coordinate_offset=precursor_coordinate_offset,
        active_strand=route_family_active_strand(route_family),
        intended_site_sequence=release_intended_site_sequence,
    )
    if release_match is None:
        return ReleasedEvaluationResult(
            status="no_release_path",
            issues=release_issues,
            pre_nick_match=nick_match,
            release_match=None,
            projection=None,
            candidate=None,
        )
    projection, projection_issues = _project_released_product(
        precursor_top_strand=precursor_top_strand,
        nick_match=nick_match,
        release_match=release_match,
        constraints=constraints,
        coordinate_offset=precursor_coordinate_offset,
        route_family=route_family,
    )
    if projection is None:
        return ReleasedEvaluationResult(
            status="no_release_path",
            issues=projection_issues,
            pre_nick_match=nick_match,
            release_match=release_match,
            projection=None,
            candidate=None,
        )
    candidate, candidate_issues = _evaluate_projected_candidate(
        projection=projection,
        original_nick_match=nick_match,
        nick_entry=nick_entry,
        target=target,
    )
    if candidate is None:
        status = "post_release_projection_failed"
        if any(
            issue.code in {"HOMOLOGY_MISMATCH_LIMIT_EXCEEDED", "TERMINAL_LIGATABLE_DUPLEX_BP_OUT_OF_RANGE"}
            for issue in candidate_issues
        ):
            status = "unsatisfied"
        if any(issue.code == "MAX_UNINTERRUPTED_DUPLEX_BP_EXCEEDED" for issue in candidate_issues):
            status = "unsatisfied"
        return ReleasedEvaluationResult(
            status=status,
            issues=candidate_issues,
            pre_nick_match=nick_match,
            release_match=release_match,
            projection=projection,
            candidate=None,
        )
    return ReleasedEvaluationResult(
        status="satisfied",
        issues=[],
        pre_nick_match=nick_match,
        release_match=release_match,
        projection=projection,
        candidate=candidate,
    )


__all__ = ["ReleasedEvaluationResult", "build_projected_snapback_candidate", "evaluate_released_precursor"]
