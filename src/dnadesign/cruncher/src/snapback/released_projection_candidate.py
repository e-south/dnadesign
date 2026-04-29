"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_projection_candidate.py

Candidate reconstruction helpers for released-product projection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry, NickEvent
from dnadesign.cruncher.nickases.scanning import EvaluatedMatch
from dnadesign.cruncher.nickases.scanning import (
    build_evaluated_match as build_nick_evaluated_match,
)
from dnadesign.cruncher.nickases.scanning import (
    enumerate_site_instances as enumerate_nickase_site_instances,
)
from dnadesign.cruncher.snapback.models import CoordinateSpan, SnapbackCandidateDesign, SnapbackIssue
from dnadesign.cruncher.snapback.planner import evaluate_snapback_candidate
from dnadesign.cruncher.snapback.released_projection_common import (
    build_issue,
    build_projected_origin_event,
)
from dnadesign.cruncher.snapback.released_projection_models import (
    ReleasedFinalCandidate,
    ReleasedProductProjection,
)
from dnadesign.cruncher.snapback.released_spec_models import ReleasedFinalTargetGeometry


def precursor_extra_nick_events(
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


def build_projected_selected_match(
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
                build_issue(
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
                build_issue(
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
    selected_match = build_projected_selected_match(
        entry=nick_entry,
        input_sequence=input_sequence,
        nick_boundary=projection.rebased_nick_boundary,
        intended_strand=intended_strand,
    )
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
        all_matches=[selected_match],
        forbid_additional_target_strand_nicks=False,
        forbid_any_additional_nicks=False,
    )


def evaluate_projected_candidate(
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
    extra_nick_events, extra_target_strand_nicks = precursor_extra_nick_events(
        precursor_top_strand=projection.precursor_top_strand,
        nick_entry=nick_entry,
        original_nick_match=original_nick_match,
    )
    projected_origin_event = build_projected_origin_event(
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


def candidate_failure_status(issues: list[SnapbackIssue]) -> str:
    if any(
        issue.code in {"HOMOLOGY_MISMATCH_LIMIT_EXCEEDED", "TERMINAL_LIGATABLE_DUPLEX_BP_OUT_OF_RANGE"}
        for issue in issues
    ):
        return "unsatisfied"
    if any(issue.code == "MAX_UNINTERRUPTED_DUPLEX_BP_EXCEEDED" for issue in issues):
        return "unsatisfied"
    return "post_release_projection_failed"
