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

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry, NickEvent
from dnadesign.cruncher.nickases.scanning import (
    EvaluatedMatch,
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
from dnadesign.cruncher.snapback.models import CoordinateSpan, SnapbackIssue
from dnadesign.cruncher.snapback.planner import evaluate_snapback_candidate
from dnadesign.cruncher.snapback.released_models import (
    ReleasedFinalCandidate,
    ReleasedFinalTargetGeometry,
    ReleasedProductProjection,
    ReleasedSnapbackConstraintsSpec,
)


def _issue(code: str, message: str, **details: object) -> SnapbackIssue:
    return SnapbackIssue(code=code, message=message, details=details)


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
    normalize_to_top_strand_nick: bool,
) -> tuple[EvaluatedMatch | None, list[SnapbackIssue]]:
    selected = [match for match in matches if match.nick.boundary == target.nick_boundary_from_left]
    if normalize_to_top_strand_nick:
        selected = [match for match in selected if match.nick.strand == "primary"]
    if intended_site_sequence is not None:
        selected = [match for match in selected if match.site.matched_span_sequence == intended_site_sequence]
    if not selected:
        return (
            None,
            [
                _issue(
                    "PRE_NICK_SITE_NOT_FOUND",
                    "No pre-nick site matched the requested boundary on the precursor.",
                    nick_boundary_from_left=target.nick_boundary_from_left,
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
    target_designed_length: int,
    intended_site_sequence: str | None,
) -> tuple[ReleaseEvaluatedMatch | None, list[SnapbackIssue]]:
    selected = [match for match in matches if match.cut.top_cut_boundary == target_designed_length]
    if intended_site_sequence is not None:
        selected = [match for match in selected if match.site.matched_span_sequence == intended_site_sequence]
    if not selected:
        return (
            None,
            [
                _issue(
                    "NO_RELEASE_MATCH_FOR_TARGET_LENGTH",
                    "No release site produced the retained-product top-strand length required by the final target.",
                    required_release_top_cut=target_designed_length,
                )
            ],
        )
    if len(selected) > 1:
        return (
            None,
            [
                _issue(
                    "RELEASE_SITE_AMBIGUOUS",
                    "Multiple release sites produced the same retained-product top-strand length.",
                    match_count=len(selected),
                    required_release_top_cut=target_designed_length,
                )
            ],
        )
    return selected[0], []


def _build_rebased_nick_event(original: NickEvent, *, rebased_boundary: int) -> NickEvent:
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


def _project_released_product(
    *,
    precursor_top_strand: str,
    nick_match: EvaluatedMatch,
    release_match: ReleaseEvaluatedMatch,
    constraints: ReleasedSnapbackConstraintsSpec,
) -> tuple[ReleasedProductProjection | None, list[SnapbackIssue]]:
    precursor_length = len(precursor_top_strand)
    issues: list[SnapbackIssue] = []
    if constraints.require_release_site_downstream_of_nick and release_match.site.start < nick_match.nick.boundary:
        issues.append(
            _issue(
                "RELEASE_SITE_NOT_DOWNSTREAM_OF_NICK",
                "The release recognition site must start downstream of the intended nick boundary.",
                release_site_start=release_match.site.start,
                nick_boundary=nick_match.nick.boundary,
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
    if (
        constraints.require_nick_survives_in_retained_product
        and nick_match.nick.boundary >= release_match.cut.top_cut_boundary
    ):
        issues.append(
            _issue(
                "POST_RELEASE_NICK_LOST",
                "The retained upstream top strand would not contain the intended nick after release.",
                nick_boundary=nick_match.nick.boundary,
                release_top_cut=release_match.cut.top_cut_boundary,
            )
        )
    if issues:
        return None, issues

    retained_top_strand = precursor_top_strand[: release_match.cut.top_cut_boundary]
    retained_nickase_matches = enumerate_nickase_site_instances(
        retained_top_strand,
        coordinate_offset=0,
        entry=nick_match.variant,
    )
    nickase_site_survives = any(
        match.nick.boundary == nick_match.nick.boundary
        and match.nick.strand == nick_match.nick.strand
        and match.site.orientation == nick_match.site.orientation
        for match in retained_nickase_matches
    )
    retained_release_matches = enumerate_release_site_instances(
        retained_top_strand,
        coordinate_offset=0,
        entry=release_match.variant,
    )
    release_site_survives = any(
        match.site.orientation == release_match.site.orientation for match in retained_release_matches
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
    return (
        ReleasedProductProjection(
            precursor_top_strand=precursor_top_strand,
            precursor_length=precursor_length,
            nick_coordinate_precursor=nick_match.nick.boundary,
            release_top_cut_precursor=release_match.cut.top_cut_boundary,
            release_bottom_cut_precursor=release_match.cut.bottom_cut_boundary,
            retained_top_strand=retained_top_strand,
            retained_bottom_strand_span=(0, release_match.cut.bottom_cut_boundary),
            retained_product_length=len(retained_top_strand),
            rebased_nick_coordinate=nick_match.nick.boundary,
            nickase_site_survives_post_release=nickase_site_survives,
            release_site_survives_post_release=release_site_survives,
        ),
        [],
    )


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


def _evaluate_projected_candidate(
    *,
    projection: ReleasedProductProjection,
    original_nick_match: EvaluatedMatch,
    nick_entry: NickaseCatalogEntry,
    target: ReleasedFinalTargetGeometry,
) -> tuple[ReleasedFinalCandidate | None, list[SnapbackIssue]]:
    retained_length = projection.retained_product_length
    input_length = retained_length - target.paired_bp
    expected_input_length = target.nick_boundary_from_left + target.paired_bp + target.cap_nt
    if input_length != expected_input_length:
        return (
            None,
            [
                _issue(
                    "RETAINED_PRODUCT_DOES_NOT_PROJECT_TO_FINAL_TARGET",
                    "The retained product length does not project to the requested final target geometry.",
                    retained_product_length=retained_length,
                    input_length=input_length,
                    required_input_length=expected_input_length,
                )
            ],
        )
    if input_length <= target.nick_boundary_from_left or retained_length <= input_length:
        return (
            None,
            [
                _issue(
                    "RETAINED_PRODUCT_PROJECTION_INVALID",
                    "The retained product could not be partitioned into input sequence and foldback arm.",
                    retained_product_length=retained_length,
                    input_length=input_length,
                    nick_boundary=target.nick_boundary_from_left,
                )
            ],
        )
    input_sequence = projection.retained_top_strand[:input_length]
    foldback_arm = projection.retained_top_strand[input_length:]
    retained_matches = enumerate_nickase_site_instances(
        projection.retained_top_strand,
        coordinate_offset=0,
        entry=nick_entry,
    )
    intended_retained_match = next(
        (
            match
            for match in retained_matches
            if match.nick.boundary == projection.rebased_nick_coordinate
            and match.nick.strand == original_nick_match.nick.strand
        ),
        None,
    )
    selected_match = _build_projected_selected_match(
        entry=nick_entry,
        input_sequence=input_sequence,
        nick_boundary=projection.rebased_nick_coordinate,
        intended_strand=original_nick_match.nick.strand,
    )
    all_matches: list[EvaluatedMatch] = [selected_match]
    replaced_intended = False
    for match in retained_matches:
        if (
            not replaced_intended
            and intended_retained_match is not None
            and match.key() == intended_retained_match.key()
        ):
            replaced_intended = True
            continue
        all_matches.append(match)

    candidate, issues = evaluate_snapback_candidate(
        input_sequence=input_sequence,
        protected_region=CoordinateSpan(start=0, end=0),
        pre_nick_duplex_window=CoordinateSpan(start=0, end=input_length),
        retained_homology_window=CoordinateSpan(
            start=projection.rebased_nick_coordinate,
            end=projection.rebased_nick_coordinate + target.paired_bp,
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
    if issues or candidate is None:
        return None, issues
    rebased_event = _build_rebased_nick_event(
        original_nick_match.nick,
        rebased_boundary=projection.rebased_nick_coordinate,
    )
    return (
        ReleasedFinalCandidate(
            designed_sequence=projection.retained_top_strand,
            input_sequence=input_sequence,
            foldback_arm=foldback_arm,
            nick_boundary_from_left=candidate.nick_boundary_from_left,
            paired_bp=candidate.paired_bp,
            cap_nt=candidate.cap_nt,
            source_cap_nt=len(candidate.source_cap_sequence),
            cap_extension_nt=candidate.cap_extension_nt,
            retained_product_length_nt=projection.retained_product_length,
            input_length_nt=len(input_sequence),
            mismatch_count=candidate.mismatch_count,
            mismatch_positions=list(candidate.mismatch_positions),
            terminal_ligatable_duplex_bp=candidate.terminal_ligatable_duplex_bp,
            max_uninterrupted_duplex_bp=candidate.max_uninterrupted_duplex_bp,
            extra_nick_event_count=len(candidate.extra_nick_events),
            extra_target_strand_nick_count=len(candidate.extra_target_strand_nicks),
            gc_fraction_added=candidate.gc_fraction_added,
            max_homopolymer_run_added=candidate.max_homopolymer_run_added,
            intended_nick=rebased_event,
            extra_target_strand_nicks=list(candidate.extra_target_strand_nicks),
            extra_nick_events=list(candidate.extra_nick_events),
            post_nick_sequence=candidate.post_nick_sequence,
            nickase_site_survives_post_release=projection.nickase_site_survives_post_release,
            release_site_survives_post_release=projection.release_site_survives_post_release,
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
    normalize_to_top_strand_nick: bool = True,
) -> ReleasedEvaluationResult:
    nick_matches = enumerate_nickase_site_instances(precursor_top_strand, coordinate_offset=0, entry=nick_entry)
    nick_match, nick_issues = _select_nick_match(
        matches=nick_matches,
        target=target,
        intended_site_sequence=nick_intended_site_sequence,
        normalize_to_top_strand_nick=normalize_to_top_strand_nick,
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
    target_designed_length = target.nick_boundary_from_left + (2 * target.paired_bp) + target.cap_nt
    release_matches = enumerate_release_site_instances(precursor_top_strand, coordinate_offset=0, entry=release_entry)
    release_match, release_issues = _select_release_match(
        matches=release_matches,
        target_designed_length=target_designed_length,
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


__all__ = ["ReleasedEvaluationResult", "evaluate_released_precursor"]
