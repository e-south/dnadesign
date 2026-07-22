"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/scar_nick/planner.py

Deterministic planning and ranking for scar-nick candidates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path
from typing import Iterable

from dnadesign.cruncher.nickases.models import (
    NickaseCatalog,
    NickaseCatalogEntry,
    iupac_bases_for_symbol,
)
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeCatalog, ReleaseEnzymeEntry
from dnadesign.cruncher.release_enzymes.scanning import derive_release_cut
from dnadesign.cruncher.scar_nick.candidates import evaluate_pair_candidate
from dnadesign.cruncher.scar_nick.geometry import (
    build_nickase_geometry_audit,
    compatible_scar_sequences_from_audit,
    iupac_symbols_overlap,
    nickase_entry_rejection_reasons,
    placements_for_entry,
)
from dnadesign.cruncher.scar_nick.models import (
    CandidateRankingContext,
    JunctionSpec,
    NickaseGeometryAuditEntry,
    NickasePlacement,
    ReleasePlacement,
    ScarNickCandidate,
    ScarNickEvaluationReport,
    ScarNickReportMetadata,
    ScarNickSpecDocument,
    SearchSpec,
    ValidationIssue,
)
from dnadesign.cruncher.scar_nick.policy import classify_profile_policy
from dnadesign.cruncher.scar_nick.profiles import profile_label_s3s2s1s0
from dnadesign.cruncher.scar_nick.ranking import (
    rank_pair_candidates,
    ranking_key,
    select_profile_bucket_candidates,
    unique_sequence_candidates,
)

_BASES = ("A", "C", "G", "T")
_RESERVE_REASON_PRIORITY = {
    "MIDDLE_MIDDLE_DOUBLE_HARD": 0,
    "MORE_THAN_TWO_NON_WATSON_CRICK": 1,
    "TOO_MANY_HARD_MISMATCHES": 2,
    "INSUFFICIENT_LIGATION_SUPPORT": 3,
    "EXCESSIVE_EFFECTIVE_DISRUPTION": 4,
    "RESERVE_PROFILE_BUCKET": 5,
}


def _select_reserve_policy_examples(
    ranked_candidates: Iterable[ScarNickCandidate],
    *,
    limit: int,
) -> list[ScarNickCandidate]:
    reserve_pool = sorted(
        (
            candidate
            for candidate in ranked_candidates
            if candidate.profile_policy_status == "reserve"
            and candidate.nickase_placement is not None
            and "RETAINED_RELEASE_RECOGNITION_SITE" not in candidate.rejection_reasons
        ),
        key=lambda candidate: (
            _RESERVE_REASON_PRIORITY.get(candidate.profile_policy_reason, 99),
            candidate.profile_s3s2s1s0,
            tuple(candidate.rank_key),
        ),
    )
    selected: list[ScarNickCandidate] = []
    seen_profiles: set[str] = set()
    for candidate in reserve_pool:
        if candidate.profile_s3s2s1s0 in seen_profiles:
            continue
        seen_profiles.add(candidate.profile_s3s2s1s0)
        selected.append(candidate)
        if len(selected) >= limit:
            return selected
    return selected


def _release_placement(entry: ReleaseEnzymeEntry, *, required_terminal_scar_nt: int) -> ReleasePlacement:
    cut_at_origin = derive_release_cut(entry=entry, start=0, orientation="forward")
    recognition_site_start = -cut_at_origin.top_cut_boundary
    recognition_site_end = recognition_site_start + entry.recognition_len
    top_cut_boundary = 0
    bottom_cut_boundary = cut_at_origin.bottom_cut_boundary - cut_at_origin.top_cut_boundary
    retained_scar_start = top_cut_boundary
    retained_scar_end = bottom_cut_boundary
    recognition_site_excised = (
        recognition_site_end <= retained_scar_start or recognition_site_start >= retained_scar_end
    )
    return ReleasePlacement(
        variant_id=entry.variant_id,
        orientation="forward",
        recognition_sequence=entry.recognition_sequence,
        source_catalog_id=entry.source_catalog_id,
        source_url=entry.source_url or "",
        commercial_confidence=entry.commercial_confidence,
        warning_codes=[str(code).strip().upper() for code in entry.warning_codes],
        recognition_site_start=recognition_site_start,
        recognition_site_end=recognition_site_end,
        top_cut_boundary=top_cut_boundary,
        bottom_cut_boundary=bottom_cut_boundary,
        retained_scar_start=retained_scar_start,
        retained_scar_end=retained_scar_end,
        retained_scar_nt=required_terminal_scar_nt,
        recognition_site_excised=recognition_site_excised,
    )


def _validate_release_geometry(
    spec: ScarNickSpecDocument,
    release_catalog: ReleaseEnzymeCatalog,
) -> tuple[ReleasePlacement | None, list[ValidationIssue], list[str]]:
    release_by_id = release_catalog.by_id()
    variant_id = spec.processing.release.variant_id
    if variant_id not in release_by_id:
        return (
            None,
            [
                ValidationIssue(
                    code="RELEASE_VARIANT_NOT_FOUND",
                    message=f"Release enzyme variant {variant_id!r} was not found in the resolved catalog.",
                )
            ],
            [],
        )
    entry = release_by_id[variant_id]
    placement = _release_placement(
        entry,
        required_terminal_scar_nt=spec.processing.release.required_terminal_scar_nt,
    )
    issues: list[ValidationIssue] = []
    forbidden_release_sites = [entry.recognition_sequence]
    observed_scar_nt = placement.bottom_cut_boundary - placement.top_cut_boundary
    if observed_scar_nt != spec.processing.release.required_terminal_scar_nt:
        issues.append(
            ValidationIssue(
                code="NON_TERMINAL_RELEASE_SCAR",
                message="Release enzyme cut geometry does not produce the required terminal scar length.",
                details={
                    "variant_id": variant_id,
                    "observed_scar_nt": observed_scar_nt,
                    "required_terminal_scar_nt": spec.processing.release.required_terminal_scar_nt,
                },
            )
        )
    if spec.processing.release.recognition_site_must_be_excised and not placement.recognition_site_excised:
        issues.append(
            ValidationIssue(
                code="RELEASE_RECOGNITION_SITE_RETAINED_BY_GEOMETRY",
                message="Release recognition site overlaps the retained terminal scar.",
                details={"variant_id": variant_id},
            )
        )
    return placement, issues, forbidden_release_sites


def _motif_allows_base(motif: str, *, motif_offset: int, base: str) -> bool:
    if motif_offset < 0 or motif_offset >= len(motif):
        return True
    return base in iupac_bases_for_symbol(motif[motif_offset])


def _nickase_entry_rejection_reasons(
    entry: NickaseCatalogEntry,
    *,
    min_recognition_nt: int,
    disallowed_warning_codes: list[str],
) -> list[str]:
    return nickase_entry_rejection_reasons(
        entry,
        min_recognition_nt=min_recognition_nt,
        disallowed_warning_codes=disallowed_warning_codes,
    )


def _candidate_matches_sequence_of_interest(
    *,
    left_base: str,
    right_base: str,
    release_placement: ReleasePlacement,
    nickase_placement: NickasePlacement,
) -> bool:
    retained_sequence = left_base
    for retained_offset, base in enumerate(retained_sequence):
        motif_offset = retained_offset - nickase_placement.source_site_start
        if not _motif_allows_base(nickase_placement.motif_top_5to3, motif_offset=motif_offset, base=base):
            return False

    for release_offset, release_symbol in enumerate(release_placement.recognition_sequence):
        coordinate = release_placement.recognition_site_start + release_offset
        if 0 <= coordinate < len(retained_sequence) and not iupac_symbols_overlap(
            release_symbol,
            retained_sequence[coordinate],
        ):
            return False
        motif_offset = coordinate - nickase_placement.source_site_start
        if 0 <= motif_offset < len(nickase_placement.motif_top_5to3) and not iupac_symbols_overlap(
            nickase_placement.motif_top_5to3[motif_offset],
            release_symbol,
        ):
            return False
    return True


def _nickase_placements(
    nickase_catalog: NickaseCatalog,
    *,
    terminal_boundary: int,
    target_strand: str,
    min_recognition_nt: int,
    disallowed_warning_codes: list[str],
) -> list[NickasePlacement]:
    placements: list[NickasePlacement] = []
    for boundary in (terminal_boundary, terminal_boundary - 1, terminal_boundary + 1):
        for entry in nickase_catalog.entries:
            placements.extend(
                placements_for_entry(
                    entry,
                    terminal_boundary=terminal_boundary,
                    boundary=boundary,
                    target_strand=target_strand,
                    min_recognition_nt=min_recognition_nt,
                    disallowed_warning_codes=disallowed_warning_codes,
                )
            )
    return sorted(
        placements,
        key=lambda item: (
            item.boundary_distance,
            item.variant_id,
            item.strand,
            item.orientation,
            item.source_site_start,
            item.source_site_end,
        ),
    )


def _best_nickase_placement_for_candidate(
    left_base: str,
    right_base: str,
    placements: list[NickasePlacement],
    *,
    release_placement: ReleasePlacement,
    terminal_required: bool,
) -> NickasePlacement | None:
    for placement in placements:
        if terminal_required and not placement.exact_terminal:
            continue
        if _candidate_matches_sequence_of_interest(
            left_base=left_base,
            right_base=right_base,
            release_placement=release_placement,
            nickase_placement=placement,
        ):
            return placement
    return None


def _all_four_mers() -> tuple[str, ...]:
    return tuple("".join(bases) for bases in product(_BASES, repeat=4))


def _candidate_inputs(
    search: SearchSpec,
    junction: JunctionSpec,
    *,
    left_bases: Iterable[str] | None = None,
) -> list[tuple[str, str]]:
    if search.mode != "curated_panel":
        raise ValueError(f"Unsupported scar-nick search mode: {search.mode}")
    left_pool = tuple(sorted(set(left_bases))) if left_bases is not None else _all_four_mers()
    pairs = [(left, right) for left in left_pool for right in _all_four_mers()]
    seed = (junction.left_base, junction.right_base)
    if seed in pairs:
        pairs.remove(seed)
    return [seed, *pairs]


def _candidate_profile_needed_for_report(
    profile: str,
    *,
    context: CandidateRankingContext,
    s0_match_required: bool,
) -> bool:
    decision = classify_profile_policy(profile, context=context, s0_match_required=s0_match_required)
    if decision.status == "reserve":
        return True
    if context.target_profile_buckets:
        return profile in context.target_profile_buckets
    return decision.status == "active"


def build_scar_nick_report(
    spec: ScarNickSpecDocument,
    *,
    spec_path: Path,
    workspace_root: Path,
    release_catalog: ReleaseEnzymeCatalog,
    nickase_catalog: NickaseCatalog,
) -> ScarNickEvaluationReport:
    release_placement, issues, forbidden_release_sites = _validate_release_geometry(spec, release_catalog)
    terminal_boundary = spec.junction.overhang_length
    nickase_placements: list[NickasePlacement] = []
    nickase_geometry_audit: list[NickaseGeometryAuditEntry] = []
    compatible_scar_sequences: tuple[str, ...] = ()
    metadata_warnings: list[str] = []
    if not issues and release_placement is not None:
        nickase_geometry_audit = build_nickase_geometry_audit(
            nickase_catalog,
            release_placement=release_placement,
            terminal_boundary=terminal_boundary,
            target_strand=spec.processing.nick.target_strand,
            min_recognition_nt=spec.search.min_nickase_recognition_nt,
            disallowed_warning_codes=spec.search.disallowed_nickase_warning_codes,
        )
        compatible_scar_sequences = compatible_scar_sequences_from_audit(nickase_geometry_audit)
        entry_rejection_counts = Counter(
            reason
            for entry in nickase_catalog.entries
            for reason in _nickase_entry_rejection_reasons(
                entry,
                min_recognition_nt=spec.search.min_nickase_recognition_nt,
                disallowed_warning_codes=spec.search.disallowed_nickase_warning_codes,
            )
        )
        if entry_rejection_counts:
            metadata_warnings.append("NICKASE_CATALOG_ENTRIES_FILTERED_BY_SCAR_NICK_POLICY")
        eligible_entry_count = len(nickase_catalog.entries) - sum(
            1
            for entry in nickase_catalog.entries
            if _nickase_entry_rejection_reasons(
                entry,
                min_recognition_nt=spec.search.min_nickase_recognition_nt,
                disallowed_warning_codes=spec.search.disallowed_nickase_warning_codes,
            )
        )
        if eligible_entry_count == 0:
            issues.append(
                ValidationIssue(
                    code="NO_ELIGIBLE_NICKASE_CATALOG_ENTRIES",
                    message="No nickase catalog entries passed the scar-nick catalog policy.",
                    details={
                        "min_nickase_recognition_nt": spec.search.min_nickase_recognition_nt,
                        "disallowed_nickase_warning_codes": spec.search.disallowed_nickase_warning_codes,
                        "entry_rejection_counts": dict(sorted(entry_rejection_counts.items())),
                    },
                )
            )
        if not issues:
            nickase_placements = _nickase_placements(
                nickase_catalog,
                terminal_boundary=terminal_boundary,
                target_strand=spec.processing.nick.target_strand,
                min_recognition_nt=spec.search.min_nickase_recognition_nt,
                disallowed_warning_codes=spec.search.disallowed_nickase_warning_codes,
            )
            exact_nickase_placements = [placement for placement in nickase_placements if placement.exact_terminal]
            if spec.processing.nick.terminal_nick_required and not exact_nickase_placements:
                issues.append(
                    ValidationIssue(
                        code="NO_EXACT_TERMINAL_NICK",
                        message="No nickase placement can nick an allowed strand exactly at the terminal boundary.",
                        details={"terminal_boundary": terminal_boundary},
                    )
                )

    accepted: list[ScarNickCandidate] = []
    reserve_candidates: list[ScarNickCandidate] = []
    rejected_controls: list[ScarNickCandidate] = []
    enumerated_count = 0
    if not issues and release_placement is not None:
        evaluated: list[ScarNickCandidate] = []
        candidate_inputs = _candidate_inputs(
            spec.search,
            spec.junction,
            left_bases=compatible_scar_sequences,
        )
        enumerated_count = len(candidate_inputs)
        profile_needed: dict[str, bool] = {}
        for left_base, right_base in candidate_inputs:
            seed_candidate = (left_base, right_base) == (spec.junction.left_base, spec.junction.right_base)
            profile = profile_label_s3s2s1s0(
                left_base,
                right_base,
                allow_gt_wobble=spec.ranking_context.allow_gt_wobble,
            )
            needed = profile_needed.get(profile)
            if needed is None:
                needed = _candidate_profile_needed_for_report(
                    profile,
                    context=spec.ranking_context,
                    s0_match_required=spec.junction.s0_match_required,
                )
                profile_needed[profile] = needed
            if not needed and not seed_candidate:
                continue
            nickase_placement = _best_nickase_placement_for_candidate(
                left_base,
                right_base,
                nickase_placements,
                release_placement=release_placement,
                terminal_required=spec.processing.nick.terminal_nick_required,
            )
            candidate = evaluate_pair_candidate(
                left_base=left_base,
                right_base=right_base,
                context=spec.ranking_context,
                s0_match_required=spec.junction.s0_match_required,
                forbidden_release_sites=forbidden_release_sites,
                release_placement=release_placement,
                nickase_placement=nickase_placement,
            )
            if nickase_placement is None:
                candidate = candidate.model_copy(
                    update={
                        "rejection_reasons": [
                            *candidate.rejection_reasons,
                            "NO_COMPATIBLE_TERMINAL_NICK",
                        ]
                    }
                )
            evaluated.append(candidate)
            if seed_candidate:
                rejected_controls.append(candidate)

        ranked = rank_pair_candidates(evaluated, context=spec.ranking_context)
        accepted = select_profile_bucket_candidates(
            (candidate for candidate in ranked if not candidate.rejection_reasons),
            context=spec.ranking_context,
            limit=spec.search.max_hits,
        )
        accepted = [
            candidate.model_copy(update={"rank": rank, "rank_key": list(ranking_key(candidate, spec.ranking_context))})
            for rank, candidate in enumerate(accepted, start=1)
        ]
        reserve_candidates = _select_reserve_policy_examples(ranked, limit=spec.search.max_hits)
        rejected_controls = [candidate for candidate in rejected_controls if candidate.rejection_reasons]

        covered_buckets = {candidate.profile_s3s2s1s0 for candidate in accepted}
        missing_buckets = [
            bucket for bucket in spec.ranking_context.target_profile_buckets if bucket not in covered_buckets
        ]
        if missing_buckets:
            issues.append(
                ValidationIssue(
                    code="TARGET_PROFILE_BUCKETS_UNCOVERED",
                    message="No accepted scar-nick candidate covered one or more target profile buckets.",
                    details={"missing_profile_buckets": missing_buckets},
                )
            )

    if not accepted and not issues:
        issues.append(
            ValidationIssue(
                code="NO_ACCEPTED_SCAR_NICK_CANDIDATES",
                message="No pair candidates satisfied scar-nick profile, release, and nick constraints.",
            )
        )

    status = "satisfied" if accepted and not issues else "unsatisfied"
    return ScarNickEvaluationReport(
        status=status,
        spec_name=spec.scar_nick.name,
        workspace_root=str(workspace_root),
        spec_path=str(spec_path),
        metadata=ScarNickReportMetadata(
            spec_schema_version=spec.scar_nick.schema_version,
            contract=spec.scar_nick.contract,
            terminal_boundary=terminal_boundary,
            release_variant_id=spec.processing.release.variant_id,
            nick_target_strand=spec.processing.nick.target_strand,
            release_catalog_preset_ids=list(release_catalog.preset_ids),
            nickase_catalog_preset_ids=list(nickase_catalog.preset_ids),
            enumerated_candidate_count=enumerated_count,
            accepted_candidate_count=len(accepted),
            materialized_candidate_count=len(unique_sequence_candidates(accepted, limit=spec.search.materialize_top_k)),
            compatible_nickase_placement_count=sum(1 for entry in nickase_geometry_audit if entry.compatible),
            enzyme_compatible_scar_count=len(compatible_scar_sequences),
            warnings=metadata_warnings,
        ),
        release_placement=release_placement,
        issues=issues,
        nickase_geometry_audit=nickase_geometry_audit,
        candidates=accepted,
        reserve_candidates=reserve_candidates,
        rejected_reference_candidates=rejected_controls,
    )


__all__ = [
    "build_scar_nick_report",
]
