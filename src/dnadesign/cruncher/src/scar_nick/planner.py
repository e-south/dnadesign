"""
--------------------------------------------------------------------------------
<cruncher project>
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
    normalize_dna,
    reverse_complement,
)
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeCatalog, ReleaseEnzymeEntry
from dnadesign.cruncher.release_enzymes.scanning import derive_release_cut
from dnadesign.cruncher.scar_nick.geometry import (
    build_nickase_geometry_audit,
    compatible_scar_sequences_from_audit,
)
from dnadesign.cruncher.scar_nick.geometry import (
    entry_commercial_confidence as _geometry_entry_commercial_confidence,
)
from dnadesign.cruncher.scar_nick.geometry import (
    entry_warning_codes as _geometry_entry_warning_codes,
)
from dnadesign.cruncher.scar_nick.geometry import (
    iupac_symbol_is_fully_degenerate as _geometry_iupac_symbol_is_fully_degenerate,
)
from dnadesign.cruncher.scar_nick.geometry import (
    iupac_symbols_overlap as _geometry_iupac_symbols_overlap,
)
from dnadesign.cruncher.scar_nick.geometry import (
    nickase_entry_rejection_reasons as _geometry_nickase_entry_rejection_reasons,
)
from dnadesign.cruncher.scar_nick.geometry import (
    nickase_recognition_nt as _geometry_nickase_recognition_nt,
)
from dnadesign.cruncher.scar_nick.geometry import (
    placement_respects_terminal_downstream_rule as _geometry_placement_respects_terminal_downstream_rule,
)
from dnadesign.cruncher.scar_nick.geometry import (
    placements_for_entry as _geometry_placements_for_entry,
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
from dnadesign.cruncher.scar_nick.profiles import classify_pair_profile
from dnadesign.cruncher.scar_nick.ranking import (
    rank_pair_candidates,
    ranking_key,
    select_profile_bucket_candidates,
    unique_sequence_candidates,
)
from dnadesign.cruncher.utils.hashing import sha256_bytes

_BASES = ("A", "C", "G", "T")
_ALL_BASES = frozenset(_BASES)
_RESERVE_REASON_PRIORITY = {
    "MIDDLE_MIDDLE_DOUBLE_HARD": 0,
    "MORE_THAN_TWO_NON_WATSON_CRICK": 1,
    "TOO_MANY_HARD_MISMATCHES": 2,
    "INSUFFICIENT_LIGATION_SUPPORT": 3,
    "EXCESSIVE_EFFECTIVE_DISRUPTION": 4,
    "RESERVE_PROFILE_BUCKET": 5,
}


def _candidate_id(left_base: str, right_base: str) -> str:
    return sha256_bytes(f"{left_base}/{right_base}".encode("utf-8"))[:12]


def _gc_fraction(*sequences: str) -> float:
    joined = "".join(sequences)
    if not joined:
        return 0.0
    return sum(1 for base in joined if base in {"G", "C"}) / len(joined)


def _reference_distances(
    left_base: str,
    right_base: str,
    context: CandidateRankingContext,
) -> tuple[int | None, dict[str, int]]:
    distances: dict[str, int] = {}
    for label, reference in sorted(context.optional_reference_profiles.items()):
        observed_bases = left_base + right_base
        expected_bases = reference.left_base + reference.right_base
        distance = sum(
            1 for observed, expected in zip(observed_bases, expected_bases, strict=True) if observed != expected
        )
        distances[label] = distance
    if not distances:
        return None, {}
    if "working_control" in distances:
        return distances["working_control"], distances
    return min(distances.values()), distances


def _contains_forbidden_release_site(retained_sequence: str, forbidden_release_sites: Iterable[str]) -> bool:
    retained = normalize_dna(retained_sequence)
    for raw_site in forbidden_release_sites:
        site = normalize_dna(raw_site)
        if site in retained or reverse_complement(site) in retained:
            return True
    return False


def _pair_identities(profile) -> dict[str, str]:
    return {pair.site: f"{pair.left_base}:{pair.right_base}" for pair in profile.pairs}


def _tnna_flag(sequence: str) -> bool:
    scar = normalize_dna(sequence)
    return len(scar) == 4 and scar[0] == "T" and scar[3] == "A"


def _surviving_strand(nicked_strand: str | None) -> str | None:
    if nicked_strand == "top":
        return "bottom"
    if nicked_strand == "bottom":
        return "top"
    return None


def _append_rejection(rejection_reasons: list[str], reason: str) -> None:
    if reason not in rejection_reasons:
        rejection_reasons.append(reason)


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


def evaluate_pair_candidate(
    *,
    left_base: str,
    right_base: str,
    context: CandidateRankingContext,
    s0_match_required: bool,
    forbidden_release_sites: list[str],
    release_placement: ReleasePlacement | None = None,
    nickase_placement: NickasePlacement | None = None,
    nick_distance: int = 0,
) -> ScarNickCandidate:
    left = normalize_dna(left_base)
    right = normalize_dna(right_base)
    profile = classify_pair_profile(left, right, allow_gt_wobble=context.allow_gt_wobble)
    policy_decision = classify_profile_policy(
        profile.profile_s3s2s1s0,
        context=context,
        s0_match_required=s0_match_required,
    )
    pair_identities = _pair_identities(profile)
    retained_sequence = left
    rejection_reasons: list[str] = []
    if policy_decision.status == "reject":
        _append_rejection(rejection_reasons, policy_decision.reason)
    elif policy_decision.status == "reserve":
        _append_rejection(rejection_reasons, f"PROFILE_POLICY_RESERVE:{policy_decision.reason}")
    if profile.profile_s3s2s1s0 in context.reject_profiles:
        _append_rejection(rejection_reasons, "REJECTED_PROFILE_BUCKET")
    if context.target_profile_buckets and profile.profile_s3s2s1s0 not in context.target_profile_buckets:
        _append_rejection(rejection_reasons, "PROFILE_BUCKET_NOT_TARGETED")
    if profile.ligation_support < context.min_ligation_support:
        _append_rejection(rejection_reasons, "INSUFFICIENT_LIGATION_SUPPORT")
    if profile.effective_disruption > context.max_effective_disruption:
        _append_rejection(rejection_reasons, "EXCESSIVE_EFFECTIVE_DISRUPTION")
    if _contains_forbidden_release_site(retained_sequence, forbidden_release_sites):
        _append_rejection(rejection_reasons, "RETAINED_RELEASE_RECOGNITION_SITE")

    reference_distance, reference_distances = _reference_distances(left, right, context)
    terminal_boundary = release_placement.retained_scar_end if release_placement is not None else 4
    nick_boundary = (
        nickase_placement.boundary if nickase_placement is not None else terminal_boundary + int(nick_distance)
    )
    nicked_strand = None if nickase_placement is None else nickase_placement.strand
    surviving_strand = _surviving_strand(nicked_strand)
    candidate = ScarNickCandidate(
        candidate_id=_candidate_id(left, right),
        left_base=left,
        right_base=right,
        retained_scar=left,
        retained_product_sequence=retained_sequence,
        profile_s3s2s1s0=profile.profile_s3s2s1s0,
        profile_payload_outward=profile.profile_payload_outward,
        profile_policy_status=policy_decision.status,
        profile_policy_reason=policy_decision.reason,
        s0_match_required=s0_match_required,
        pair_classes=profile.pairs,
        s3_pair_identity=pair_identities["S3"],
        s2_pair_identity=pair_identities["S2"],
        s1_pair_identity=pair_identities["S1"],
        s0_pair_identity=pair_identities["S0"],
        m_count=profile.watson_crick_count,
        w_count=profile.wobble_count,
        x_count=profile.hard_mismatch_count,
        non_watson_crick_count=profile.non_watson_crick_count,
        middle_hard_count=profile.middle_hard_count,
        middle_wobble_count=profile.middle_wobble_count,
        worst_hard_mismatch_tier=profile.worst_hard_mismatch_tier,
        hard_mismatch_tier_sum=profile.hard_mismatch_tier_sum,
        middle_hard_mismatch_tier_sum=profile.middle_hard_mismatch_tier_sum,
        edge_hard_mismatch_tier_sum=profile.edge_hard_mismatch_tier_sum,
        ligation_support=profile.ligation_support,
        effective_disruption=profile.effective_disruption,
        tnna_flag=_tnna_flag(left),
        nicked_strand=nicked_strand,
        surviving_strand=surviving_strand,
        retained_scar_source="top_display_retained_scar_domain",
        discarded_strand_enzyme_burden=nicked_strand,
        release_placement=release_placement,
        retained_scar_nt=len(left),
        nickase_placement=nickase_placement,
        nickase_site=(
            None
            if nickase_placement is None
            else (
                f"{nickase_placement.variant_id}:{nickase_placement.orientation}"
                f"[{nickase_placement.source_site_start},{nickase_placement.source_site_end})"
            )
        ),
        nick_boundary=nick_boundary,
        terminal_boundary=terminal_boundary,
        nick_distance=abs(nick_boundary - terminal_boundary),
        gc_fraction=_gc_fraction(left, right),
        reference_control_distance=reference_distance,
        reference_distances=reference_distances,
        rejection_reasons=rejection_reasons,
    )
    return candidate.model_copy(update={"rank_key": list(ranking_key(candidate, context))})


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


def _iupac_symbol_is_fully_degenerate(symbol: str) -> bool:
    return _geometry_iupac_symbol_is_fully_degenerate(symbol)


def _nickase_recognition_nt(entry: NickaseCatalogEntry) -> int:
    return _geometry_nickase_recognition_nt(entry)


def _entry_warning_codes(entry: NickaseCatalogEntry) -> list[str]:
    return _geometry_entry_warning_codes(entry)


def _entry_commercial_confidence(entry: NickaseCatalogEntry) -> str | None:
    return _geometry_entry_commercial_confidence(entry)


def _nickase_entry_rejection_reasons(
    entry: NickaseCatalogEntry,
    *,
    min_recognition_nt: int,
    disallowed_warning_codes: list[str],
) -> list[str]:
    return _geometry_nickase_entry_rejection_reasons(
        entry,
        min_recognition_nt=min_recognition_nt,
        disallowed_warning_codes=disallowed_warning_codes,
    )


def _iupac_symbols_overlap(left_symbol: str, right_symbol: str) -> bool:
    return _geometry_iupac_symbols_overlap(left_symbol, right_symbol)


def _placement_respects_terminal_downstream_rule(placement: NickasePlacement) -> bool:
    return _geometry_placement_respects_terminal_downstream_rule(placement)


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
        if 0 <= coordinate < len(retained_sequence) and not _iupac_symbols_overlap(
            release_symbol,
            retained_sequence[coordinate],
        ):
            return False
        motif_offset = coordinate - nickase_placement.source_site_start
        if 0 <= motif_offset < len(nickase_placement.motif_top_5to3) and not _iupac_symbols_overlap(
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
                _placements_for_entry(
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


def _placements_for_entry(
    entry: NickaseCatalogEntry,
    *,
    terminal_boundary: int,
    boundary: int,
    target_strand: str = "bottom",
    min_recognition_nt: int = 4,
    disallowed_warning_codes: list[str] | None = None,
) -> list[NickasePlacement]:
    return _geometry_placements_for_entry(
        entry,
        terminal_boundary=terminal_boundary,
        boundary=boundary,
        target_strand=target_strand,
        min_recognition_nt=min_recognition_nt,
        disallowed_warning_codes=disallowed_warning_codes or [],
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
        for left_base, right_base in _candidate_inputs(
            spec.search,
            spec.junction,
            left_bases=compatible_scar_sequences,
        ):
            enumerated_count += 1
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
            if (left_base, right_base) == (spec.junction.left_base, spec.junction.right_base):
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


def render_markdown_report(report: ScarNickEvaluationReport) -> str:
    run_dir = report.run_dir
    if run_dir:
        try:
            run_dir = str(Path(run_dir).resolve().relative_to(Path(report.workspace_root).resolve()))
        except ValueError:
            run_dir = Path(run_dir).name
    lines = [
        f"# Scar-Nick Report: {report.spec_name}",
        "",
        f"- status: {report.status}",
        f"- workflow: {report.workflow}",
        f"- terminal_boundary: {report.metadata.terminal_boundary}",
        f"- release_variant: {report.metadata.release_variant_id}",
        f"- accepted_candidates: {len(report.candidates)}",
        f"- compatible_nickase_placements: {report.metadata.compatible_nickase_placement_count}",
        f"- enzyme_compatible_scars: {report.metadata.enzyme_compatible_scar_count}",
    ]
    if run_dir:
        lines.append(f"- run_dir: {run_dir}")
    lines.extend(
        [
            "",
            "## Handoff Tables",
            "",
            "- candidate_table: `export/table__scar_nick_candidates.csv`",
            "- candidate_pair_call_table: `export/table__scar_nick_candidate_pair_calls.csv`",
            "- nickase_geometry_audit_table: `export/table__scar_nick_nickase_geometry_audit.csv`",
        ]
    )
    if report.candidates:
        lines.extend(["", "## Candidates"])
        for candidate in report.candidates:
            lines.append(
                f"- rank {candidate.rank}: `{candidate.left_base}/{candidate.right_base}` "
                f"profile={candidate.profile_s3s2s1s0} "
                f"policy={candidate.profile_policy_status}:{candidate.profile_policy_reason} "
                f"non_wc={candidate.non_watson_crick_count} "
                f"middle_hard={candidate.middle_hard_count} "
                f"hard_tier={candidate.hard_mismatch_tier_sum} "
                f"middle_hard_tier={candidate.middle_hard_mismatch_tier_sum} "
                f"nick={candidate.nickase_site}"
            )
    if report.reserve_candidates:
        lines.extend(["", "## Reserve Profile Examples"])
        for candidate in report.reserve_candidates:
            lines.append(
                f"- `{candidate.left_base}/{candidate.right_base}` "
                f"profile={candidate.profile_s3s2s1s0} "
                f"policy={candidate.profile_policy_status}:{candidate.profile_policy_reason} "
                f"non_wc={candidate.non_watson_crick_count} "
                f"nick={candidate.nickase_site}"
            )
    if report.issues:
        lines.extend(["", "## Issues"])
        for issue in report.issues:
            lines.append(f"- {issue.code}: {issue.message}")
    return "\n".join(lines) + "\n"


__all__ = [
    "build_scar_nick_report",
    "evaluate_pair_candidate",
    "render_markdown_report",
]
