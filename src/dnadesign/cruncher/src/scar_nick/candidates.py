"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/scar_nick/candidates.py

Candidate construction and per-pair policy annotation for scar_nick.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

from dnadesign.cruncher.nickases.models import normalize_dna, reverse_complement
from dnadesign.cruncher.scar_nick.models import (
    CandidateRankingContext,
    NickasePlacement,
    ReleasePlacement,
    ScarNickCandidate,
)
from dnadesign.cruncher.scar_nick.policy import classify_profile_policy
from dnadesign.cruncher.scar_nick.profiles import classify_pair_profile
from dnadesign.cruncher.scar_nick.ranking import ranking_key
from dnadesign.cruncher.scar_nick.semantics import surviving_strand_for_nick
from dnadesign.cruncher.utils.hashing import sha256_bytes


def _candidate_id(
    left_base: str,
    right_base: str,
    *,
    release_placement: ReleasePlacement | None,
    nickase_placement: NickasePlacement | None,
) -> str:
    parts = [f"left={left_base}", f"right={right_base}"]
    if release_placement is None:
        parts.append("release=none")
    else:
        parts.extend(
            [
                f"release_variant={release_placement.variant_id}",
                f"release_orientation={release_placement.orientation}",
                f"release_site={release_placement.recognition_sequence}",
                f"release_span={release_placement.recognition_site_start}:{release_placement.recognition_site_end}",
                f"release_cut={release_placement.top_cut_boundary}:{release_placement.bottom_cut_boundary}",
            ]
        )
    if nickase_placement is None:
        parts.append("nickase=none")
    else:
        parts.extend(
            [
                f"nickase_variant={nickase_placement.variant_id}",
                f"nickase_orientation={nickase_placement.orientation}",
                f"nickase_motif={nickase_placement.motif_top_5to3}",
                f"nickase_span={nickase_placement.source_site_start}:{nickase_placement.source_site_end}",
                f"nickase_strand={nickase_placement.strand}",
                f"nickase_boundary={nickase_placement.boundary}:{nickase_placement.terminal_boundary}",
            ]
        )
    return sha256_bytes("|".join(parts).encode("utf-8"))[:12]


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


def _append_rejection(rejection_reasons: list[str], reason: str) -> None:
    if reason not in rejection_reasons:
        rejection_reasons.append(reason)


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
    surviving_strand = surviving_strand_for_nick(nicked_strand)
    candidate = ScarNickCandidate(
        candidate_id=_candidate_id(
            left,
            right,
            release_placement=release_placement,
            nickase_placement=nickase_placement,
        ),
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


__all__ = ["evaluate_pair_candidate"]
