"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/ligation_scoring.py

Ligation-aware ranking helpers for YIU 4-bp junction mismatch selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dnadesign.cruncher.yiu.candidate_generation import CandidatePlan
from dnadesign.cruncher.yiu.domain_models import ChosenLigationKey, LigationMismatchRationale
from dnadesign.cruncher.yiu.scoring import apply_candidate_sequences

LigationProfile = Literal["none", "t4", "t7", "t3", "pbcv1", "hlig3"]

_POSITION_CLASS_BY_OFFSET = {0: "edge", 1: "middle", 2: "middle", 3: "edge"}
_STRICT_PURINE_MISMATCH_CLASSES = frozenset({"AG", "GG"})
_PERMISSIVE_PROFILES = frozenset({"t3", "pbcv1", "hlig3"})


@dataclass(frozen=True)
class CandidateLigationScore:
    chosen_key: ChosenLigationKey
    mismatch_rationales: tuple[LigationMismatchRationale, ...]
    key: tuple[int, int, int, bool, int]


@dataclass(frozen=True)
class CandidateLigationFilterResult:
    admissible: bool
    failure_fields: tuple[str, ...]


def canonical_mismatch_class(base_a: str, base_b: str) -> str:
    return "".join(sorted((base_a, base_b)))


def position_class_for_offset(offset: int) -> Literal["edge", "middle"]:
    return _POSITION_CLASS_BY_OFFSET[offset]


def mismatch_class_tier(*, mismatch_class: str, ligation_profile: LigationProfile) -> int:
    if ligation_profile == "none":
        return 0
    if mismatch_class == "GT":
        return 0
    if mismatch_class in _STRICT_PURINE_MISMATCH_CLASSES:
        return 1 if ligation_profile in _PERMISSIVE_PROFILES else 2
    return 3


def _bad_pattern_penalty(
    *,
    candidate: CandidatePlan,
    reference_payload_sequence: str,
    reference_complement_sequence: str,
) -> int:
    selected_payload, selected_complement = apply_candidate_sequences(
        candidate=candidate,
        reference_payload_sequence=reference_payload_sequence,
        reference_complement_sequence=reference_complement_sequence,
    )
    _ = selected_payload
    # YIU stores aligned complement as 3'->5'; reverse the selected junction slice to score the
    # displayed cohesive-end sequence in 5'->3'. TNNA is a conservative late tie-break only.
    overhang_5to3 = selected_complement[candidate.junction_start : candidate.junction_end][::-1]
    return int(len(overhang_5to3) == 4 and overhang_5to3[0] == "T" and overhang_5to3[3] == "A")


def build_candidate_ligation_score(
    *,
    candidate: CandidatePlan,
    ligation_profile: LigationProfile,
    bad_pattern_heuristics: bool,
    force_bad_pattern_penalty: bool = False,
    reference_payload_sequence: str,
    reference_complement_sequence: str,
) -> CandidateLigationScore:
    rationales: list[LigationMismatchRationale] = []
    for mutation in sorted(candidate.mutations, key=lambda item: item.junction_offset):
        mismatch_class = canonical_mismatch_class(mutation.mutated_base, mutation.opposing_base)
        rationales.append(
            LigationMismatchRationale(
                payload_index=mutation.payload_index,
                junction_offset=mutation.junction_offset,
                position_class=position_class_for_offset(mutation.junction_offset),
                mutated_strand=mutation.mutated_strand,  # ownership matters for PWM, not mismatch class
                native_base=mutation.native_base,
                partner_base=mutation.opposing_base,
                canonical_mismatch_class=mismatch_class,
                class_tier=mismatch_class_tier(mismatch_class=mismatch_class, ligation_profile=ligation_profile),
            )
        )
    tiers = [entry.class_tier for entry in rationales]
    chosen_key = ChosenLigationKey(
        worst_mismatch_class_tier=max(tiers, default=0),
        total_mismatch_class_tier=sum(tiers),
        middle_mismatch_count=candidate.middle_mismatch_count,
        # Bilotti et al. do not directly enumerate every 2-mismatch geometry YIU can generate; double-middle is
        # treated as an engineering extrapolation grounded in their edge-vs-middle trend.
        double_middle_flag=candidate.double_middle_flag,
        bad_pattern_penalty=(
            _bad_pattern_penalty(
                candidate=candidate,
                reference_payload_sequence=reference_payload_sequence,
                reference_complement_sequence=reference_complement_sequence,
            )
            if bad_pattern_heuristics or force_bad_pattern_penalty
            else 0
        ),
    )
    return CandidateLigationScore(
        chosen_key=chosen_key,
        mismatch_rationales=tuple(rationales),
        key=(
            chosen_key.worst_mismatch_class_tier,
            chosen_key.total_mismatch_class_tier,
            chosen_key.middle_mismatch_count,
            chosen_key.double_middle_flag,
            chosen_key.bad_pattern_penalty,
        ),
    )


def evaluate_hard_ligation_filter(
    *,
    ligation_score: CandidateLigationScore,
    max_worst_mismatch_class_tier: int,
    max_middle_mismatch_count: int,
    allow_double_middle: bool,
    allow_tnna_like_overhangs: bool,
) -> CandidateLigationFilterResult:
    failure_fields: list[str] = []
    if ligation_score.chosen_key.worst_mismatch_class_tier > max_worst_mismatch_class_tier:
        failure_fields.append("max_worst_mismatch_class_tier")
    if ligation_score.chosen_key.middle_mismatch_count > max_middle_mismatch_count:
        failure_fields.append("max_middle_mismatch_count")
    if not allow_double_middle and ligation_score.chosen_key.double_middle_flag:
        failure_fields.append("allow_double_middle")
    if not allow_tnna_like_overhangs and ligation_score.chosen_key.bad_pattern_penalty > 0:
        failure_fields.append("allow_tnna_like_overhangs")
    return CandidateLigationFilterResult(
        admissible=not failure_fields,
        failure_fields=tuple(failure_fields),
    )


__all__ = [
    "CandidateLigationScore",
    "CandidateLigationFilterResult",
    "LigationProfile",
    "build_candidate_ligation_score",
    "evaluate_hard_ligation_filter",
    "canonical_mismatch_class",
    "mismatch_class_tier",
    "position_class_for_offset",
]
