"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/scar_nick/profiles.py

Pair-profile helpers for 4-bp terminal scar-nick junctions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.nickases.models import normalize_dna, reverse_complement
from dnadesign.cruncher.scar_nick.models import PairClass, PairProfile
from dnadesign.cruncher.scar_nick.semantics import PROFILE_ORDER_S3S2S1S0, S_SITE_ORDER
from dnadesign.cruncher.yiu.ligation_scoring import (
    canonical_mismatch_class,
    mismatch_class_tier,
    position_class_for_offset,
)

_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
_GT_WOBBLE_PAIRS = {("G", "T"), ("T", "G")}


def _normalize_pair_inputs(left_base: str, right_base: str) -> tuple[str, str]:
    left = normalize_dna(left_base)
    right = normalize_dna(right_base)
    if len(left) != 4 or len(right) != 4:
        raise ValueError("scar-nick pair profiles require 4 nt left and right bases.")
    return left, right


def _pair_class(
    left_base: str,
    right_base: str,
    aligned_right_base: str,
    *,
    allow_gt_wobble: bool,
) -> tuple[str, str | None, int]:
    if left_base == aligned_right_base:
        return "M", None, 0
    mismatch_class = canonical_mismatch_class(left_base, right_base)
    if allow_gt_wobble and mismatch_class == "GT":
        return "W", mismatch_class, mismatch_class_tier(mismatch_class=mismatch_class, ligation_profile="t4")
    return "X", mismatch_class, mismatch_class_tier(mismatch_class=mismatch_class, ligation_profile="t4")


def profile_label_s3s2s1s0(
    left_base: str,
    right_base: str,
    *,
    profile_order: str = PROFILE_ORDER_S3S2S1S0,
    allow_gt_wobble: bool = True,
) -> str:
    """Return only the S3/S2/S1/S0 profile label without building full pair models."""

    if profile_order != PROFILE_ORDER_S3S2S1S0:
        raise ValueError(f"scar-nick profile_order must be {PROFILE_ORDER_S3S2S1S0}.")
    left, right = _normalize_pair_inputs(left_base, right_base)
    labels: list[str] = []
    for source_offset in range(4):
        left_symbol = left[source_offset]
        right_symbol = right[3 - source_offset]
        if left_symbol == _COMPLEMENT[right_symbol]:
            labels.append("M")
        elif allow_gt_wobble and (left_symbol, right_symbol) in _GT_WOBBLE_PAIRS:
            labels.append("W")
        else:
            labels.append("X")
    return "".join(labels)


def classify_pair_profile(
    left_base: str,
    right_base: str,
    *,
    profile_order: str = PROFILE_ORDER_S3S2S1S0,
    allow_gt_wobble: bool = True,
) -> PairProfile:
    """Classify a 4-bp paired junction profile in S3/S2/S1/S0 order.

    `right_base` is read antiparallel in S3/S2/S1/S0 order. Watson-Crick matches
    use the complement of that physical right-hand base. Wobble calls use the
    physical left:right pair and are therefore limited to G:T or T:G.
    """

    if profile_order != PROFILE_ORDER_S3S2S1S0:
        raise ValueError(f"scar-nick profile_order must be {PROFILE_ORDER_S3S2S1S0}.")
    left, right = _normalize_pair_inputs(left_base, right_base)
    aligned_right = reverse_complement(right)

    pairs: list[PairClass] = []
    for position, source_offset in enumerate(range(4)):
        left_symbol = left[source_offset]
        right_symbol = right[3 - source_offset]
        aligned_symbol = aligned_right[source_offset]
        class_label, mismatch_class, tier = _pair_class(
            left_symbol,
            right_symbol,
            aligned_symbol,
            allow_gt_wobble=allow_gt_wobble,
        )
        pairs.append(
            PairClass(
                position=position,
                site=S_SITE_ORDER[position],
                source_offset=source_offset,
                left_base=left_symbol,
                right_base=right_symbol,
                aligned_right_base=aligned_symbol,
                class_label=class_label,
                position_class=position_class_for_offset(position),
                canonical_mismatch_class=mismatch_class,
                class_tier_t4=tier,
            )
        )
    profile_s3s2s1s0 = "".join(pair.class_label for pair in pairs)
    hard_mismatch_count = sum(1 for pair in pairs if pair.class_label == "X")
    wobble_count = sum(1 for pair in pairs if pair.class_label == "W")
    non_watson_crick_count = hard_mismatch_count + wobble_count
    watson_crick_count = sum(1 for pair in pairs if pair.class_label == "M")
    middle_hard_count = sum(1 for pair in pairs if pair.site in {"S2", "S1"} and pair.class_label == "X")
    middle_wobble_count = sum(1 for pair in pairs if pair.site in {"S2", "S1"} and pair.class_label == "W")
    hard_pairs = [pair for pair in pairs if pair.class_label == "X"]
    middle_hard_mismatch_tier_sum = sum(pair.class_tier_t4 for pair in hard_pairs if pair.site in {"S2", "S1"})
    edge_hard_mismatch_tier_sum = sum(pair.class_tier_t4 for pair in hard_pairs if pair.site in {"S3", "S0"})
    hard_mismatch_tier_sum = middle_hard_mismatch_tier_sum + edge_hard_mismatch_tier_sum
    return PairProfile(
        profile_s3s2s1s0=profile_s3s2s1s0,
        profile_payload_outward=profile_s3s2s1s0[::-1],
        pairs=pairs,
        hard_mismatch_count=hard_mismatch_count,
        wobble_count=wobble_count,
        non_watson_crick_count=non_watson_crick_count,
        watson_crick_count=watson_crick_count,
        middle_hard_count=middle_hard_count,
        middle_wobble_count=middle_wobble_count,
        worst_hard_mismatch_tier=max((pair.class_tier_t4 for pair in hard_pairs), default=0),
        hard_mismatch_tier_sum=hard_mismatch_tier_sum,
        middle_hard_mismatch_tier_sum=middle_hard_mismatch_tier_sum,
        edge_hard_mismatch_tier_sum=edge_hard_mismatch_tier_sum,
        ligation_support=float(watson_crick_count) + 0.5 * float(wobble_count),
        effective_disruption=float(hard_mismatch_count) + 0.5 * float(wobble_count),
        s0_class=pairs[-1].class_label,
    )


__all__ = ["classify_pair_profile", "profile_label_s3s2s1s0"]
