"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/scar_nick/policy.py

Ligation-aware profile policy for terminal scar-nick panel selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

ProfilePolicyStatus = Literal["active", "reserve", "reject"]


class ProfilePolicyContext(Protocol):
    active_max_hard_mismatches: int
    active_max_non_watson_crick_pairs: int
    forbid_active_middle_middle_double_hard: bool
    min_ligation_support: float
    max_effective_disruption: float
    reject_profiles: list[str]
    reserve_profiles: list[str]


@dataclass(frozen=True)
class ProfilePolicyDecision:
    status: ProfilePolicyStatus
    reason: str


def normalize_profile(value: str) -> str:
    text = str(value or "").strip().upper()
    if len(text) != 4 or any(char not in {"M", "W", "X"} for char in text):
        raise ValueError("pair profiles must contain exactly four M/W/X symbols.")
    return text


def profile_non_watson_crick_count(profile: str) -> int:
    return sum(1 for symbol in profile if symbol in {"W", "X"})


def profile_ligation_support(profile: str) -> float:
    return float(profile.count("M")) + 0.5 * float(profile.count("W"))


def profile_effective_disruption(profile: str) -> float:
    return float(profile.count("X")) + 0.5 * float(profile.count("W"))


def classify_profile_policy(
    profile: str,
    *,
    context: ProfilePolicyContext,
    s0_match_required: bool = True,
) -> ProfilePolicyDecision:
    """Classify one S3_S2_S1_S0 profile for active scar-nick panel selection."""

    normalized = normalize_profile(profile)
    s3, s2, s1, s0 = normalized
    if s0_match_required and s0 != "M":
        return ProfilePolicyDecision(status="reject", reason="S0_PAIR_NOT_WATSON_CRICK")
    if normalized in context.reject_profiles:
        return ProfilePolicyDecision(status="reject", reason="REJECTED_PROFILE_BUCKET")
    if normalized in context.reserve_profiles:
        return ProfilePolicyDecision(status="reserve", reason="RESERVE_PROFILE_BUCKET")

    non_wc_count = profile_non_watson_crick_count(normalized)
    hard_count = normalized.count("X")
    if non_wc_count > context.active_max_non_watson_crick_pairs:
        return ProfilePolicyDecision(status="reserve", reason="MORE_THAN_TWO_NON_WATSON_CRICK")
    if hard_count > context.active_max_hard_mismatches:
        return ProfilePolicyDecision(status="reserve", reason="TOO_MANY_HARD_MISMATCHES")
    if context.forbid_active_middle_middle_double_hard and s2 == "X" and s1 == "X":
        return ProfilePolicyDecision(status="reserve", reason="MIDDLE_MIDDLE_DOUBLE_HARD")
    if profile_ligation_support(normalized) < context.min_ligation_support:
        return ProfilePolicyDecision(status="reserve", reason="INSUFFICIENT_LIGATION_SUPPORT")
    if profile_effective_disruption(normalized) > context.max_effective_disruption:
        return ProfilePolicyDecision(status="reserve", reason="EXCESSIVE_EFFECTIVE_DISRUPTION")
    if hard_count == 2 and s3 != "X":
        return ProfilePolicyDecision(status="reserve", reason="DOUBLE_HARD_WITHOUT_S3_EDGE")
    return ProfilePolicyDecision(status="active", reason="ACTIVE_PROFILE_POLICY")


__all__ = [
    "ProfilePolicyDecision",
    "ProfilePolicyStatus",
    "classify_profile_policy",
    "normalize_profile",
    "profile_effective_disruption",
    "profile_ligation_support",
    "profile_non_watson_crick_count",
]
