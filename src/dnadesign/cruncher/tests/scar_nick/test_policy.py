"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/scar_nick/test_policy.py

Profile policy tests for ligation-aware scar-nick panel selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.scar_nick.models import CandidateRankingContext
from dnadesign.cruncher.scar_nick.policy import classify_profile_policy


def _context() -> CandidateRankingContext:
    return CandidateRankingContext(
        target_profile_buckets=[],
        reject_profiles=["MMMM"],
        allow_gt_wobble=True,
        active_max_hard_mismatches=2,
        active_max_non_watson_crick_pairs=2,
        forbid_active_middle_middle_double_hard=True,
    )


def test_s0_must_match_at_profile_policy_boundary() -> None:
    decision = classify_profile_policy("MXXX", context=_context())

    assert decision.status == "reject"
    assert decision.reason == "S0_PAIR_NOT_WATSON_CRICK"


def test_middle_middle_double_hard_is_reserve_not_active() -> None:
    decision = classify_profile_policy("MXXM", context=_context())

    assert decision.status == "reserve"
    assert decision.reason == "MIDDLE_MIDDLE_DOUBLE_HARD"


def test_edge_including_double_hard_profiles_are_active() -> None:
    context = _context()

    assert classify_profile_policy("XXMM", context=context).status == "active"
    assert classify_profile_policy("XMXM", context=context).status == "active"


def test_two_wobble_profiles_are_active() -> None:
    context = _context()

    for profile in ["WWMM", "WMWM", "MWWM"]:
        assert classify_profile_policy(profile, context=context).status == "active"


def test_three_non_wc_profiles_are_reserve() -> None:
    context = _context()

    for profile in ["XWWM", "WXWM", "WWXM", "WWWM"]:
        decision = classify_profile_policy(profile, context=context)
        assert decision.status == "reserve"
        assert decision.reason == "MORE_THAN_TWO_NON_WATSON_CRICK"
