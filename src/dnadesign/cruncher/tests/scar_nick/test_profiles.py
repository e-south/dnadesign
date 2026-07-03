"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/scar_nick/test_profiles.py

Profile semantics for terminal scar-nick junctions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.scar_nick.profiles import classify_pair_profile, profile_label_s3s2s1s0


def test_s3_s2_s1_s0_profile_matches_reference_controls() -> None:
    assert classify_pair_profile("CGGG", "ACAG").profile_s3s2s1s0 == "MXMX"
    assert classify_pair_profile("CAAG", "CTCG").profile_s3s2s1s0 == "MXMM"
    assert profile_label_s3s2s1s0("CGGG", "ACAG") == "MXMX"
    assert profile_label_s3s2s1s0("CAAG", "CTCG") == "MXMM"


def test_profile_reports_pair_classes_in_s3_s2_s1_s0_order() -> None:
    profile = classify_pair_profile("CGGG", "ACAG")

    assert profile.profile_s3s2s1s0 == "MXMX"
    assert profile.profile_payload_outward == "XMXM"
    assert [entry.position for entry in profile.pairs] == [0, 1, 2, 3]
    assert [entry.site for entry in profile.pairs] == ["S3", "S2", "S1", "S0"]
    assert [entry.class_label for entry in profile.pairs] == ["M", "X", "M", "X"]
    assert profile.hard_mismatch_count == 2
    assert profile.worst_hard_mismatch_tier == 2
    assert profile.hard_mismatch_tier_sum == 4
    assert profile.middle_hard_mismatch_tier_sum == 2
    assert profile.edge_hard_mismatch_tier_sum == 2
    assert profile.s0_class == "X"


def test_gt_wobble_is_explicit_when_allowed() -> None:
    wobble = classify_pair_profile("GGGG", "TTTC", allow_gt_wobble=True)
    strict = classify_pair_profile("GGGG", "TTTC", allow_gt_wobble=False)

    assert wobble.profile_s3s2s1s0 == "MWWW"
    assert profile_label_s3s2s1s0("GGGG", "TTTC", allow_gt_wobble=True) == "MWWW"
    assert wobble.wobble_count == 3
    assert wobble.middle_wobble_count == 2
    assert wobble.hard_mismatch_count == 0
    assert wobble.hard_mismatch_tier_sum == 0
    assert wobble.non_watson_crick_count == 3
    assert strict.profile_s3s2s1s0 == "MXXX"
    assert profile_label_s3s2s1s0("GGGG", "TTTC", allow_gt_wobble=False) == "MXXX"
    assert strict.hard_mismatch_count == 3
    assert strict.worst_hard_mismatch_tier == 0
    assert strict.hard_mismatch_tier_sum == 0
    assert strict.non_watson_crick_count == 3
