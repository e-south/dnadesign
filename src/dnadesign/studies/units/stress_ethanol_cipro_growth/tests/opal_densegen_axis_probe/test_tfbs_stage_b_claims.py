from __future__ import annotations

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.tfbs.stage_b.claims import (
    build_tfbs_stage_b_claim_assessment,
)


def test_stage_b_claim_assessment_separates_valid_null_signals_from_confound_controls() -> None:
    pair_summary = pd.DataFrame(
        [
            {
                "label_name": "lexA_present",
                "label_family_id": "tf_family_presence",
                "negative_control_claim_status": "VALID_AS_NEGATIVE_CONTROL",
                "peer_review_claim_status": "positive_exceeds_null",
                "final_positive_minus_null_lift_ratio": 1.02,
                "trapezoid_auc_positive_minus_null_lift_ratio": 1.00,
            },
            {
                "label_name": "lexA_in_slot0",
                "label_family_id": "tf_slot_family_presence",
                "negative_control_claim_status": "CONFOUND_CONTROL_ONLY",
                "peer_review_claim_status": "null_is_confound_control_only",
                "final_positive_minus_null_lift_ratio": 1.63,
                "trapezoid_auc_positive_minus_null_lift_ratio": 2.19,
            },
            {
                "label_name": "cpxR_or_baeR_present",
                "label_family_id": "tf_family_presence",
                "negative_control_claim_status": "VALID_AS_NEGATIVE_CONTROL",
                "peer_review_claim_status": "not_separated_from_null",
                "final_positive_minus_null_lift_ratio": -0.2,
                "trapezoid_auc_positive_minus_null_lift_ratio": 0.1,
            },
        ]
    )

    claims = build_tfbs_stage_b_claim_assessment(pair_summary)

    status_by_label = dict(zip(claims["label_name"], claims["claim_readiness_status"], strict=True))
    assert status_by_label == {
        "lexA_present": "READY_AS_VALID_NULL_LEARNABILITY_SIGNAL",
        "lexA_in_slot0": "LIMITED_TO_CONFOUND_CONTROL_DIAGNOSTIC",
        "cpxR_or_baeR_present": "BLOCKED_NOT_SEPARATED_FROM_NULL",
    }
    assert bool(claims.loc[claims["label_name"] == "lexA_present", "claim_readiness_bool"].iloc[0]) is True
    assert claims.loc[claims["label_name"] == "lexA_in_slot0", "manuscript_claim_boundary"].iloc[0] == (
        "Report as a confound-control diagnostic only; do not claim valid-null learnability separation."
    )


def test_stage_b_claim_assessment_fails_fast_on_missing_peer_review_columns() -> None:
    with pytest.raises(ValueError, match="missing column"):
        build_tfbs_stage_b_claim_assessment(pd.DataFrame({"label_name": ["lexA_present"]}))


def test_stage_b_claim_assessment_fails_fast_on_nonfinite_delta() -> None:
    pair_summary = pd.DataFrame(
        [
            {
                "label_name": "lexA_present",
                "label_family_id": "tf_family_presence",
                "negative_control_claim_status": "VALID_AS_NEGATIVE_CONTROL",
                "peer_review_claim_status": "positive_exceeds_null",
                "final_positive_minus_null_lift_ratio": float("inf"),
                "trapezoid_auc_positive_minus_null_lift_ratio": 1.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="non-finite final lift delta"):
        build_tfbs_stage_b_claim_assessment(pair_summary)
