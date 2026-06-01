from __future__ import annotations

import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.tfbs.stage_b.semantics import (
    TFBS_STAGE_B_ORACLE_ROLES,
    TFBS_STAGE_B_PROBE_FAMILY,
    TFBS_STAGE_B_SCOPE,
    TFBS_STAGE_B_SPLIT_ID,
    TFBS_STAGE_B_STAGE,
    TfbsStageBRunIdentity,
    stage_b_dataset_id,
    validate_stage_b_oracle_role,
    validate_stage_b_split_id,
)


def test_stage_b_identity_terms_are_stable() -> None:
    identity = TfbsStageBRunIdentity(
        label_name="lexA_present",
        oracle_role="positive",
        split_id=TFBS_STAGE_B_SPLIT_ID,
        seed=7,
    )

    assert TFBS_STAGE_B_PROBE_FAMILY == "densegen_tfbs_learnability_probe_v1"
    assert TFBS_STAGE_B_STAGE == "B"
    assert TFBS_STAGE_B_SCOPE == "stage_b_sentinel_initial"
    assert TFBS_STAGE_B_ORACLE_ROLES == ("positive", "matched_null")
    assert identity.run_key == "tfbs_lexA_present_positive_random_id_seed7"
    assert identity.campaign_slug == "tfbs_v1_lexa_present_positive_random_id_seed7"
    assert stage_b_dataset_id(split_id=TFBS_STAGE_B_SPLIT_ID, seed=7).endswith("_tfbs_random_id_seed7")


def test_stage_b_semantics_fail_fast_on_unknown_terms() -> None:
    with pytest.raises(ValueError, match="unsupported Stage B oracle role"):
        validate_stage_b_oracle_role("negative")
    with pytest.raises(ValueError, match="supports split_id"):
        validate_stage_b_split_id("leave_sigma35_variant")
