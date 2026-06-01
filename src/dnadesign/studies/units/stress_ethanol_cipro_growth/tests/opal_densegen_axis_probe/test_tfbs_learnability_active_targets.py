from __future__ import annotations

import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.tfbs.active_targets import (
    minimum_tfbs_learnability_target_set,
    tfbs_learnability_active_target_spec,
    tfbs_learnability_sentinel_target_specs,
    validate_tfbs_learnability_target_set,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.tfbs.schema import (
    TFBS_LEARNABILITY_MINIMUM_TARGET_SET,
    TFBS_LEARNABILITY_SENTINEL_TARGET_SET,
)


def test_binary_presence_target_uses_expected_scalar_channel_not_plan_similarity() -> None:
    spec = tfbs_learnability_active_target_spec("lexA_present")

    assert spec.label_family_id == "tf_family_presence"
    assert spec.y_expected_length == 1
    assert spec.transforms_y["name"] == "vector_from_table_v1"
    assert spec.transforms_y["params"]["value_columns"] == ["lexA_present"]
    assert spec.objectives == (
        {
            "name": "vector_channel_v1",
            "params": {"channel_index": 0, "channel_name": "lexA_present", "mode": "maximize"},
        },
    )
    assert spec.score_ref == "vector_channel_v1/lexA_present"
    assert spec.score_label == "Predicted P(LexA present)"
    assert "vector_target_similarity_v1" not in str(spec.to_dict())
    assert "wet-lab phenotype" in spec.interpretation_boundary


def test_count_fraction_and_slot_targets_have_explicit_axis_labels() -> None:
    fraction = tfbs_learnability_active_target_spec("lexA_count_fraction")
    slot = tfbs_learnability_active_target_spec("cpxR_or_baeR_in_slot2")

    assert fraction.label_family_id == "tf_family_count_fraction"
    assert fraction.score_label == "Predicted E[LexA count / 3]"
    assert fraction.score_axis["limits"] == [0.0, 1.0]
    assert slot.label_family_id == "tf_slot_family_presence"
    assert slot.score_label == "Predicted P(CpxR or BaeR in rightmost TFBS slot)"
    assert slot.score_axis["scale_class"] == "tfbs_expected_scalar_unit_interval"


def test_sentinel_and_minimum_target_sets_are_strict_v1_labels() -> None:
    assert (
        tuple(spec.label_name for spec in tfbs_learnability_sentinel_target_specs())
        == TFBS_LEARNABILITY_SENTINEL_TARGET_SET
    )
    assert minimum_tfbs_learnability_target_set() == TFBS_LEARNABILITY_MINIMUM_TARGET_SET
    assert validate_tfbs_learnability_target_set(("lexA_present", "lexA_present", "cpxR_present")) == (
        "lexA_present",
        "cpxR_present",
    )


def test_tfbs_target_validation_rejects_plan_logic_and_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unsupported TFBS learnability active label"):
        tfbs_learnability_active_target_spec("densegen_plan_logic4")

    with pytest.raises(ValueError, match="unsupported TFBS learnability target label"):
        validate_tfbs_learnability_target_set(("lexA_present", "surprise_label"))
