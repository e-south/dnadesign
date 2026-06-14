from __future__ import annotations

import pytest

from .probe_modules import probe_module

_active_targets = probe_module("tfbs.active_targets")
minimum_tfbs_learnability_target_set = _active_targets.minimum_tfbs_learnability_target_set
tfbs_learnability_active_target_spec = _active_targets.tfbs_learnability_active_target_spec
tfbs_learnability_sentinel_target_specs = _active_targets.tfbs_learnability_sentinel_target_specs
validate_tfbs_learnability_target_set = _active_targets.validate_tfbs_learnability_target_set

_schema = probe_module("tfbs.schema")
TFBS_LEARNABILITY_CANONICAL_COUNT_FRACTION_TARGET_SET = _schema.TFBS_LEARNABILITY_CANONICAL_COUNT_FRACTION_TARGET_SET
TFBS_LEARNABILITY_MINIMUM_TARGET_SET = _schema.TFBS_LEARNABILITY_MINIMUM_TARGET_SET
TFBS_LEARNABILITY_SENTINEL_TARGET_SET = _schema.TFBS_LEARNABILITY_SENTINEL_TARGET_SET
TFBS_LEARNABILITY_SLOT_POSITION_COUNT_FIXED_SENTINEL_TARGET_SET = (
    _schema.TFBS_LEARNABILITY_SLOT_POSITION_COUNT_FIXED_SENTINEL_TARGET_SET
)
TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET = _schema.TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET
TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET = _schema.TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET

_profiles = probe_module("tfbs.profiles")
CANONICAL_COUNT_FRACTION_PROFILE_ID = _profiles.CANONICAL_COUNT_FRACTION_PROFILE_ID
SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID = _profiles.SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID
SLOT_POSITION_SENTINEL_PROFILE_ID = _profiles.SLOT_POSITION_SENTINEL_PROFILE_ID
SLOT_POSITION_PROFILE_ID = _profiles.SLOT_POSITION_PROFILE_ID
canonical_count_fraction_label_names = _profiles.canonical_count_fraction_label_names
slot_position_count_fixed_sentinel_label_names = _profiles.slot_position_count_fixed_sentinel_label_names
slot_position_label_names = _profiles.slot_position_label_names
slot_position_sentinel_label_names = _profiles.slot_position_sentinel_label_names
tfbs_label_names_for_profile_id = _profiles.tfbs_label_names_for_profile_id
tfbs_target_profile_for_labels = _profiles.tfbs_target_profile_for_labels
tfbs_target_profile_for_profile_id = _profiles.tfbs_target_profile_for_profile_id


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
    assert TFBS_LEARNABILITY_SENTINEL_TARGET_SET == TFBS_LEARNABILITY_CANONICAL_COUNT_FRACTION_TARGET_SET
    assert canonical_count_fraction_label_names() == (
        "lexA_count_fraction",
        "cpxR_count_fraction",
        "baeR_count_fraction",
    )
    assert (
        tuple(spec.label_name for spec in tfbs_learnability_sentinel_target_specs())
        == TFBS_LEARNABILITY_SENTINEL_TARGET_SET
    )
    assert minimum_tfbs_learnability_target_set() == TFBS_LEARNABILITY_MINIMUM_TARGET_SET
    assert validate_tfbs_learnability_target_set(("lexA_present", "lexA_present", "cpxR_present")) == (
        "lexA_present",
        "cpxR_present",
    )


def test_tfbs_target_profiles_separate_canonical_positional_and_custom_label_sets() -> None:
    canonical = tfbs_target_profile_for_labels(TFBS_LEARNABILITY_SENTINEL_TARGET_SET).to_manifest()
    with pytest.raises(ValueError, match="ambiguous TFBS slot-position sentinel labels"):
        tfbs_target_profile_for_labels(TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET)
    sentinel = tfbs_target_profile_for_profile_id(SLOT_POSITION_SENTINEL_PROFILE_ID).to_manifest()
    count_fixed = tfbs_target_profile_for_profile_id(SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID).to_manifest()
    positional = tfbs_target_profile_for_labels(TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET).to_manifest()
    custom = tfbs_target_profile_for_labels(("lexA_present",)).to_manifest()

    assert canonical["profile_id"] == CANONICAL_COUNT_FRACTION_PROFILE_ID
    assert canonical["canonical"] is True
    assert canonical["label_family_ids"] == ["tf_family_count_fraction"]
    assert slot_position_sentinel_label_names() == TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET
    assert (
        tfbs_label_names_for_profile_id(SLOT_POSITION_SENTINEL_PROFILE_ID)
        == TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET
    )
    assert tfbs_target_profile_for_profile_id(SLOT_POSITION_SENTINEL_PROFILE_ID).to_manifest() == sentinel
    assert sentinel["profile_id"] == SLOT_POSITION_SENTINEL_PROFILE_ID
    assert sentinel["profile_role"] == "boundary_stage_b_sentinel_probe"
    assert sentinel["canonical"] is False
    assert sentinel["label_names"] == ["lexA_in_slot0", "cpxR_or_baeR_in_slot2"]
    assert sentinel["label_family_ids"] == ["tf_slot_family_presence"]
    assert "sentinel" in sentinel["interpretation_boundary"].lower()
    assert (
        slot_position_count_fixed_sentinel_label_names()
        == TFBS_LEARNABILITY_SLOT_POSITION_COUNT_FIXED_SENTINEL_TARGET_SET
    )
    assert count_fixed["profile_id"] == SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID
    assert count_fixed["profile_role"] == "boundary_stage_b_count_fixed_sentinel_probe"
    assert count_fixed["canonical"] is False
    assert count_fixed["label_names"] == ["lexA_in_slot0", "cpxR_or_baeR_in_slot2"]
    assert "exactly one target-family motif" in count_fixed["interpretation_boundary"]
    assert slot_position_label_names() == TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET
    assert tfbs_label_names_for_profile_id(SLOT_POSITION_PROFILE_ID) == TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET
    assert tfbs_target_profile_for_profile_id(SLOT_POSITION_PROFILE_ID).to_manifest() == positional
    assert positional["profile_id"] == SLOT_POSITION_PROFILE_ID
    assert positional["profile_role"] == "boundary_stage_b_probe"
    assert positional["canonical"] is False
    assert positional["label_family_ids"] == ["tf_slot_family_presence"]
    assert "count-matched slot-position" in positional["interpretation_boundary"].lower()
    assert custom["profile_id"] == "custom_tfbs_learnability_label_set"
    assert custom["canonical"] is False
    assert custom["label_family_ids"] == ["tf_family_presence"]


def test_tfbs_target_validation_rejects_plan_logic_and_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unsupported TFBS learnability active label"):
        tfbs_learnability_active_target_spec("densegen_plan_logic4")

    with pytest.raises(ValueError, match="unsupported TFBS learnability target label"):
        validate_tfbs_learnability_target_set(("lexA_present", "surprise_label"))
