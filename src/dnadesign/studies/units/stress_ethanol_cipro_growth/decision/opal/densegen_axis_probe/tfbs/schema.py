"""Stable schema constants for the DenseGen TFBS learnability probe v1."""

from __future__ import annotations

import hashlib
import json

TFBS_LEARNABILITY_ORACLE_VERSION = "densegen_tfbs_learnability_positive_v1"
TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION = "densegen_tfbs_learnability_family_content_matched_null_v1"
TFBS_LEARNABILITY_SLOT_GEOMETRY_NULL_VERSION = "densegen_tfbs_learnability_slot_geometry_count_matched_null_v1"
TFBS_LEARNABILITY_SLOT_POSITION_COUNT_FIXED_NULL_VERSION = (
    "densegen_tfbs_learnability_slot_position_count_fixed_shuffled_null_v1"
)
TFBS_LEARNABILITY_SCHEMA_VERSION = "stress_ethanol_cipro_growth.densegen_tfbs_learnability.v1"
TFBS_LEARNABILITY_ACTIVE_LABEL_FAMILIES = (
    "tf_family_count",
    "tf_family_presence",
    "tf_family_count_fraction",
    "tf_slot_family_presence",
)
TFBS_LEARNABILITY_REQUIRED_LABEL_COLUMNS = (
    "id",
    "quality_flag",
    "lexA_count",
    "cpxR_count",
    "baeR_count",
    "cpxR_or_baeR_count",
    "lexA_present",
    "cpxR_present",
    "baeR_present",
    "cpxR_or_baeR_present",
    "lexA_count_fraction",
    "cpxR_count_fraction",
    "baeR_count_fraction",
    "cpxR_or_baeR_count_fraction",
    "lexA_in_slot0",
    "lexA_in_slot1",
    "lexA_in_slot2",
    "cpxR_or_baeR_in_slot0",
    "cpxR_or_baeR_in_slot1",
    "cpxR_or_baeR_in_slot2",
    "slot0_family",
    "slot1_family",
    "slot2_family",
    "sigma35_variant",
    "sigma10_consensus_identity",
    "spacer_length",
    "sigma35_offset_raw",
    "sigma10_offset_raw",
    "sigma35_end_raw",
    "sigma10_end_raw",
    "oracle_version",
    "label_recipe_hash",
)
TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES = (
    "lexA_present",
    "cpxR_present",
    "baeR_present",
    "cpxR_or_baeR_present",
    "lexA_count_fraction",
    "cpxR_count_fraction",
    "baeR_count_fraction",
    "cpxR_or_baeR_count_fraction",
    "lexA_in_slot0",
    "lexA_in_slot1",
    "lexA_in_slot2",
    "cpxR_or_baeR_in_slot0",
    "cpxR_or_baeR_in_slot1",
    "cpxR_or_baeR_in_slot2",
)
TFBS_LEARNABILITY_MINIMUM_TARGET_SET = (
    "lexA_present",
    "cpxR_present",
    "baeR_present",
    "cpxR_or_baeR_present",
    "lexA_count_fraction",
    "cpxR_or_baeR_count_fraction",
    "lexA_in_slot0",
    "lexA_in_slot1",
    "lexA_in_slot2",
    "cpxR_or_baeR_in_slot0",
    "cpxR_or_baeR_in_slot1",
    "cpxR_or_baeR_in_slot2",
)
TFBS_LEARNABILITY_CANONICAL_COUNT_FRACTION_TARGET_SET = (
    "lexA_count_fraction",
    "cpxR_count_fraction",
    "baeR_count_fraction",
)
TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET = (
    "lexA_in_slot0",
    "lexA_in_slot1",
    "lexA_in_slot2",
    "cpxR_or_baeR_in_slot0",
    "cpxR_or_baeR_in_slot1",
    "cpxR_or_baeR_in_slot2",
)
TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET = (
    "lexA_in_slot0",
    "cpxR_or_baeR_in_slot2",
)
TFBS_LEARNABILITY_SLOT_POSITION_COUNT_FIXED_SENTINEL_TARGET_SET = (
    *TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET,
)
TFBS_LEARNABILITY_SENTINEL_TARGET_SET = (*TFBS_LEARNABILITY_CANONICAL_COUNT_FRACTION_TARGET_SET,)
TFBS_LEARNABILITY_NULL_VIABILITY_STATUSES = (
    "PASS",
    "PASS_WITH_COARSENING",
    "FAIL_WEAK_EXCHANGEABILITY",
    "FAIL_LABEL_DISTRIBUTION_CHANGED",
    "FAIL_COUNT_MATCHING_CHANGED",
)

_LABEL_RECIPE_PAYLOAD = {
    "schema_version": TFBS_LEARNABILITY_SCHEMA_VERSION,
    "oracle_version": TFBS_LEARNABILITY_ORACLE_VERSION,
    "active_label_families": list(TFBS_LEARNABILITY_ACTIVE_LABEL_FAMILIES),
    "slot_coordinate": "offset_raw",
    "tfbs_entries_per_row": 3,
    "fixed_elements_per_row": 2,
}
TFBS_LEARNABILITY_LABEL_RECIPE_HASH = hashlib.sha256(
    json.dumps(_LABEL_RECIPE_PAYLOAD, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
