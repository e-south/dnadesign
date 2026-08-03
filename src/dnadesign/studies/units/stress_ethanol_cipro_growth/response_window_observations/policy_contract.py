"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/policy_contract.py

Exact schema and scientific semantics for the observation policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

SCHEMA_ID = "stress_ethanol_cipro_growth.response_window_observation_policy.v3"
SCHEMA_VERSION = "3"
STUDY_ID = "stress_ethanol_cipro_growth"
APPROVAL_STATUSES = frozenset({"review_required", "approved"})

TOP_LEVEL_FIELDS = {
    "schema_id",
    "schema_version",
    "study_id",
    "policy_id",
    "approval",
    "source_manifests",
    "label_identity",
    "aggregation",
    "censoring",
    "unbound_reader_designs",
    "repeat_decisions",
}
APPROVAL_FIELDS = {"status", "approved_by", "approved_at", "rationale"}
SOURCE_MANIFEST_FIELDS = {
    "reader_bundle_sha256",
    "reader_record_receipt_sha256",
    "candidate_bindings_sha256",
}
LABEL_FIELDS = {"y_space", "observed_round", "batch_id", "primary_reduction_id", "value_order"}
AGGREGATION_FIELDS = {
    "experiment_unit",
    "label_source_strategy",
    "singleton",
    "repeated",
    "uncertainty",
    "event_time_sensitivity",
}
UNCERTAINTY_FIELDS = {
    "method",
    "experiment_resampling",
    "reader_draw_resampling",
    "samples",
    "confidence_level",
    "random_seed",
    "minimum_reader_draws_per_experiment",
}
EXPECTED_AGGREGATION_SEMANTICS = {
    "experiment_unit": "reader_experiment",
    "label_source_strategy": "explicit_policy_selection",
    "singleton": "identity",
    "repeated": "selected_reader_experiment_identity",
    "event_time_sensitivity": "separate",
}
EXPECTED_UNCERTAINTY_SEMANTICS = {
    "method": "selected_reader_joint_bootstrap",
    "experiment_resampling": "none",
    "reader_draw_resampling": "one_joint_draw_per_sample",
}
CENSORING_FIELDS = {"primary_value_requirement", "nonexact_label_action"}
EXPECTED_CENSORING_SEMANTICS = {
    "primary_value_requirement": "exact",
    "nonexact_label_action": "exclude_candidate",
}

__all__ = [
    "AGGREGATION_FIELDS",
    "APPROVAL_FIELDS",
    "APPROVAL_STATUSES",
    "CENSORING_FIELDS",
    "EXPECTED_AGGREGATION_SEMANTICS",
    "EXPECTED_CENSORING_SEMANTICS",
    "EXPECTED_UNCERTAINTY_SEMANTICS",
    "LABEL_FIELDS",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SOURCE_MANIFEST_FIELDS",
    "STUDY_ID",
    "TOP_LEVEL_FIELDS",
    "UNCERTAINTY_FIELDS",
]
