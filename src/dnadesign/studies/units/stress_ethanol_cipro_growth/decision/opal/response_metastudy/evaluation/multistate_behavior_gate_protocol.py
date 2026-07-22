"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_gate_protocol.py

Study-owned completion-gate contract for the behavior shadow evaluation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .multistate_behavior_protocol_fields import (
    BehaviorProtocolError,
    require_exact_fields,
    require_literal,
    require_mapping,
)


@dataclass(frozen=True)
class BehaviorFaceValidityControl:
    """Existing assay north star used only for biological face-validity review."""

    selection_view_id: str
    design_id: str
    display_label: str


@dataclass(frozen=True)
class BehaviorCompletionGateProtocol:
    """Prespecified evidence design; it is not an automatic promotion rule."""

    normalization_quantiles: tuple[float, ...]
    normalization_primary_quantile: float
    normalization_holdout: Literal["leave_one_source_experiment_out"]
    validation_label_source: Literal["verified_observed_label_promotion"]
    validation_seeds: tuple[int, ...]
    validation_split: Literal["leave_one_label_source_experiment_out"]
    validation_minimum_source_experiment_groups: int
    validation_x_preprocessing: Literal["identity_train_fold_only"]
    validation_y_fit_space: Literal["raw_reader_response_window_vector_v1"]
    validation_scoring_parameters: Literal["train_fold_only_exclude_heldout_experiment_and_candidates"]
    validation_primary_metric: Literal["median_within_heldout_group_spearman"]
    validation_secondary_metric: Literal["pooled_oof_spearman"]
    validation_model_source: Literal["registered_prediction_run_receipt"]
    validation_model_name: Literal["random_forest"]
    validation_model_nonseed_params_sha256: str
    allocation_strategy: Literal["round_robin_next_best_unallocated"]
    allocation_deduplicate_by: Literal["sequence"]
    allocation_expected_unique_count: int
    allocation_view_priority: tuple[str, ...]
    face_validity_controls: tuple[BehaviorFaceValidityControl, ...]
    face_validity_unclaimed_positive_control_views: tuple[str, ...]
    face_validity_evidence_role: Literal["diagnostic_only_no_acceptance_threshold"]


def parse_behavior_completion_gate(payload: object) -> BehaviorCompletionGateProtocol:
    """Parse exact, prespecified robustness and validation conventions."""

    gate = require_mapping(payload, context="completion_gate")
    require_exact_fields(
        gate,
        {
            "normalization_sensitivity",
            "grouped_objective_validation",
            "allocation_comparison",
            "biological_face_validity",
        },
        context="completion_gate",
    )
    normalization = require_mapping(
        gate["normalization_sensitivity"],
        context="completion_gate.normalization_sensitivity",
    )
    require_exact_fields(
        normalization,
        {"quantiles", "primary_quantile", "source_experiment_holdout"},
        context="completion_gate.normalization_sensitivity",
    )
    raw_quantiles = normalization["quantiles"]
    if not isinstance(raw_quantiles, list) or any(isinstance(value, bool) for value in raw_quantiles):
        raise BehaviorProtocolError("completion-gate normalization quantiles must be a numeric list.")
    try:
        quantiles = tuple(float(value) for value in raw_quantiles)
    except (TypeError, ValueError) as exc:
        raise BehaviorProtocolError("completion-gate normalization quantiles must be numeric.") from exc
    if quantiles != (0.50, 0.75, 0.90, 0.95, 0.99):
        raise BehaviorProtocolError("completion-gate normalization quantiles must be q50, q75, q90, q95, q99.")
    if float(normalization["primary_quantile"]) != 0.90:
        raise BehaviorProtocolError("completion-gate primary normalization quantile must be q90.")
    require_literal(
        normalization,
        "source_experiment_holdout",
        "leave_one_source_experiment_out",
        context="completion_gate.normalization_sensitivity",
    )

    validation = require_mapping(
        gate["grouped_objective_validation"],
        context="completion_gate.grouped_objective_validation",
    )
    require_exact_fields(
        validation,
        {
            "label_source",
            "seeds",
            "split_strategy",
            "minimum_source_experiment_groups",
            "x_preprocessing",
            "y_fit_space",
            "scoring_parameters",
            "primary_metric",
            "secondary_metric",
            "model_source",
            "model_name",
            "model_nonseed_params_sha256",
        },
        context="completion_gate.grouped_objective_validation",
    )
    literals = {
        "label_source": "verified_observed_label_promotion",
        "split_strategy": "leave_one_label_source_experiment_out",
        "x_preprocessing": "identity_train_fold_only",
        "y_fit_space": "raw_reader_response_window_vector_v1",
        "scoring_parameters": "train_fold_only_exclude_heldout_experiment_and_candidates",
        "primary_metric": "median_within_heldout_group_spearman",
        "secondary_metric": "pooled_oof_spearman",
        "model_source": "registered_prediction_run_receipt",
        "model_name": "random_forest",
    }
    for field, expected in literals.items():
        require_literal(validation, field, expected, context="completion_gate.grouped_objective_validation")
    seeds = validation["seeds"]
    if seeds != [3, 7, 19, 29, 43]:
        raise BehaviorProtocolError("completion-gate grouped validation seeds must be [3, 7, 19, 29, 43].")
    if validation["minimum_source_experiment_groups"] != 3:
        raise BehaviorProtocolError("completion-gate grouped validation requires at least three source experiments.")
    model_digest = validation["model_nonseed_params_sha256"]
    if (
        not isinstance(model_digest, str)
        or len(model_digest) != 64
        or any(character not in "0123456789abcdef" for character in model_digest)
    ):
        raise BehaviorProtocolError("completion-gate model_nonseed_params_sha256 must be a lowercase SHA-256 digest.")

    allocation = require_mapping(gate["allocation_comparison"], context="completion_gate.allocation_comparison")
    require_exact_fields(
        allocation,
        {"strategy", "deduplicate_by", "expected_unique_count", "view_priority"},
        context="completion_gate.allocation_comparison",
    )
    require_literal(
        allocation,
        "strategy",
        "round_robin_next_best_unallocated",
        context="completion_gate.allocation_comparison",
    )
    require_literal(allocation, "deduplicate_by", "sequence", context="completion_gate.allocation_comparison")
    if allocation["expected_unique_count"] != 18:
        raise BehaviorProtocolError("completion-gate allocation must require exactly 18 unique sequences.")
    if allocation["view_priority"] != ["ethanol", "ciprofloxacin", "and"]:
        raise BehaviorProtocolError("completion-gate allocation view priority drifted.")

    face_validity = require_mapping(
        gate["biological_face_validity"], context="completion_gate.biological_face_validity"
    )
    require_exact_fields(
        face_validity,
        {"controls", "unclaimed_positive_control_views", "evidence_role"},
        context="completion_gate.biological_face_validity",
    )
    expected_controls = [
        {"selection_view_id": "ethanol", "design_id": "pDual-10-spyp", "display_label": "SpyP"},
        {"selection_view_id": "ciprofloxacin", "design_id": "pDual-10-sulAp", "display_label": "sulAp"},
    ]
    if face_validity["controls"] != expected_controls:
        raise BehaviorProtocolError("completion-gate biological face-validity controls drifted.")
    if face_validity["unclaimed_positive_control_views"] != ["and"]:
        raise BehaviorProtocolError("completion-gate unclaimed positive-control views drifted.")
    require_literal(
        face_validity,
        "evidence_role",
        "diagnostic_only_no_acceptance_threshold",
        context="completion_gate.biological_face_validity",
    )

    return BehaviorCompletionGateProtocol(
        normalization_quantiles=quantiles,
        normalization_primary_quantile=0.90,
        normalization_holdout="leave_one_source_experiment_out",
        validation_label_source="verified_observed_label_promotion",
        validation_seeds=(3, 7, 19, 29, 43),
        validation_split="leave_one_label_source_experiment_out",
        validation_minimum_source_experiment_groups=3,
        validation_x_preprocessing="identity_train_fold_only",
        validation_y_fit_space="raw_reader_response_window_vector_v1",
        validation_scoring_parameters="train_fold_only_exclude_heldout_experiment_and_candidates",
        validation_primary_metric="median_within_heldout_group_spearman",
        validation_secondary_metric="pooled_oof_spearman",
        validation_model_source="registered_prediction_run_receipt",
        validation_model_name="random_forest",
        validation_model_nonseed_params_sha256=model_digest,
        allocation_strategy="round_robin_next_best_unallocated",
        allocation_deduplicate_by="sequence",
        allocation_expected_unique_count=18,
        allocation_view_priority=("ethanol", "ciprofloxacin", "and"),
        face_validity_controls=tuple(BehaviorFaceValidityControl(**record) for record in expected_controls),
        face_validity_unclaimed_positive_control_views=("and",),
        face_validity_evidence_role="diagnostic_only_no_acceptance_threshold",
    )


__all__ = [
    "BehaviorCompletionGateProtocol",
    "BehaviorFaceValidityControl",
    "parse_behavior_completion_gate",
]
