"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_decision.py

Typed study decision for the multistate behavior shadow evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .multistate_behavior_shadow import VerifiedMultistateBehaviorShadow

DECISION_SCHEMA_ID = "stress_ethanol_cipro_growth.multistate_response_behavior_shadow_decision.v1"
_EVIDENCE_TABLES = (
    "normalization_sensitivity",
    "grouped_objective_validation",
    "allocation_comparison",
    "observed_control_face_validity",
    "family_cardinality_pressure",
    "rmf_replay_calibration",
    "prediction_vectors",
    "prediction_scores",
    "prediction_surface_diagnostics",
    "hard_behavior_summary",
)


def build_multistate_behavior_decision(
    preview: VerifiedMultistateBehaviorShadow,
    *,
    artifact_inventory: Mapping[str, Mapping[str, object]],
    independent_audit: Mapping[str, object],
    independent_audit_sha256: str,
) -> dict[str, object]:
    """Build an honest split verdict; semantic fit is not predictive efficacy."""

    labels = preview.completion.validation_labels
    audit = dict(independent_audit)
    audit_blockers = audit.get("blockers")
    audit_passed = audit.get("status") == "pass" and isinstance(audit_blockers, list) and not audit_blockers
    evidence = {
        table_id: _artifact_receipt(artifact_inventory, artifact_id=f"table__{table_id}")
        for table_id in _EVIDENCE_TABLES
    }
    return {
        "schema_id": DECISION_SCHEMA_ID,
        "schema_version": "1",
        "study_id": preview.normalization.protocol.study_id,
        "protocol_id": preview.normalization.protocol.protocol_id,
        "promotion_decision": "no_go",
        "semantic_fit": {
            "verdict": "go",
            "basis": "strictly_monotonic_threshold_free_three_family_behavior_matches_declared_intent",
        },
        "shadow_implementation": {
            "verdict": "go" if audit_passed else "no_go",
            "basis": (
                "pure_objective_property_tests_digest_bound_study_replay_and_independent_audit_pass"
                if audit_passed
                else "independent_adversarial_implementation_audit_has_unresolved_blockers"
            ),
        },
        "normalization_robustness": normalization_robustness_decision(preview.completion.normalization_sensitivity),
        "predictive_support": predictive_support_decision(
            preview.completion.grouped_objective_validation,
            promoted_candidate_count=labels.promoted_candidate_count,
        ),
        "prospective_hill_climb_efficacy": {
            "verdict": "unproven",
            "required_evidence": "predictions_frozen_before_new_measurements_across_prospective_rounds",
        },
        "technical_readiness": {
            "verdict": "go_for_shadow_review_only",
            "corrected_reader_reference_identity_verified": True,
            "reused_central_labels_exactly_equal": True,
            "new_observation_version_required": False,
            "reason": "corrected_bundle_changes_bootstrap_reference_identity_not_central_candidate_vectors",
        },
        "campaign_disposition": {
            "verdict": "no_go",
            "action": "do_not_activate_or_replace_a_campaign_without_explicit_study_signoff",
            "allocation_candidate_overlap": _allocation_overlap(preview.completion.allocation_comparison),
        },
        "synthesis": {"verdict": "prohibited"},
        "comparison_boundaries": {
            "same_prediction_objective_disagreement": "objective_behavior_on_one_fixed_raw_prediction_matrix",
            "prediction_to_truth": "grouped_out_of_fold_ordering_against_verified_exact_labels",
            "predictor_fit_space": "registered_random_forest_fits_raw_eight_component_y_not_the_behavior_scalar",
            "growth_viability_and_burden": "separate_assay_qc_not_encoded_by_this_objective",
            "state_space_scope": "within_view_ranking_same_ordered_states_target_mask_normalization_and_protocol_only",
            "cross_view_score_comparison": "prohibited",
            "cross_state_space_score_comparison": "prohibited_without_one_common_state_contract",
            "conformance_or_feasibility_claim": "not_supported_by_behavior_score",
            "hard_bottleneck_selection_role": "diagnostic_only_does_not_constrain_selection",
            "not_equivalent": True,
        },
        "source_equivalence": {
            "corrected_reader_bundle_manifest_sha256": preview.source["reader_bundle_manifest_sha256"],
            "prior_observation_reader_bundle_manifest_sha256": "sha256:"
            + preview.normalization.protocol.source_equivalence.prior_observation_reader_bundle_sha256,
            "central_label_equivalence_sha256": labels.central_label_equivalence_sha256,
            "label_artifact_sha256": labels.label_artifact_sha256,
            "promotion_manifest_sha256": labels.source["promotion_manifest_sha256"],
            "reference_unit_count": preview.reference_identity.reference_unit_count,
            "reference_bootstrap_row_count": preview.reference_identity.bootstrap_row_count,
            "normalization_reference_unit_count": 0,
        },
        "independent_adversarial_implementation_audit": {
            **audit,
            "evidence_sha256": _digest(independent_audit_sha256),
        },
        "evidence": evidence,
        "claim_boundary": "semantic_improvement_does_not_establish_better_predictive_hill_climbing",
    }


def _artifact_receipt(
    inventory: Mapping[str, Mapping[str, object]],
    *,
    artifact_id: str,
) -> dict[str, object]:
    record = inventory.get(artifact_id)
    if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
        raise ValueError(f"decision evidence artifact {artifact_id!r} is absent or malformed.")
    return dict(record)


def normalization_robustness_decision(frame: pd.DataFrame) -> dict[str, object]:
    """Project the prespecified normalization-sensitivity evidence into a verdict record."""

    return {
        "verdict": "characterized_not_tuned",
        "minimum_rank_spearman_vs_primary": float(frame["score_spearman_vs_primary"].min()),
        "minimum_raw_top_k_overlap": int(frame["raw_top_k_overlap"].min()),
        "scenario_count": int(frame["scenario_id"].nunique()),
        "claim": "sensitivity_describes_scale_dependence_and_is_not_an_acceptance_threshold",
    }


def predictive_support_decision(
    frame: pd.DataFrame,
    *,
    promoted_candidate_count: int,
) -> dict[str, object]:
    """Summarize grouped evidence across seeds without presenting one worst seed as the screen."""

    seed_summary = frame.drop_duplicates(["seed", "selection_view_id", "objective_name"])
    objective_view = (
        seed_summary.groupby(["objective_name", "selection_view_id"], sort=True)[
            ["median_within_group_spearman", "pooled_oof_spearman"]
        ]
        .median()
        .reset_index()
    )
    return {
        "verdict": "insufficient_for_policy_promotion",
        "promoted_candidate_count": int(promoted_candidate_count),
        "label_source_experiment_count": int(frame["label_source_reader_experiment_id"].nunique()),
        "minimum_rank_defined_group_count": int(frame["rank_defined_group_count"].min()),
        "seed_count": int(frame["seed"].nunique()),
        "weakest_median_within_group_spearman": _finite_min(objective_view["median_within_group_spearman"]),
        "weakest_pooled_oof_spearman": _finite_min(objective_view["pooled_oof_spearman"]),
        "claim": "retrospective_prediction_to_truth_support_not_prospective_hill_climb_efficacy",
    }


def _finite_min(series: pd.Series) -> float:
    values = series.to_numpy(dtype=float, na_value=np.nan)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("decision predictive-support evidence contains no finite correlations.")
    return float(np.min(finite))


def _allocation_overlap(frame: pd.DataFrame) -> int:
    sets = [set(rows["id"].astype(str)) for _, rows in frame.groupby("objective_name", sort=False)]
    if len(sets) != 2:
        raise ValueError("decision allocation evidence must contain exactly two objectives.")
    return len(sets[0] & sets[1])


def _digest(value: str) -> str:
    digest = str(value).removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("independent audit evidence must use a lowercase SHA-256 digest.")
    return "sha256:" + digest


__all__ = [
    "DECISION_SCHEMA_ID",
    "build_multistate_behavior_decision",
    "normalization_robustness_decision",
    "predictive_support_decision",
]
