"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_decision_verification.py

Fail-closed decision, audit, report, and plot checks for behavior evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..evaluation.multistate_behavior_protocol import MultistateBehaviorShadowProtocol
from ..reporting.multistate_behavior_report import render_multistate_behavior_report
from .multistate_behavior_decision import (
    DECISION_SCHEMA_ID,
    normalization_robustness_decision,
    predictive_support_decision,
)
from .multistate_behavior_record_fields import mapping, prefixed_digest, require_fields, require_literals

_DECISION_FIELDS = {
    "schema_id",
    "schema_version",
    "study_id",
    "protocol_id",
    "promotion_decision",
    "semantic_fit",
    "shadow_implementation",
    "normalization_robustness",
    "predictive_support",
    "prospective_hill_climb_efficacy",
    "technical_readiness",
    "campaign_disposition",
    "synthesis",
    "comparison_boundaries",
    "source_equivalence",
    "independent_adversarial_implementation_audit",
    "evidence",
    "claim_boundary",
}

_AUDIT_FIELDS = {
    "auditor_id",
    "completed_at",
    "schema_id",
    "schema_version",
    "status",
    "method",
    "scope",
    "findings",
    "blockers",
    "reviewed_source_commit",
    "reviewed_preliminary_manifest_sha256",
}
# Audit v1 is an immutable review receipt. A different reviewed snapshot needs
# a new audit record and verifier contract rather than in-place provenance drift.
_AUDIT_PROVENANCE = {
    "auditor_id": "codex_subagent.metric_adversarial_audit.v1",
    "completed_at": "2026-07-17T22:49:20Z",
    "reviewed_source_commit": "ee83c807d88f2f58958dc8cff78290dd90dbb826",  # pragma: allowlist secret
    "reviewed_preliminary_manifest_sha256": ("sha256:dbf9e651a467df76f12ffa22c6b15f8515264a8d515731a601fc26274c66869f"),
}


def verify_behavior_decision_artifacts(
    root: Path,
    *,
    manifest: dict[str, object],
    artifacts: dict[str, dict[str, object]],
    decision: dict[str, object],
    audit: dict[str, object],
    tables: dict[str, pd.DataFrame],
    protocol: MultistateBehaviorShadowProtocol,
) -> None:
    """Require the human-facing verdict to remain bound to machine evidence."""

    require_fields(decision, _DECISION_FIELDS, context="decision")
    require_literals(
        decision,
        {
            "schema_id": DECISION_SCHEMA_ID,
            "schema_version": "1",
            "study_id": protocol.study_id,
            "protocol_id": protocol.protocol_id,
            "promotion_decision": "no_go",
            "claim_boundary": "semantic_improvement_does_not_establish_better_predictive_hill_climbing",
        },
        context="decision",
    )
    require_literals(
        mapping(decision["semantic_fit"], context="decision.semantic_fit"),
        {
            "verdict": "go",
            "basis": "strictly_monotonic_threshold_free_three_family_behavior_matches_declared_intent",
        },
        context="semantic_fit",
        exact=True,
    )
    audit_passed = audit.get("status") == "pass" and audit.get("blockers") == []
    require_literals(
        mapping(decision["shadow_implementation"], context="shadow_implementation"),
        {
            "verdict": "go" if audit_passed else "no_go",
            "basis": (
                "pure_objective_property_tests_digest_bound_study_replay_and_independent_audit_pass"
                if audit_passed
                else "independent_adversarial_implementation_audit_has_unresolved_blockers"
            ),
        },
        context="shadow_implementation",
        exact=True,
    )
    robustness = normalization_robustness_decision(tables["normalization_sensitivity"])
    if mapping(decision["normalization_robustness"], context="normalization_robustness") != robustness:
        raise ValueError("decision normalization-robustness verdict does not derive from its evidence table.")
    promoted_count = int(tables["grouped_objective_validation"]["candidate_id"].nunique())
    predictive = predictive_support_decision(
        tables["grouped_objective_validation"],
        promoted_candidate_count=promoted_count,
    )
    if mapping(decision["predictive_support"], context="predictive_support") != predictive:
        raise ValueError("decision predictive-support verdict does not derive from grouped evidence.")
    require_literals(
        mapping(decision["prospective_hill_climb_efficacy"], context="prospective_hill_climb_efficacy"),
        {
            "verdict": "unproven",
            "required_evidence": "predictions_frozen_before_new_measurements_across_prospective_rounds",
        },
        context="prospective_hill_climb_efficacy",
        exact=True,
    )
    overlap = _allocation_overlap(tables["allocation_comparison"])
    require_literals(
        mapping(decision["campaign_disposition"], context="campaign_disposition"),
        {
            "verdict": "no_go",
            "action": "do_not_activate_or_replace_a_campaign_without_explicit_study_signoff",
            "allocation_candidate_overlap": overlap,
        },
        context="campaign_disposition",
        exact=True,
    )
    require_literals(
        mapping(decision["synthesis"], context="synthesis"), {"verdict": "prohibited"}, context="synthesis", exact=True
    )
    _verify_audit(decision, audit=audit, artifacts=artifacts)
    _verify_evidence_receipts(decision, artifacts=artifacts)
    require_literals(
        mapping(decision["technical_readiness"], context="technical_readiness"),
        {
            "verdict": "go_for_shadow_review_only",
            "corrected_reader_reference_identity_verified": True,
            "reused_central_labels_exactly_equal": True,
            "new_observation_version_required": False,
            "reason": "corrected_bundle_changes_bootstrap_reference_identity_not_central_candidate_vectors",
        },
        context="technical_readiness",
        exact=True,
    )
    require_literals(
        mapping(decision["comparison_boundaries"], context="comparison_boundaries"),
        {
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
        context="comparison_boundaries",
        exact=True,
    )
    source = mapping(decision["source_equivalence"], context="source_equivalence")
    if source.get("corrected_reader_bundle_manifest_sha256") != manifest["source"]["reader_bundle_manifest_sha256"]:
        raise ValueError("decision corrected Reader source disagrees with the manifest.")
    if source.get("prior_observation_reader_bundle_manifest_sha256") != (
        "sha256:" + protocol.source_equivalence.prior_observation_reader_bundle_sha256
    ):
        raise ValueError("decision prior observation Reader source disagrees with the protocol.")
    for field in ("central_label_equivalence_sha256", "label_artifact_sha256", "promotion_manifest_sha256"):
        prefixed_digest(source.get(field), field=f"source_equivalence.{field}")
    for field in ("reference_unit_count", "reference_bootstrap_row_count"):
        value = source.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"decision source equivalence {field!r} must be positive integer evidence.")
    if source.get("normalization_reference_unit_count") != 0:
        raise ValueError("decision source equivalence must prove that no reference unit entered normalization.")
    expected_report = render_multistate_behavior_report(
        decision,
        grouped_validation=tables["grouped_objective_validation"],
        allocation_comparison=tables["allocation_comparison"],
        hard_behavior_summary=tables["hard_behavior_summary"],
        observed_control_face_validity=tables["observed_control_face_validity"],
        family_cardinality_pressure=tables["family_cardinality_pressure"],
    )
    report_path = root / str(artifacts["report"]["path"])
    if report_path.read_text(encoding="utf-8") != expected_report:
        raise ValueError("behavior report does not derive exactly from decision and tables.")
    for artifact_id in (
        "plot__normalization_robustness",
        "plot__grouped_objective_validation",
        "plot__allocation_family_decomposition",
    ):
        path = root / str(artifacts[artifact_id]["path"])
        if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"behavior plot {artifact_id!r} is not a PNG artifact.")


def _verify_audit(
    decision: dict[str, object],
    *,
    audit: dict[str, object],
    artifacts: dict[str, dict[str, object]],
) -> None:
    verify_behavior_adversarial_audit_record(audit)
    embedded = mapping(decision["independent_adversarial_implementation_audit"], context="decision audit")
    if {key: value for key, value in embedded.items() if key != "evidence_sha256"} != audit:
        raise ValueError("decision independent audit content disagrees with the inventoried audit artifact.")
    if embedded.get("evidence_sha256") != "sha256:" + str(artifacts["independent_adversarial_audit"]["sha256"]):
        raise ValueError("decision independent audit digest disagrees with the artifact inventory.")
    implementation = mapping(decision["shadow_implementation"], context="shadow_implementation")
    expected = "go" if audit["status"] == "pass" and not audit["blockers"] else "no_go"
    if implementation.get("verdict") != expected:
        raise ValueError("shadow implementation verdict does not follow the independent audit.")


def verify_behavior_adversarial_audit_record(audit: dict[str, object]) -> None:
    """Verify the exact independent reviewer and source snapshot for audit v1."""

    require_fields(audit, _AUDIT_FIELDS, context="independent audit")
    if audit["schema_id"] != "stress_ethanol_cipro_growth.multistate_response_behavior_adversarial_audit.v1":
        raise ValueError("independent audit schema identity drifted.")
    if audit["schema_version"] != "1":
        raise ValueError("independent audit schema version drifted.")
    for field, expected in _AUDIT_PROVENANCE.items():
        if audit[field] != expected:
            raise ValueError(f"independent audit {field} disagrees with the reviewed snapshot.")
    auditor_id = audit["auditor_id"]
    if not isinstance(auditor_id, str) or not auditor_id.startswith("codex_subagent."):
        raise ValueError("independent audit auditor_id must identify the stable Codex subagent role.")
    completed_at = audit["completed_at"]
    if not isinstance(completed_at, str):
        raise ValueError("independent audit completed_at must be a UTC timestamp.")
    try:
        datetime.strptime(completed_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("independent audit completed_at must be an exact UTC timestamp.") from exc
    commit = audit["reviewed_source_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("independent audit reviewed_source_commit must be one full lowercase Git SHA.")
    prefixed_digest(
        audit["reviewed_preliminary_manifest_sha256"],
        field="independent audit reviewed_preliminary_manifest_sha256",
    )
    if audit["status"] not in {"pass", "fail"}:
        raise ValueError("independent audit status must be pass or fail.")
    for field in ("scope", "findings", "blockers"):
        if not isinstance(audit[field], list) or any(
            not isinstance(value, str) or not value.strip() for value in audit[field]
        ):
            raise ValueError(f"independent audit {field} must be a list of nonempty findings.")


def _verify_evidence_receipts(
    decision: dict[str, object],
    *,
    artifacts: dict[str, dict[str, object]],
) -> None:
    evidence = mapping(decision["evidence"], context="decision.evidence")
    expected = {
        "normalization_sensitivity",
        "grouped_objective_validation",
        "allocation_comparison",
        "observed_control_face_validity",
        "family_cardinality_pressure",
        "rmf_replay_calibration",
        "prediction_vectors",
        "prediction_scores",
        "hard_behavior_summary",
    }
    if set(evidence) != expected:
        raise ValueError("decision evidence receipt identities drifted.")
    for table_id, receipt in evidence.items():
        if receipt != artifacts[f"table__{table_id}"]:
            raise ValueError(f"decision evidence receipt {table_id!r} disagrees with the manifest.")


def _allocation_overlap(frame: pd.DataFrame) -> int:
    candidate_sets = [set(rows["id"].astype(str)) for _, rows in frame.groupby("objective_name", sort=False)]
    if len(candidate_sets) != 2:
        raise ValueError("decision allocation evidence must contain exactly two objectives.")
    return len(candidate_sets[0] & candidate_sets[1])


__all__ = ["verify_behavior_adversarial_audit_record", "verify_behavior_decision_artifacts"]
