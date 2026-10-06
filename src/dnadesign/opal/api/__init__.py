"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/api/__init__.py

Public OPAL APIs intended for cross-package consumers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_EXPORTS = {
    "MULTISTATE_RESPONSE_BEHAVIOR_API_VERSION": ".multistate_response_behavior",
    "OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION": ".observed_labels",
    "OBSERVED_LABELS_API_VERSION": ".observed_labels",
    "OBSERVED_OBJECTIVE_HISTORY_API_VERSION": ".observed_objective_history",
    "READER_EVIDENCE_API_VERSION": ".reader_evidence",
    "READER_EVIDENCE_ARTIFACT_ENTRY_POINT": ".reader_evidence",
    "READER_EVIDENCE_MANIFEST_ADAPTER": ".reader_evidence",
    "ReaderEvidenceArtifactAdapter": ".reader_evidence",
    "ReaderEvidenceManifestAdapterError": ".reader_evidence",
    "ReaderEvidenceManifestProjection": ".reader_evidence",
    "CandidateExclusionSetBinding": ".observed_labels",
    "ObservedLabelPromotionBinding": ".observed_labels",
    "ObservedLabelVerificationError": ".observed_labels",
    "MultistateResponseBehaviorClearances": ".multistate_response_behavior",
    "MultistateResponseBehaviorScore": ".multistate_response_behavior",
    "RESPONSE_MAGNITUDE_FEASIBILITY_API_VERSION": ".response_magnitude_feasibility",
    "ResponseMagnitudeFeasibilityComponents": ".response_magnitude_feasibility",
    "ResponseMagnitudeFeasibilityScore": ".response_magnitude_feasibility",
    "RUN_SERIES_SCHEMA_VERSION": ".observed_objective_history",
    "SELECTION_ALLOCATION_PREVIEW_API_VERSION": ".selection_allocation",
    "SELECTION_VIEW_PERFORMANCE_API_VERSION": ".selection_view_performance",
    "SFXI_API_VERSION": ".sfxi",
    "SFXI_REFERENCE_OVERLAY_FIELDS": ".sfxi",
    "SFXI_REFERENCE_OVERLAY_NAMESPACE": ".sfxi",
    "SFXI_REFERENCE_OVERLAY_PREFIX": ".sfxi",
    "SFXI_REFERENCE_OVERLAY_SCHEMA_VERSION": ".sfxi",
    "SFXI_STATE_ORDER": ".sfxi",
    "SFXIScoringConfig": ".sfxi",
    "SFXIScoringResult": ".sfxi",
    "SelectionAllocationPreview": ".selection_allocation",
    "SelectionViewPerformance": ".selection_view_performance",
    "VerifiedObservedLabelPromotion": ".observed_labels",
    "VerifiedObservedLabelSnapshot": ".observed_labels",
    "binary_target_mask": ".response_magnitude_feasibility",
    "build_candidate_exclusion_projection": ".observed_labels",
    "candidate_exclusion_sets_from_config": ".observed_labels",
    "candidate_snapshot_record": ".observed_labels",
    "calibrate_response_magnitude_feasibility": ".response_magnitude_feasibility",
    "optional_reader_evidence_artifact_adapter": ".reader_evidence",
    "parse_reader_evidence_manifest_adapter": ".reader_evidence",
    "reader_evidence_artifact_adapter": ".reader_evidence",
    "register_reader_evidence_artifact_adapter": ".reader_evidence",
    "observed_objective_run_contract_sha256": ".observed_objective_history",
    "preview_round_robin_next_best_unallocated": ".selection_allocation",
    "render_selection_view_performance": ".selection_view_performance",
    "multistate_response_behavior_clearances": ".multistate_response_behavior",
    "response_magnitude_feasibility_components": ".response_magnitude_feasibility",
    "score_response_magnitude_feasibility": ".response_magnitude_feasibility",
    "score_multistate_response_behavior": ".multistate_response_behavior",
    "score_vec8": ".sfxi",
    "score_vec8_with_denom": ".sfxi",
    "selection_view_performance": ".selection_view_performance",
    "to_sfxi_reference_overlay_records": ".sfxi",
    "validate_sfxi_reference_overlay_records": ".sfxi",
    "validated_response_magnitude": ".response_magnitude_feasibility",
    "verify_observed_label_snapshot": ".observed_labels",
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _PUBLIC_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))
