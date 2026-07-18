"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/api/__init__.py

Public OPAL APIs intended for cross-package consumers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .multistate_response_behavior import (
    MULTISTATE_RESPONSE_BEHAVIOR_API_VERSION,
    MultistateResponseBehaviorClearances,
    MultistateResponseBehaviorScore,
    multistate_response_behavior_clearances,
    score_multistate_response_behavior,
)
from .observed_labels import (
    OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION,
    OBSERVED_LABELS_API_VERSION,
    ObservedLabelPromotionBinding,
    ObservedLabelVerificationError,
    VerifiedObservedLabelPromotion,
    VerifiedObservedLabelSnapshot,
    candidate_snapshot_record,
    verify_observed_label_snapshot,
)
from .observed_objective_history import (
    OBSERVED_OBJECTIVE_HISTORY_API_VERSION,
    RUN_SERIES_SCHEMA_VERSION,
    observed_objective_run_contract_sha256,
)
from .reader_evidence import (
    READER_EVIDENCE_API_VERSION,
    READER_EVIDENCE_MANIFEST_ADAPTER,
    ReaderEvidenceManifestAdapterError,
    ReaderEvidenceManifestProjection,
    parse_reader_evidence_manifest_adapter,
)
from .response_magnitude_feasibility import (
    RESPONSE_MAGNITUDE_FEASIBILITY_API_VERSION,
    ResponseMagnitudeFeasibilityComponents,
    ResponseMagnitudeFeasibilityScore,
    binary_target_mask,
    calibrate_response_magnitude_feasibility,
    response_magnitude_feasibility_components,
    score_response_magnitude_feasibility,
    validated_response_magnitude,
)
from .selection_allocation import (
    SELECTION_ALLOCATION_PREVIEW_API_VERSION,
    SelectionAllocationPreview,
    preview_round_robin_next_best_unallocated,
)
from .sfxi import (
    SFXI_API_VERSION,
    SFXI_REFERENCE_OVERLAY_FIELDS,
    SFXI_REFERENCE_OVERLAY_NAMESPACE,
    SFXI_REFERENCE_OVERLAY_PREFIX,
    SFXI_REFERENCE_OVERLAY_SCHEMA_VERSION,
    SFXI_STATE_ORDER,
    SFXIScoringConfig,
    SFXIScoringResult,
    score_vec8,
    score_vec8_with_denom,
    to_sfxi_reference_overlay_records,
    validate_sfxi_reference_overlay_records,
)

__all__ = [
    "MULTISTATE_RESPONSE_BEHAVIOR_API_VERSION",
    "OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION",
    "OBSERVED_LABELS_API_VERSION",
    "OBSERVED_OBJECTIVE_HISTORY_API_VERSION",
    "READER_EVIDENCE_API_VERSION",
    "READER_EVIDENCE_MANIFEST_ADAPTER",
    "ReaderEvidenceManifestAdapterError",
    "ReaderEvidenceManifestProjection",
    "ObservedLabelPromotionBinding",
    "ObservedLabelVerificationError",
    "MultistateResponseBehaviorClearances",
    "MultistateResponseBehaviorScore",
    "RESPONSE_MAGNITUDE_FEASIBILITY_API_VERSION",
    "ResponseMagnitudeFeasibilityComponents",
    "ResponseMagnitudeFeasibilityScore",
    "RUN_SERIES_SCHEMA_VERSION",
    "SELECTION_ALLOCATION_PREVIEW_API_VERSION",
    "SFXI_API_VERSION",
    "SFXI_REFERENCE_OVERLAY_FIELDS",
    "SFXI_REFERENCE_OVERLAY_NAMESPACE",
    "SFXI_REFERENCE_OVERLAY_PREFIX",
    "SFXI_REFERENCE_OVERLAY_SCHEMA_VERSION",
    "SFXI_STATE_ORDER",
    "SFXIScoringConfig",
    "SFXIScoringResult",
    "SelectionAllocationPreview",
    "VerifiedObservedLabelPromotion",
    "VerifiedObservedLabelSnapshot",
    "binary_target_mask",
    "candidate_snapshot_record",
    "calibrate_response_magnitude_feasibility",
    "parse_reader_evidence_manifest_adapter",
    "observed_objective_run_contract_sha256",
    "preview_round_robin_next_best_unallocated",
    "multistate_response_behavior_clearances",
    "response_magnitude_feasibility_components",
    "score_response_magnitude_feasibility",
    "score_multistate_response_behavior",
    "score_vec8",
    "score_vec8_with_denom",
    "to_sfxi_reference_overlay_records",
    "validate_sfxi_reference_overlay_records",
    "validated_response_magnitude",
    "verify_observed_label_snapshot",
]
