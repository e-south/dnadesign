"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/api/__init__.py

Public OPAL APIs intended for cross-package consumers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

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
    "OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION",
    "OBSERVED_LABELS_API_VERSION",
    "ObservedLabelPromotionBinding",
    "ObservedLabelVerificationError",
    "RESPONSE_MAGNITUDE_FEASIBILITY_API_VERSION",
    "ResponseMagnitudeFeasibilityComponents",
    "ResponseMagnitudeFeasibilityScore",
    "SFXI_API_VERSION",
    "SFXI_REFERENCE_OVERLAY_FIELDS",
    "SFXI_REFERENCE_OVERLAY_NAMESPACE",
    "SFXI_REFERENCE_OVERLAY_PREFIX",
    "SFXI_REFERENCE_OVERLAY_SCHEMA_VERSION",
    "SFXI_STATE_ORDER",
    "SFXIScoringConfig",
    "SFXIScoringResult",
    "VerifiedObservedLabelPromotion",
    "VerifiedObservedLabelSnapshot",
    "binary_target_mask",
    "candidate_snapshot_record",
    "calibrate_response_magnitude_feasibility",
    "response_magnitude_feasibility_components",
    "score_response_magnitude_feasibility",
    "score_vec8",
    "score_vec8_with_denom",
    "to_sfxi_reference_overlay_records",
    "validate_sfxi_reference_overlay_records",
    "validated_response_magnitude",
    "verify_observed_label_snapshot",
]
