"""
Public OPAL APIs intended for cross-package consumers.
"""

from __future__ import annotations

from .sfxi import (
    SFXI_API_VERSION,
    SFXI_REFERENCE_OVERLAY_FIELDS,
    SFXI_REFERENCE_OVERLAY_NAMESPACE,
    SFXI_REFERENCE_OVERLAY_PREFIX,
    SFXI_REFERENCE_OVERLAY_SCHEMA_VERSION,
    SFXIScoringConfig,
    SFXIScoringResult,
    score_vec8,
    to_sfxi_reference_overlay_records,
    validate_sfxi_reference_overlay_records,
)

__all__ = [
    "SFXI_API_VERSION",
    "SFXI_REFERENCE_OVERLAY_FIELDS",
    "SFXI_REFERENCE_OVERLAY_NAMESPACE",
    "SFXI_REFERENCE_OVERLAY_PREFIX",
    "SFXI_REFERENCE_OVERLAY_SCHEMA_VERSION",
    "SFXIScoringConfig",
    "SFXIScoringResult",
    "score_vec8",
    "to_sfxi_reference_overlay_records",
    "validate_sfxi_reference_overlay_records",
]
