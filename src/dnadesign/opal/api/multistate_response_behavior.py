"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/api/multistate_response_behavior.py

Public Multistate Response Behavior mathematics API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..src.objectives.multistate_response_behavior_math import (
    MultistateResponseBehaviorClearances,
    MultistateResponseBehaviorScore,
    binary_target_mask,
    multistate_response_behavior_clearances,
    score_multistate_response_behavior,
    validated_response_signal,
    validated_softmin_scale,
    validated_state_ids,
)

MULTISTATE_RESPONSE_BEHAVIOR_API_VERSION = "1"

__all__ = [
    "MULTISTATE_RESPONSE_BEHAVIOR_API_VERSION",
    "MultistateResponseBehaviorClearances",
    "MultistateResponseBehaviorScore",
    "binary_target_mask",
    "multistate_response_behavior_clearances",
    "score_multistate_response_behavior",
    "validated_response_signal",
    "validated_softmin_scale",
    "validated_state_ids",
]
