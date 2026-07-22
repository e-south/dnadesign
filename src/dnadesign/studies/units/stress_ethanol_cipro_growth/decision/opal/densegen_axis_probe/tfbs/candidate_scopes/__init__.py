"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/candidate_scopes/__init__.py

Candidate-scope contracts for DenseGen TFBS probe campaign surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .builders import build_count_fixed_slot_position_scope, filter_labels_to_scope
from .contracts import (
    COUNT_FIXED_SLOT_POSITION_SCOPE_POLICY_ID,
    COUNT_FIXED_SLOT_POSITION_SCOPE_VALUE,
    TfbsCandidateScope,
    TfbsCandidateScopePolicy,
    count_fixed_slot_position_scope_policy,
    is_count_fixed_slot_position_label,
)

__all__ = [
    "COUNT_FIXED_SLOT_POSITION_SCOPE_POLICY_ID",
    "COUNT_FIXED_SLOT_POSITION_SCOPE_VALUE",
    "TfbsCandidateScope",
    "TfbsCandidateScopePolicy",
    "build_count_fixed_slot_position_scope",
    "count_fixed_slot_position_scope_policy",
    "filter_labels_to_scope",
    "is_count_fixed_slot_position_label",
]
