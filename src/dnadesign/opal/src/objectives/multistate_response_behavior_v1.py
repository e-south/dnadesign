"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/objectives/multistate_response_behavior_v1.py

Threshold-free Multistate Response Behavior objective plugin.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..core.objective_result import ObjectiveResultV2
from ..core.round_context import PluginCtx, roundctx_contract
from ..registries.objectives import register_objective
from .multistate_response_behavior_math import (
    OBJECTIVE_NAME,
    MultistateResponseBehaviorClearances,
    MultistateResponseBehaviorScore,
    multistate_response_behavior_clearances,
    score_multistate_response_behavior,
)

SCORE_CHANNELS = ("behavior_score",)
DIAGNOSTIC_CHANNELS = (
    "hard_bottleneck_clearance",
    "compensation_gap",
    "maximum_compensation_gap",
    "response_family_score",
    "on_signal_family_score",
    "off_signal_suppression_family_score",
)


@roundctx_contract(category="objective", requires=["core/labels_as_of_round"], produces=[])
@register_objective(OBJECTIVE_NAME, family="multistate_response_behavior")
def multistate_response_behavior_v1(
    *,
    y_pred: np.ndarray,
    params: Dict[str, Any],
    ctx: Optional[PluginCtx],
    train_view=None,
    y_pred_std=None,
) -> ObjectiveResultV2:
    """Score aligned response and reference-relative signal states with a smooth bottleneck."""

    del ctx, train_view, y_pred_std
    if not isinstance(params, Mapping):
        raise ValueError(f"{OBJECTIVE_NAME}: params must be a mapping.")
    required = {"state_ids", "target_mask", "normalization"}
    missing = sorted(required - set(params))
    extra = sorted(set(params) - required)
    if missing or extra:
        raise ValueError(f"{OBJECTIVE_NAME}: params do not match the contract; missing={missing}, extra={extra}.")

    scored = score_multistate_response_behavior(
        y_pred,
        state_ids=params["state_ids"],
        target_mask=params["target_mask"],
        normalization=params["normalization"],
    )
    scores = {"behavior_score": np.asarray(scored.behavior_score, dtype=float)}
    limiting_bottleneck_weight = scored.coordinate_weights[
        np.arange(len(scored.behavior_score)),
        scored.limiting_coordinate_index,
    ]
    limiting_prior_weight = scored.coordinate_prior_weights[scored.limiting_coordinate_index]
    diagnostics = {
        **{channel: np.asarray(getattr(scored, channel), dtype=float) for channel in DIAGNOSTIC_CHANNELS},
        "all_reference_directions_met": scored.all_reference_directions_met.astype(np.int8),
        "limiting_coordinate_index": scored.limiting_coordinate_index,
        "limiting_coordinate_prior_weight": limiting_prior_weight,
        "limiting_coordinate_bottleneck_weight": limiting_bottleneck_weight,
        "uncertainty_emitted": False,
        "summary_stats": {
            "candidate_count": int(scored.behavior_score.size),
            "all_reference_directions_met_count": int(np.sum(scored.all_reference_directions_met)),
            "behavior_score_min": float(np.min(scored.behavior_score)),
            "behavior_score_median": float(np.median(scored.behavior_score)),
            "behavior_score_max": float(np.max(scored.behavior_score)),
        },
    }
    return ObjectiveResultV2(
        scores_by_name=scores,
        uncertainty_by_name={},
        diagnostics=diagnostics,
        modes_by_name={channel: "maximize" for channel in SCORE_CHANNELS},
    )


multistate_response_behavior_v1.__opal_score_channels__ = SCORE_CHANNELS
multistate_response_behavior_v1.__opal_uncertainty_channels__ = ()
multistate_response_behavior_v1.__opal_score_modes__ = {channel: "maximize" for channel in SCORE_CHANNELS}
multistate_response_behavior_v1.__opal_observed_replay_contract__ = "pointwise_params_v1"


__all__ = [
    "DIAGNOSTIC_CHANNELS",
    "OBJECTIVE_NAME",
    "SCORE_CHANNELS",
    "MultistateResponseBehaviorClearances",
    "MultistateResponseBehaviorScore",
    "multistate_response_behavior_clearances",
    "multistate_response_behavior_v1",
    "score_multistate_response_behavior",
]
