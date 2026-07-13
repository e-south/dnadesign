"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/objectives/response_magnitude_feasibility_v1.py

Non-compensatory Response-Magnitude Feasibility objective.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..core.objective_result import ObjectiveResultV2
from ..core.round_context import PluginCtx, roundctx_contract
from ..registries.objectives import register_objective
from .response_magnitude_feasibility_math import (
    OBJECTIVE_NAME,
    ResponseMagnitudeFeasibilityComponents,
    response_magnitude_feasibility_components,
    score_response_magnitude_feasibility,
)


@roundctx_contract(category="objective", requires=["core/labels_as_of_round"], produces=[])
@register_objective(OBJECTIVE_NAME)
def response_magnitude_feasibility_v1(
    *,
    y_pred: np.ndarray,
    params: Dict[str, Any],
    ctx: Optional[PluginCtx],
    train_view=None,
    y_pred_std=None,
) -> ObjectiveResultV2:
    """Score aligned response/magnitude states with maximin constraints."""

    del ctx, train_view, y_pred_std
    if not isinstance(params, Mapping):
        raise ValueError(f"{OBJECTIVE_NAME}: params must be a mapping.")
    allowed = {"state_ids", "target_mask", "calibration"}
    extra = sorted(set(params) - allowed)
    if extra:
        raise ValueError(f"{OBJECTIVE_NAME}: unknown params: {extra}.")
    missing = sorted(allowed - set(params))
    if missing:
        raise ValueError(f"{OBJECTIVE_NAME}: missing required params: {missing}.")

    state_ids = _validated_state_ids(params["state_ids"])
    target_mask = list(params["target_mask"])
    if len(target_mask) != len(state_ids):
        raise ValueError(
            f"{OBJECTIVE_NAME}: state_ids and target_mask must have equal length; "
            f"got {len(state_ids)} and {len(target_mask)}."
        )
    scored = score_response_magnitude_feasibility(
        y_pred,
        target_mask=target_mask,
        calibration=params["calibration"],
    )
    components = scored.components
    feasibility_margin = scored.feasibility_margin

    scores = {
        "feasibility_margin": np.asarray(feasibility_margin, dtype=float),
        "response_separation": np.asarray(components.response_separation, dtype=float),
        "on_magnitude_floor": np.asarray(components.on_magnitude_floor, dtype=float),
        "off_magnitude_ceiling": np.asarray(components.off_magnitude_ceiling, dtype=float),
    }
    diagnostics = {
        "state_ids": state_ids,
        "target_mask": [int(value) for value in target_mask],
        "calibration": scored.calibration,
        "response_constraint_margin": scored.response_constraint_margin,
        "on_magnitude_constraint_margin": scored.on_magnitude_constraint_margin,
        "off_magnitude_constraint_margin": scored.off_magnitude_constraint_margin,
        "feasible": np.asarray(feasibility_margin >= 0.0, dtype=bool),
        "uncertainty_emitted": False,
        "summary_stats": {
            "feasible_count": int(np.sum(feasibility_margin >= 0.0)),
            "candidate_count": int(feasibility_margin.size),
            "feasibility_margin_min": float(np.min(feasibility_margin)),
            "feasibility_margin_median": float(np.median(feasibility_margin)),
            "feasibility_margin_max": float(np.max(feasibility_margin)),
        },
    }
    return ObjectiveResultV2(
        scores_by_name=scores,
        uncertainty_by_name={},
        diagnostics=diagnostics,
        modes_by_name={
            "feasibility_margin": "maximize",
            "response_separation": "maximize",
            "on_magnitude_floor": "maximize",
            "off_magnitude_ceiling": "minimize",
        },
    )


def _validated_state_ids(raw: object) -> list[str]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError(f"{OBJECTIVE_NAME}: state_ids must contain at least two ordered strings.")
    values = [str(value).strip() for value in raw]
    if any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{OBJECTIVE_NAME}: state_ids must be non-empty and unique; got {values}.")
    return values


response_magnitude_feasibility_v1.__opal_score_channels__ = (
    "feasibility_margin",
    "response_separation",
    "on_magnitude_floor",
    "off_magnitude_ceiling",
)
response_magnitude_feasibility_v1.__opal_uncertainty_channels__ = ()
response_magnitude_feasibility_v1.__opal_score_modes__ = {
    "feasibility_margin": "maximize",
    "response_separation": "maximize",
    "on_magnitude_floor": "maximize",
    "off_magnitude_ceiling": "minimize",
}


__all__ = [
    "OBJECTIVE_NAME",
    "ResponseMagnitudeFeasibilityComponents",
    "response_magnitude_feasibility_components",
    "response_magnitude_feasibility_v1",
]
