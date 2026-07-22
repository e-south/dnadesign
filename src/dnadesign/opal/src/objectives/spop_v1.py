"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/objectives/spop_v1.py

Objective plugin logic for spop v1 OPAL objectives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core.objective_result import ObjectiveResultV2
from ..core.round_context import PluginCtx, roundctx_contract
from ..registries.objectives import register_objective

SPOP_OBJECTIVE_NAME = "spop_v1"
SPOP_SCORE_CHANNEL = "spop"
SPOP_READER_METRIC_ID = "reader_spop_endpoint_dose_mean_v1"
SPOP_NUMERIC_SCOPE = "reader_experiment_normalized_tf_sponging"


@roundctx_contract(
    category="objective",
    requires=["core/labels_as_of_round"],
    produces=[],
)
@register_objective(SPOP_OBJECTIVE_NAME)
def spop_v1(
    *,
    y_pred: np.ndarray,
    params: Dict[str, Any],
    ctx: Optional[PluginCtx],
    train_view,
    y_pred_std,
) -> ObjectiveResultV2:
    """Select on a predicted Reader SPOP endpoint scalar."""

    del ctx, train_view, y_pred_std
    if params:
        extra = sorted(str(key) for key in params)
        raise ValueError(f"[{SPOP_OBJECTIVE_NAME}] params must be empty; got {extra}.")
    if not (isinstance(y_pred, np.ndarray) and y_pred.ndim == 2):
        raise ValueError(f"[{SPOP_OBJECTIVE_NAME}] Expected y_pred shape (n, 1); got {getattr(y_pred, 'shape', None)}.")
    if y_pred.shape[1] != 1:
        raise ValueError(f"[{SPOP_OBJECTIVE_NAME}] Expected y_pred with 1 column; got {y_pred.shape[1]}.")

    scores = np.asarray(y_pred[:, 0], dtype=float).ravel()
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"[{SPOP_OBJECTIVE_NAME}] selected prediction channel contains non-finite values.")
    diagnostics = {
        "metric_id": SPOP_READER_METRIC_ID,
        "numeric_scope": SPOP_NUMERIC_SCOPE,
        "score_channel": SPOP_SCORE_CHANNEL,
        "negative_prediction_count": int(np.sum(scores < 0.0)),
        "summary_stats": {
            "score_min": float(np.nanmin(scores)) if scores.size else float("nan"),
            "score_median": float(np.nanmedian(scores)) if scores.size else float("nan"),
            "score_max": float(np.nanmax(scores)) if scores.size else float("nan"),
        },
    }
    return ObjectiveResultV2(
        scores_by_name={SPOP_SCORE_CHANNEL: scores},
        uncertainty_by_name={},
        diagnostics=diagnostics,
        modes_by_name={SPOP_SCORE_CHANNEL: "maximize"},
    )


spop_v1.__opal_score_channels__ = (SPOP_SCORE_CHANNEL,)
spop_v1.__opal_uncertainty_channels__ = ()
spop_v1.__opal_score_modes__ = {SPOP_SCORE_CHANNEL: "maximize"}


__all__ = [
    "SPOP_NUMERIC_SCOPE",
    "SPOP_OBJECTIVE_NAME",
    "SPOP_READER_METRIC_ID",
    "SPOP_SCORE_CHANNEL",
    "spop_v1",
]
