"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/objectives/vector_target_similarity_v1.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core.objective_result import ObjectiveResultV2
from ..core.round_context import PluginCtx, roundctx_contract
from ..registries.objectives import register_objective


@roundctx_contract(
    category="objective",
    requires=["core/labels_as_of_round"],
    produces=[],
)
@register_objective("vector_target_similarity_v1")
def vector_target_similarity_v1(
    *,
    y_pred: np.ndarray,
    params: Dict[str, Any],
    ctx: Optional[PluginCtx],
    train_view,
    y_pred_std,
) -> ObjectiveResultV2:
    """Score finite vector predictions by closeness to a declared target vector."""

    del ctx, train_view, y_pred_std
    if not (isinstance(y_pred, np.ndarray) and y_pred.ndim == 2):
        raise ValueError(
            f"[vector_target_similarity_v1] Expected y_pred shape (n, k); got {getattr(y_pred, 'shape', None)}."
        )
    if y_pred.shape[1] < 1:
        raise ValueError("[vector_target_similarity_v1] Expected y_pred with at least one channel.")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("[vector_target_similarity_v1] y_pred contains non-finite values.")

    target = np.asarray((params or {}).get("target_vector", []), dtype=float).ravel()
    if target.size != y_pred.shape[1]:
        raise ValueError(
            "[vector_target_similarity_v1] target_vector length "
            f"{target.size} does not match y_pred width {y_pred.shape[1]}."
        )
    if not np.all(np.isfinite(target)):
        raise ValueError("[vector_target_similarity_v1] target_vector contains non-finite values.")

    mse = np.mean((np.asarray(y_pred, dtype=float) - target[None, :]) ** 2, axis=1)
    if not np.all(np.isfinite(mse)):
        raise ValueError("[vector_target_similarity_v1] computed MSE contains non-finite values.")
    scores = -mse
    diagnostics = {
        "target_vector": target.astype(float).tolist(),
        "metric": "negative_mse",
        "summary_stats": {
            "negative_mse_min": float(np.nanmin(scores)) if scores.size else float("nan"),
            "negative_mse_median": float(np.nanmedian(scores)) if scores.size else float("nan"),
            "negative_mse_max": float(np.nanmax(scores)) if scores.size else float("nan"),
            "mse_min": float(np.nanmin(mse)) if mse.size else float("nan"),
            "mse_median": float(np.nanmedian(mse)) if mse.size else float("nan"),
            "mse_max": float(np.nanmax(mse)) if mse.size else float("nan"),
        },
    }
    return ObjectiveResultV2(
        scores_by_name={"negative_mse": scores},
        uncertainty_by_name={},
        diagnostics=diagnostics,
        modes_by_name={"negative_mse": "maximize"},
    )


vector_target_similarity_v1.__opal_score_channels__ = ("negative_mse",)
vector_target_similarity_v1.__opal_uncertainty_channels__ = ()
vector_target_similarity_v1.__opal_score_modes__ = {"negative_mse": "maximize"}
