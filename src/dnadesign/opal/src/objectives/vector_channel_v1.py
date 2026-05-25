"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/objectives/vector_channel_v1.py

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
@register_objective("vector_channel_v1")
def vector_channel_v1(
    *,
    y_pred: np.ndarray,
    params: Dict[str, Any],
    ctx: Optional[PluginCtx],
    train_view,
    y_pred_std,
) -> ObjectiveResultV2:
    """Emit one selectable score channel from a finite numeric prediction vector."""

    del ctx, train_view, y_pred_std
    if not (isinstance(y_pred, np.ndarray) and y_pred.ndim == 2):
        raise ValueError(f"[vector_channel_v1] Expected y_pred shape (n, k); got {getattr(y_pred, 'shape', None)}.")
    if y_pred.shape[1] < 1:
        raise ValueError("[vector_channel_v1] Expected y_pred with at least one channel.")

    channel_index = int((params or {}).get("channel_index", 0))
    if channel_index < 0 or channel_index >= y_pred.shape[1]:
        raise ValueError(
            f"[vector_channel_v1] channel_index {channel_index} is out of bounds for y_pred width {y_pred.shape[1]}."
        )
    channel_name = str((params or {}).get("channel_name", f"channel_{channel_index}")).strip()
    if not channel_name:
        raise ValueError("[vector_channel_v1] channel_name must be non-empty.")
    mode = str((params or {}).get("mode", "maximize")).strip().lower()
    if mode not in {"maximize", "minimize"}:
        raise ValueError("[vector_channel_v1] mode must be 'maximize' or 'minimize'.")

    scores = np.asarray(y_pred[:, channel_index], dtype=float).ravel()
    if not np.all(np.isfinite(scores)):
        raise ValueError("[vector_channel_v1] selected prediction channel contains non-finite values.")
    diagnostics = {
        "channel_index": channel_index,
        "channel_name": channel_name,
        "mode": mode,
        "summary_stats": {
            "score_min": float(np.nanmin(scores)) if scores.size else float("nan"),
            "score_median": float(np.nanmedian(scores)) if scores.size else float("nan"),
            "score_max": float(np.nanmax(scores)) if scores.size else float("nan"),
        },
    }
    return ObjectiveResultV2(
        scores_by_name={channel_name: scores},
        uncertainty_by_name={},
        diagnostics=diagnostics,
        modes_by_name={channel_name: mode},
    )
