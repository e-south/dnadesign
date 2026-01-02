"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/objectives/scalar_identity_v1.py

Identity objective for scalar Y. Produces score = y_pred[:, 0].

Module Author(s): Eric J. South (extended by Codex)
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core.round_context import PluginCtx, roundctx_contract
from ..registries.objectives import register_objective


class ObjectiveResult:
    def __init__(
        self,
        *,
        score: np.ndarray,
        scalar_uncertainty: Optional[np.ndarray],
        diagnostics: Dict[str, Any],
    ) -> None:
        self.score = score
        self.scalar_uncertainty = scalar_uncertainty
        self.diagnostics = diagnostics


@roundctx_contract(
    category="objective",
    requires=["core/labels_as_of_round"],
    produces=[],
)
@register_objective("scalar_identity_v1")
def scalar_identity_v1(
    *,
    y_pred: np.ndarray,
    params: Dict[str, Any],
    ctx: Optional[PluginCtx] = None,
    train_view=None,
) -> ObjectiveResult:
    if not (isinstance(y_pred, np.ndarray) and y_pred.ndim == 2):
        raise ValueError(f"[scalar_identity_v1] Expected y_pred shape (n, 1); got {getattr(y_pred, 'shape', None)}.")
    if y_pred.shape[1] != 1:
        raise ValueError(f"[scalar_identity_v1] Expected y_pred with 1 column; got {y_pred.shape[1]}.")

    scores = np.asarray(y_pred[:, 0], dtype=float).ravel()
    diagnostics = {
        "summary_stats": {
            "score_min": float(np.nanmin(scores)) if scores.size else float("nan"),
            "score_median": float(np.nanmedian(scores)) if scores.size else float("nan"),
            "score_max": float(np.nanmax(scores)) if scores.size else float("nan"),
        },
    }
    return ObjectiveResult(score=scores, scalar_uncertainty=None, diagnostics=diagnostics)
