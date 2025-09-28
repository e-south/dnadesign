"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/selection/top_n.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from ..registries.selections import register_selection


@register_selection("top_n")
def top_n(
    *,
    ids,
    scores,
    top_k: int,  # not used here; registry will use it
    objective: str = "maximize",
    tie_handling: str = "competition_rank",  # not used here; registry will use it
    **_,
):
    """
    Minimal 'top_n' that emits a deterministic best-first order.
    - Primary key = score (desc if maximize, asc if minimize)
    - Secondary key = id (always ascending)
    - Non-finite scores (NaN/Inf) are pushed to the end.
    Registry fills ranks/selected.
    """
    ids = np.asarray(ids, dtype=str)
    scores = np.asarray(scores, dtype=float)
    if ids.shape[0] != scores.shape[0]:
        raise ValueError("ids and scores must have same length")

    maximize = str(objective).strip().lower().startswith("max")

    # Build a primary sort key so that np.lexsort can always sort ASC,
    # keeping id ASC as the tie-breaker regardless of objective.
    primary = np.where(
        np.isfinite(scores), -scores if maximize else scores, np.inf
    )  # sink non-finite

    # lexsort uses the *last* key as primary → (ids, primary)
    order_idx = np.lexsort((ids, primary)).astype(int)

    return {"order_idx": order_idx}
