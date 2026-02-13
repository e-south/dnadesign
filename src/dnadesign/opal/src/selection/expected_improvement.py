"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/selection/top_n.py

Module Author(s): Eric J. South, Elm Markert
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..core.round_context import roundctx_contract
from ..registries.selection import register_selection

@roundctx_contract(category="selection", requires=[], produces=[])

@register_selection("expected_improvement")
def ei(
    *,
    ids,
    scores,
    scalar_uncertainty,
    top_k: int,  # not used here; registry will use it
    objective: str = "maximize",
    tie_handling: str = "competition_rank",  # not used here; registry will use it
    alpha: float = 1,
    beta: float = 1,
    ctx=None,
    **_,
):
    """
    Minimal 'Expected Improvement' for Bayesian Optimization.
    - Primary key = score (desc if maximize, asc if minimize)
    - Secondary key = id (always ascending)
    - Non-finite scores (NaN/Inf) are pushed to the end.
    Registry fills ranks/selected.
    """

    ids = np.asarray(ids, dtype = str)
    preds = np.asarray(scores, dtype=float)
    std_devs = np.asarray(scalar_uncertainty, dtype=float)


    if ids.shape[0] != preds.shape[0]:
        raise ValueError("ids and predictions must have same length")
    
    # Converting variance to standard deviation
    std_devs = np.sqrt(std_devs)
    # Normalizing standard deviations to avoid extremely large EI values
    norm_std_devs = std_devs/np.nanmax(std_devs)

    max_sfxi = np.nanmax(preds)
    diffs = preds - max_sfxi
    z_vals = diffs / std_devs
    scores = alpha*np.multiply(diffs, norm.cdf(z_vals)) + beta*np.multiply(norm_std_devs, norm.pdf(z_vals))
    norm_scores = (scores - np.nanmin(scores))
    norm_scores = norm_scores / np.nanmax(norm_scores)
    scores = norm_scores
    
    maximize = str(objective).strip().lower().startswith("max")

    # Build a primary sort key so that np.lexsort can always sort ASC,
    # keeping id ASC as the tie-breaker regardless of objective.
    primary = np.where(np.isfinite(scores), -scores if maximize else scores, np.inf)  # sink non-finite

    # lexsort uses the *last* key as primary → (ids, primary)
    order_idx = np.lexsort((ids, preds, primary)).astype(int)

    return {"order_idx": order_idx, "score": scores}