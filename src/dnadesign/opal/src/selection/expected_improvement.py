"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/selection/expected_improvement.py

Ranks candidates with expected-improvement acquisition using explicit
score/uncertainty channel inputs.

Module Author(s): Elm Markert
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..core.round_context import roundctx_contract
from ..registries.selection import register_selection


def _minmax_01(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("[expected_improvement] non-finite acquisition; check scores/uncertainty.")
    if hi <= lo:
        return np.zeros_like(arr)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _validate_uncertainty_std(ids: np.ndarray, scores: np.ndarray, scalar_uncertainty) -> np.ndarray:
    if scalar_uncertainty is None:
        raise ValueError("[expected_improvement] scalar_uncertainty is required.")
    unc_std = np.asarray(scalar_uncertainty, dtype=float).reshape(-1)
    if unc_std.size != ids.size:
        raise ValueError(
            f"[expected_improvement] uncertainty length mismatch: got {unc_std.size}, expected {ids.size}."
        )
    if not np.all(np.isfinite(unc_std)):
        raise ValueError("[expected_improvement] uncertainty must be finite.")
    if np.any(unc_std <= 0.0):
        raise ValueError("[expected_improvement] uncertainty must be > 0 for all candidates.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("[expected_improvement] scores must be finite.")
    return unc_std


@roundctx_contract(category="selection", requires=[], produces=[])
@register_selection("expected_improvement")
def ei(
    *,
    ids,
    scores,
    scalar_uncertainty,
    top_k: int,
    objective: str,
    tie_handling: str,
    alpha: float = 1.0,
    beta: float = 1.0,
    ctx=None,
    **_,
):
    del top_k, tie_handling, ctx
    ids = np.asarray(ids, dtype=str).reshape(-1)
    preds = np.asarray(scores, dtype=float).reshape(-1)
    if ids.size != preds.size:
        raise ValueError("[expected_improvement] ids and scores must have the same length.")
    uncertainty_std = _validate_uncertainty_std(ids, preds, scalar_uncertainty)

    mode = str(objective).strip().lower()
    if mode not in {"maximize", "minimize"}:
        raise ValueError("[expected_improvement] objective must be 'maximize' or 'minimize'.")
    alpha = float(alpha)
    beta = float(beta)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("[expected_improvement] alpha must be finite and >= 0.")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("[expected_improvement] beta must be finite and >= 0.")

    if mode == "maximize":
        incumbent = float(np.max(preds))
        improvement = preds - incumbent
    else:
        incumbent = float(np.min(preds))
        improvement = incumbent - preds

    z = improvement / uncertainty_std
    exploit = improvement * norm.cdf(z)
    uncertainty_std_explore = _minmax_01(uncertainty_std)
    explore = uncertainty_std_explore * norm.pdf(z)
    acquisition = alpha * exploit + beta * explore
    if not np.all(np.isfinite(acquisition)):
        raise ValueError("[expected_improvement] non-finite acquisition; check scores/uncertainty.")
    acquisition_norm = _minmax_01(acquisition)

    primary = np.where(np.isfinite(acquisition_norm), -acquisition_norm, np.inf)
    order_idx = np.lexsort((ids, primary)).astype(int)
    return {"order_idx": order_idx, "score": acquisition_norm}
