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
    if np.any(unc_std < 0.0):
        raise ValueError("[expected_improvement] uncertainty must be non-negative.")
    if not np.any(unc_std > 0.0):
        raise ValueError("[expected_improvement] uncertainty cannot be all zeros.")
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

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.divide(
            improvement,
            uncertainty_std,
            out=np.zeros_like(improvement, dtype=float),
            where=uncertainty_std > 0.0,
        )
    exploit = improvement * norm.cdf(z)
    explore = uncertainty_std * norm.pdf(z)
    acquisition_pos = alpha * exploit + beta * explore
    # Deterministic EI limit: sigma -> 0 gives max(improvement, 0), scaled by alpha.
    acquisition_zero = alpha * np.maximum(improvement, 0.0)
    acquisition = np.where(uncertainty_std > 0.0, acquisition_pos, acquisition_zero)
    if not np.all(np.isfinite(acquisition)):
        raise ValueError("[expected_improvement] non-finite acquisition; check scores/uncertainty.")

    primary = np.where(np.isfinite(acquisition), -acquisition, np.inf)
    order_idx = np.lexsort((ids, primary)).astype(int)
    return {"order_idx": order_idx, "score": acquisition}
