"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/objectives/logic_plus_effect_v1.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

# src/dnadesign/opal/src/objectives/logic_plus_effect_v1.py
from __future__ import annotations

from typing import Any, Dict, NamedTuple

import numpy as np

from ...registries.objectives import register_objective
from ..exceptions import OpalError


class ObjectiveResult(NamedTuple):
    score: np.ndarray
    logic_fidelity: np.ndarray
    effect_scaled: np.ndarray
    denom_used: float


def _logic_fidelity_l2_norm01(v: np.ndarray, s: np.ndarray) -> np.ndarray:
    # v in [0,1]^4, s in [0,1]^4
    dmax = np.maximum(s, 1.0 - s)
    D = float(np.sqrt((dmax**2).sum()))
    if D == 0.0:
        return np.ones(v.shape[0], dtype=float)
    d = np.linalg.norm(v - s.reshape(1, -1), axis=1) / D
    return np.clip(1.0 - d, 0.0, 1.0)


def _effect_scale_per_round(
    e_pred: np.ndarray,
    e_labels: np.ndarray,
    p: int,
    min_n: int,
    fallback_p: int,
    eps: float,
) -> tuple[np.ndarray, float]:
    e_labels = e_labels[np.isfinite(e_labels)]
    if e_labels.size >= min_n:
        denom = float(np.percentile(e_labels, p))
    else:
        # conservative fallback; avoid zero denom
        p_main = np.percentile(e_labels, p) if e_labels.size else 0.0
        p_fb = np.percentile(e_labels, fallback_p) if e_labels.size else 0.0
        denom = float(max(p_main, p_fb, eps))
    scaled = np.clip(e_pred / max(denom, eps), 0.0, 1.0)
    return scaled, denom


@register_objective("logic_plus_effect_v1")
def objective_logic_plus_effect_v1(
    *,
    y_pred_vec5: np.ndarray,  # shape (n,5)
    params: Dict[str, Any],
    e_labels_for_scaling: np.ndarray,  # 1-D
) -> ObjectiveResult:
    if y_pred_vec5.ndim != 2 or y_pred_vec5.shape[1] != 5:
        raise OpalError(
            f"logic_plus_effect_v1 expects predictions of shape (n,5), got {y_pred_vec5.shape}"
        )

    v = y_pred_vec5[:, :4]
    e = y_pred_vec5[:, 4]

    p = params
    s = np.array(p["setpoint_vector"], dtype=float)
    w_logic = float(p["weighting_between_logic_and_effect"]["logic_weight"])
    w_effect = float(p["weighting_between_logic_and_effect"]["effect_weight"])
    beta = float(p["combination_of_logic_and_effect"]["logic_exponent_beta"])
    gamma = float(p["combination_of_logic_and_effect"]["effect_exponent_gamma"])

    # fidelity [0,1]
    F = _logic_fidelity_l2_norm01(v, s)

    # effect scaling per spec
    es = p["effect_size_scaling_for_selection"]
    percentile = int(es["percentile"])
    min_labels = int(es["low_sample_fallback"]["min_labeled_count"])
    fallback_percentile = int(es["low_sample_fallback"]["fallback_percentile"])
    eps = float(es["low_sample_fallback"]["epsilon"])

    E_scaled, denom = _effect_scale_per_round(
        e, e_labels_for_scaling, percentile, min_labels, fallback_percentile, eps
    )

    # product with exponents + weights; both terms are in [0,1]
    score = (np.power(F, beta) ** w_logic) * (np.power(E_scaled, gamma) ** w_effect)
    return ObjectiveResult(
        score=score, logic_fidelity=F, effect_scaled=E_scaled, denom_used=float(denom)
    )
