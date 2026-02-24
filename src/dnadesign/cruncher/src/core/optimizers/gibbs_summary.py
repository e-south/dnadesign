"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/gibbs_summary.py

Build summary payloads for Gibbs annealing optimizer state and diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def final_softmin_beta(optimizer) -> float | None:
    if optimizer.softmin_of is None:
        return None
    return float(optimizer.softmin_of(optimizer.total_sweeps - 1))


def final_mcmc_beta(optimizer) -> float | None:
    if optimizer.total_sweeps < 1:
        return None
    return float(optimizer.mcmc_beta_of(optimizer.total_sweeps - 1))


def build_objective_schedule_summary(optimizer) -> Dict[str, object]:
    return {
        "total_sweeps": optimizer.total_sweeps,
        "beta_ladder_base": list(optimizer.beta_ladder_base),
        "beta_ladder_final": list(optimizer.beta_ladder),
        "mcmc_cooling": dict(optimizer.mcmc_cooling_summary),
    }


def build_stats(optimizer) -> Dict[str, object]:
    totals = dict(optimizer.move_tally)
    accepted = dict(optimizer.accept_tally)
    acceptance_rate = {k: (accepted.get(k, 0) / totals[k]) if totals.get(k, 0) else 0.0 for k in totals}
    mh_kinds = [k for k in totals if k != "S"]
    mh_total = sum(totals.get(k, 0) for k in mh_kinds)
    mh_accept = sum(accepted.get(k, 0) for k in mh_kinds)
    acceptance_rate_mh = (mh_accept / mh_total) if mh_total else 0.0
    all_total = sum(totals.values())
    all_accept = sum(accepted.values())
    acceptance_rate_all = (all_accept / all_total) if all_total else 0.0
    eps = 1.0e-12
    mh_deltas = [
        abs(float(ms.get("delta", 0.0)))
        for ms in optimizer.move_stats
        if ms.get("move_kind") in mh_kinds and ms.get("delta") is not None
    ]
    if mh_deltas:
        delta_abs_median_mh = float(np.median(mh_deltas))
        delta_frac_zero_mh = float(np.mean([d <= eps for d in mh_deltas]))
        score_change_rate_mh = float(np.mean([d > eps for d in mh_deltas]))
    else:
        delta_abs_median_mh = None
        delta_frac_zero_mh = None
        score_change_rate_mh = None
    beta_min = float(optimizer.beta_ladder_base[0]) if optimizer.beta_ladder_base else None
    beta_max_base = float(optimizer.beta_ladder_base[-1]) if optimizer.beta_ladder_base else None
    beta_max_final = float(max(optimizer.beta_ladder)) if optimizer.beta_ladder else None
    return {
        "moves": totals,
        "accepted": accepted,
        "acceptance_rate": acceptance_rate,
        "acceptance_rate_mh": acceptance_rate_mh,
        "acceptance_rate_all": acceptance_rate_all,
        "delta_abs_median_mh": delta_abs_median_mh,
        "delta_frac_zero_mh": delta_frac_zero_mh,
        "score_change_rate_mh": score_change_rate_mh,
        "beta_ladder_base": list(optimizer.beta_ladder_base),
        "beta_ladder_final": list(optimizer.beta_ladder),
        "beta_min": beta_min,
        "beta_max_base": beta_max_base,
        "beta_max_final": beta_max_final,
        "adaptive_moves_enabled": bool(optimizer.move_controller.enabled),
        "proposal_adapt_enabled": bool(optimizer.proposal_controller.enabled),
        "proposal_block_len_range_final": list(optimizer.move_cfg["block_len_range"]),
        "proposal_multi_k_range_final": list(optimizer.move_cfg["multi_k_range"]),
        "adaptive_weights_freeze_after_sweep": optimizer.move_adapt_freeze_after_sweep,
        "adaptive_weights_freeze_after_beta": optimizer.move_adapt_freeze_after_beta,
        "proposal_adapt_freeze_after_sweep": optimizer.proposal_adapt_freeze_after_sweep,
        "proposal_adapt_freeze_after_beta": optimizer.proposal_adapt_freeze_after_beta,
        "gibbs_inertia_enabled": bool(optimizer.gibbs_inertia_enabled),
        "gibbs_inertia_kind": optimizer.gibbs_inertia_kind,
        "gibbs_inertia_p_stay_start": optimizer.gibbs_inertia_p_stay_start,
        "gibbs_inertia_p_stay_end": optimizer.gibbs_inertia_p_stay_end,
        "unique_successes": optimizer.unique_successes,
        "move_stats": optimizer.move_stats,
        "mcmc_cooling": dict(optimizer.mcmc_cooling_summary),
        "final_softmin_beta": final_softmin_beta(optimizer),
        "final_mcmc_beta": final_mcmc_beta(optimizer),
    }
