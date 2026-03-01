"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/gibbs_runtime.py

Provide runtime controls and progress accounting for Gibbs annealing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np

from dnadesign.cruncher.core.optimizers.policies import MOVE_KINDS


def move_adaptation_frozen(
    *,
    sweep_idx: int,
    beta_now: float,
    freeze_after_sweep: int | None,
    freeze_after_beta: float | None,
) -> bool:
    if freeze_after_sweep is not None and int(sweep_idx) >= int(freeze_after_sweep):
        return True
    if freeze_after_beta is not None and float(beta_now) >= float(freeze_after_beta):
        return True
    return False


def proposal_adaptation_frozen(
    *,
    sweep_idx: int,
    beta_now: float,
    freeze_after_sweep: int | None,
    freeze_after_beta: float | None,
) -> bool:
    if freeze_after_sweep is not None and int(sweep_idx) >= int(freeze_after_sweep):
        return True
    if freeze_after_beta is not None and float(beta_now) >= float(freeze_after_beta):
        return True
    return False


def gibbs_stay_probability(
    *,
    sweep_idx: int,
    enabled: bool,
    inertia_kind: str,
    total_sweeps: int,
    p_stay_start: float,
    p_stay_end: float,
) -> float:
    if not enabled:
        return 0.0
    if inertia_kind == "fixed" or int(total_sweeps) <= 1:
        return float(np.clip(float(p_stay_end), 0.0, 1.0))
    frac = min(max(float(sweep_idx) / float(max(int(total_sweeps) - 1, 1)), 0.0), 1.0)
    value = float(p_stay_start) + frac * (float(p_stay_end) - float(p_stay_start))
    return float(np.clip(value, 0.0, 1.0))


def sample_move_kind(*, rng: np.random.Generator, move_probs: np.ndarray) -> str:
    return str(rng.choice(MOVE_KINDS, p=move_probs))


def move_detail(
    *,
    seq: np.ndarray,
    seq_before: np.ndarray,
    score_old: float,
    score_new: float,
    accepted: bool,
    gibbs_changed: bool | None = None,
) -> dict[str, object]:
    delta_hamming = int(np.count_nonzero(seq != seq_before)) if bool(accepted) else 0
    return {
        "delta": float(score_new - score_old),
        "score_old": float(score_old),
        "score_new": float(score_new),
        "delta_hamming": int(delta_hamming),
        "gibbs_changed": bool(gibbs_changed) if gibbs_changed is not None else None,
    }


def accept_metropolis(*, beta: float, new_score: float, current_combined: float, rng: np.random.Generator) -> bool:
    delta = float(beta) * (float(new_score) - float(current_combined))
    return bool(delta >= 0 or np.log(rng.random()) < delta)


def maybe_log_progress(
    *,
    logger: logging.Logger,
    telemetry,
    progress_every: int,
    phase: str,
    step: int,
    total: int,
    move_tally: Mapping[str, int],
    accept_tally: Mapping[str, int],
    best_score: float | None,
    best_meta: Sequence[int] | None,
    beta_ladder: Sequence[float],
    beta_softmin: float | None = None,
    current_score: float | None = None,
    score_mean: float | None = None,
    score_std: float | None = None,
) -> None:
    if not int(progress_every):
        return
    if int(step) % int(progress_every) != 0 and int(step) != int(total):
        return
    totals = dict(move_tally)
    accepted = dict(accept_tally)
    acceptance_rate = {k: (accepted.get(k, 0) / totals[k]) if totals.get(k, 0) else 0.0 for k in totals}
    acc_label = ", ".join(f"{k}={acceptance_rate[k]:.2f}" for k in sorted(acceptance_rate))
    mh_kinds = [k for k in totals if k != "S"]
    mh_total = sum(totals.get(k, 0) for k in mh_kinds)
    mh_accept = sum(accepted.get(k, 0) for k in mh_kinds)
    acceptance_rate_mh = (mh_accept / mh_total) if mh_total else 0.0
    all_total = sum(totals.values())
    all_accept = sum(accepted.values())
    acceptance_rate_all = (all_accept / all_total) if all_total else 0.0
    pct = (int(step) / int(total)) * 100 if int(total) else 100.0
    score_blob = ""
    if current_score is not None:
        score_blob = f" score={current_score:.3f}"
    if score_mean is not None and score_std is not None:
        score_blob += f" mean={score_mean:.3f}±{score_std:.3f}"
    if best_score is not None:
        score_blob += f" best={best_score:.3f}"
    logger.info(
        "Progress: %s %d/%d (%.1f%%) accept={%s}%s",
        phase,
        step,
        total,
        pct,
        acc_label,
        score_blob,
    )
    telemetry.update(
        phase=phase,
        step=step,
        total=total,
        progress_pct=round(pct, 2),
        acceptance_rate=acceptance_rate,
        acceptance_rate_mh=acceptance_rate_mh,
        acceptance_rate_all=acceptance_rate_all,
        current_score=current_score,
        score_mean=score_mean,
        score_std=score_std,
        beta_softmin=beta_softmin,
        beta_min=min(beta_ladder) if beta_ladder else None,
        beta_max=max(beta_ladder) if beta_ladder else None,
        best_score=best_score,
        best_chain=(int(best_meta[0]) + 1) if best_meta else None,
        best_draw=int(best_meta[1]) if best_meta else None,
    )
