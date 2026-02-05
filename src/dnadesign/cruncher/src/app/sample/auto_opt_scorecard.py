"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/auto_opt_scorecard.py

Auto-opt scorecard helpers and candidate ranking.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

import numpy as np

from dnadesign.cruncher.app.sample.auto_opt_models import AutoOptCandidate
from dnadesign.cruncher.config.schema_v2 import AutoOptConfig
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.selection.mmr import (
    compute_core_distance,
    compute_position_weights,
    tfbs_cores_from_hits,
)

_BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


def _encode_sequence_string(seq: str) -> np.ndarray:
    clean = seq.strip().upper()
    try:
        return np.array([_BASE_TO_INT[ch] for ch in clean], dtype=np.int8)
    except KeyError as exc:
        raise ValueError(f"Invalid base in sequence '{seq}'") from exc


def _scorecard_elite_subset(elites_df, *, k: int):
    if elites_df is None or elites_df.empty:
        return elites_df
    if "rank" in elites_df.columns:
        return elites_df.nsmallest(min(int(k), len(elites_df)), "rank")
    return elites_df.head(min(int(k), len(elites_df)))


def _mmr_scorecard_from_elites(
    elites_df,
    *,
    tf_names: list[str],
    selection_cfg: object,
    auto_cfg: AutoOptConfig,
    pwms: dict[str, PWM],
) -> tuple[float | None, float | None, float | None]:
    if elites_df is None or elites_df.empty:
        return None, None, None
    k = auto_cfg.policy.scorecard.k
    subset = _scorecard_elite_subset(elites_df, k=k)
    relevance = getattr(selection_cfg, "relevance", "min_per_tf_norm")
    if relevance == "combined_score_final":
        col = "combined_score_final"
    else:
        col = "min_norm"
    if col not in subset.columns:
        raise ValueError(f"Missing '{col}' column in elites.parquet for auto-opt scorecard.")
    relevance_vals = subset[col].astype(float).to_numpy()
    if relevance_vals.size == 0:
        return None, None, None
    median_relevance_raw = float(np.median(relevance_vals))

    distances: list[float] = []
    weights_by_tf: dict[str, np.ndarray] = {}
    for tf in tf_names:
        pwm = pwms.get(tf)
        if pwm is None:
            raise ValueError(f"Missing PWM for TF '{tf}'.")
        weights_by_tf[tf] = compute_position_weights(pwm)
    cores = []
    for _, row in subset.iterrows():
        if "per_tf_json" in row and isinstance(row["per_tf_json"], str):
            per_tf_hits = json.loads(row["per_tf_json"])
        else:
            per_tf_hits = row.get("per_tf")
        if not isinstance(per_tf_hits, dict):
            raise ValueError("Missing per_tf metadata for TFBS core distance in elites.parquet.")
        seq_arr = _encode_sequence_string(str(row["sequence"]))
        cores.append(tfbs_cores_from_hits(seq_arr, per_tf_hits=per_tf_hits, tf_names=tf_names))
    for idx, core_a in enumerate(cores):
        for core_b in cores[idx + 1 :]:
            distances.append(compute_core_distance(core_a, core_b, weights=weights_by_tf, tf_names=tf_names))

    mean_pairwise_distance = float(np.mean(distances)) if distances else 0.0
    pilot_score = median_relevance_raw + auto_cfg.policy.diversity_weight * mean_pairwise_distance
    return pilot_score, median_relevance_raw, mean_pairwise_distance


def _assess_candidate_quality(
    candidate: AutoOptCandidate,
    auto_cfg: AutoOptConfig,
    *,
    mode: str,
) -> list[str]:
    _ = auto_cfg
    notes: list[str] = []
    if candidate.status == "fail":
        candidate.quality = "fail"
        return notes
    if mode != "auto_opt":
        candidate.quality = "ok"
        return notes
    rhat_warn = 1.15
    rhat_fail = 1.50
    ess_ratio_warn = 0.10
    ess_ratio_fail = 0.02
    unique_warn = 0.10
    unique_fail = 0.0

    pilot_draws = None
    pilot_draws_expected = None
    pilot_chains = None
    ess_ratio = None
    pilot_short = False
    if isinstance(candidate.diagnostics, dict):
        metrics = candidate.diagnostics.get("metrics")
        if isinstance(metrics, dict):
            trace = metrics.get("trace")
            if isinstance(trace, dict):
                pilot_draws = trace.get("draws")
                pilot_draws_expected = trace.get("draws_expected")
                pilot_chains = trace.get("chains")
                ess_ratio = trace.get("ess_ratio")
    try:
        pilot_draws = int(pilot_draws) if pilot_draws is not None else None
    except (TypeError, ValueError):
        pilot_draws = None
    try:
        pilot_draws_expected = int(pilot_draws_expected) if pilot_draws_expected is not None else None
    except (TypeError, ValueError):
        pilot_draws_expected = None
    try:
        pilot_chains = int(pilot_chains) if pilot_chains is not None else None
    except (TypeError, ValueError):
        pilot_chains = None
    try:
        ess_ratio = float(ess_ratio) if ess_ratio is not None else None
    except (TypeError, ValueError):
        ess_ratio = None

    pilot_min_draws = 200
    pilot_min_fraction = 0.5
    pilot_threshold = pilot_min_draws
    if pilot_draws_expected is not None:
        pilot_threshold = max(pilot_min_draws, int(round(pilot_draws_expected * pilot_min_fraction)))
    if pilot_draws is not None and pilot_draws <= pilot_threshold:
        pilot_short = True
        notes.append(f"pilot draws={pilot_draws} <= {pilot_threshold}; diagnostics are directional at short budgets")

    quality = "ok"
    if pilot_short:
        quality = "warn"
    if candidate.unique_fraction is not None:
        if candidate.unique_fraction <= unique_fail:
            quality = "fail"
            notes.append(f"unique_fraction={candidate.unique_fraction:.2f} <= {unique_fail:.2f}")
        elif candidate.unique_fraction < unique_warn and quality != "fail":
            quality = "warn"
            notes.append(f"unique_fraction={candidate.unique_fraction:.2f} < {unique_warn:.2f}")
    if not pilot_short:
        if candidate.rhat is not None:
            if candidate.rhat >= rhat_fail:
                quality = "fail"
                notes.append(f"rhat={candidate.rhat:.3f} >= {rhat_fail}")
            elif candidate.rhat > rhat_warn:
                quality = "warn"
                notes.append(f"rhat={candidate.rhat:.3f} > {rhat_warn}")
        if ess_ratio is None and candidate.ess is not None and pilot_draws is not None:
            denom = pilot_draws
            if candidate.kind != "pt":
                denom *= pilot_chains or 1
            if denom > 0:
                ess_ratio = candidate.ess / float(denom)
        if ess_ratio is not None and ess_ratio < ess_ratio_warn and quality != "fail":
            quality = "warn"
            threshold = ess_ratio_fail if ess_ratio <= ess_ratio_fail else ess_ratio_warn
            notes.append(f"ess_ratio={ess_ratio:.3f} < {threshold:.2f}")

    candidate.quality = quality
    if quality == "warn" and not pilot_short:
        notes.append("pilot diagnostics are weak; consider increasing budgets or beta schedules")
    return notes


def _rank_auto_opt_candidates(
    candidates: list[AutoOptCandidate],
) -> list[AutoOptCandidate]:
    ranked: list[tuple[tuple[object, ...], AutoOptCandidate]] = []
    status_rank = {"ok": 2, "warn": 1, "fail": 0}
    for candidate in candidates:
        status_value = float(status_rank.get(candidate.quality, status_rank.get(candidate.status, 0)))
        candidate_id = str(candidate.run_dir)
        score = candidate.pilot_score if candidate.pilot_score is not None else float("-inf")
        median_rel = candidate.median_relevance_raw if candidate.median_relevance_raw is not None else float("-inf")
        mean_dist = candidate.mean_pairwise_distance if candidate.mean_pairwise_distance is not None else float("-inf")
        rank = (float(score), float(median_rel), float(mean_dist), status_value, candidate_id)
        ranked.append((rank, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked]


def _confidence_from_candidates(
    candidates: list[AutoOptCandidate],
) -> tuple[str, AutoOptCandidate | None, AutoOptCandidate | None]:
    if not candidates:
        return "low", None, None
    ranked = _rank_auto_opt_candidates(candidates)
    best = ranked[0]
    if len(ranked) == 1:
        return "high" if best.quality == "ok" else "medium", best, None
    second = ranked[1]
    if best.top_k_ci_low is None or second.top_k_ci_high is None:
        ci_separated = False
    else:
        ci_separated = best.top_k_ci_low > second.top_k_ci_high
    confidence_level = "high" if ci_separated else "low"
    if best.quality == "warn" and confidence_level == "high":
        confidence_level = "medium"
    if best.quality != "ok" and confidence_level == "high":
        confidence_level = "medium"
    return confidence_level, best, second


def _validate_auto_opt_candidates(
    candidates: list[AutoOptCandidate],
    *,
    allow_warn: bool,
) -> tuple[list[AutoOptCandidate], bool]:
    if not candidates:
        raise ValueError("Auto-optimize did not produce any pilot candidates.")
    viable = [c for c in candidates if c.status != "fail"]
    if not viable:
        raise ValueError(
            "Auto-optimize failed: all pilot candidates failed catastrophic checks "
            "(missing scores or no movement). Re-run with larger budgets or fix the model inputs."
        )
    if allow_warn:
        viable = [c for c in viable if c.quality != "fail"]
        if not viable:
            raise ValueError(
                "Auto-optimize failed: all pilot candidates failed quality checks. "
                "Increase budgets/replicates or adjust diagnostics thresholds."
            )
    else:
        viable = [c for c in viable if c.quality == "ok"]
        if not viable:
            raise ValueError(
                "Auto-optimize failed: no candidates passed diagnostics without warnings. "
                "Set auto_opt.policy.allow_warn=true or increase budgets/replicates."
            )
    return viable, False


def _select_auto_opt_candidate(
    candidates: list[AutoOptCandidate],
    *,
    allow_fail: bool = False,
) -> AutoOptCandidate:
    if not candidates:
        raise ValueError("Auto-optimize did not produce any pilot candidates")

    ok_candidates = [c for c in candidates if c.status != "fail"]
    if not ok_candidates and allow_fail:
        ok_candidates = list(candidates)

    ranked = _rank_auto_opt_candidates(ok_candidates)
    winner = ranked[0]
    if winner.status == "fail" and not allow_fail:
        raise ValueError("Auto-optimize failed: all pilot candidates reported missing diagnostics.")
    return winner
