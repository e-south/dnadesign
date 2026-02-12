"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/mmr_sweep.py

Replay MMR elite selection over parameter grids using saved sampling artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.trajectory import add_raw_llr_objective, build_trajectory_points
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.selection.mmr import (
    MmrCandidate,
    select_mmr_elites,
    select_score_elites,
    tfbs_cores_from_scorer,
)

_BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


def _encode_sequence(seq: str, *, row_idx: int) -> np.ndarray:
    clean = str(seq).strip().upper()
    if not clean:
        raise ValueError(f"MMR sweep row {row_idx} has an empty sequence.")
    try:
        return np.asarray([_BASE_TO_INT[base] for base in clean], dtype=np.int8)
    except KeyError as exc:
        raise ValueError(f"MMR sweep row {row_idx} contains invalid sequence characters: {seq!r}") from exc


def _full_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Full-sequence distance requires equal-length arrays.")
    if a.size == 0:
        raise ValueError("Full-sequence distance requires non-empty arrays.")
    return float(np.count_nonzero(a != b)) / float(a.size)


def _full_distance_metrics(selected: Sequence[MmrCandidate]) -> tuple[float | None, float | None, float | None]:
    if len(selected) < 2:
        return None, None, None
    pairwise: list[float] = []
    nn_vals: list[float] = []
    for i, cand_i in enumerate(selected):
        dists_i: list[float] = []
        for j, cand_j in enumerate(selected):
            if i == j:
                continue
            dist = _full_distance(np.asarray(cand_i.seq_arr, dtype=np.int8), np.asarray(cand_j.seq_arr, dtype=np.int8))
            pairwise.append(dist) if j > i else None
            dists_i.append(dist)
        if dists_i:
            nn_vals.append(float(min(dists_i)))
    if not pairwise:
        return None, None, None
    pair_arr = np.asarray(pairwise, dtype=float)
    nn_arr = np.asarray(nn_vals, dtype=float) if nn_vals else np.asarray([], dtype=float)
    median_nn = float(np.median(nn_arr)) if nn_arr.size else None
    return float(np.mean(pair_arr)), float(np.min(pair_arr)), median_nn


def _resolve_pool_size(pool_size_value: str | int, *, elite_k: int, candidate_count: int) -> int:
    if pool_size_value == "all":
        return int(candidate_count)
    if pool_size_value == "auto":
        auto_target = max(4000, 500 * max(int(elite_k), 1))
        auto_target = min(auto_target, 20_000)
        return min(auto_target, candidate_count) if candidate_count > 0 else 0
    value = int(pool_size_value)
    if value < 1:
        raise ValueError("MMR sweep pool_size must be >= 1.")
    return min(value, candidate_count) if candidate_count > 0 else 0


def _ordered_unique(values: Iterable[object]) -> list[object]:
    seen: set[tuple[type[object], object]] = set()
    out: list[object] = []
    for value in values:
        key = (type(value), value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def run_mmr_sweep(
    *,
    sequences_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    tf_names: Sequence[str],
    pwms: dict[str, PWM],
    objective_config: dict[str, object],
    bidirectional: bool,
    elite_k: int,
    pwm_pseudocounts: float,
    log_odds_clip: float | None,
    pool_size_values: Sequence[str | int],
    diversity_values: Sequence[float],
    baseline_pool_size: str | int | None,
    baseline_diversity: float | None,
) -> pd.DataFrame:
    if sequences_df is None or sequences_df.empty:
        return pd.DataFrame()
    tf_list = [str(tf) for tf in tf_names]
    if not tf_list:
        raise ValueError("MMR sweep requires at least one TF.")

    points = build_trajectory_points(
        sequences_df,
        tf_list,
        max_points=0,
        objective_config=objective_config,
    )
    if points.empty:
        return pd.DataFrame()
    if "phase" in points.columns:
        points = points[points["phase"].astype(str) == "draw"].copy()
    if points.empty:
        return pd.DataFrame()

    points = add_raw_llr_objective(
        points,
        tf_list,
        pwms=pwms,
        objective_config=objective_config,
        bidirectional=bidirectional,
        pwm_pseudocounts=pwm_pseudocounts,
        log_odds_clip=log_odds_clip,
    )

    score_cols = [f"score_{tf}" for tf in tf_list]
    norm_cols = [f"norm_llr_{tf}" for tf in tf_list]
    required = {"chain", "sweep_idx", "sequence", "objective_scalar", *score_cols, *norm_cols}
    missing = [column for column in sorted(required) if column not in points.columns]
    if missing:
        raise ValueError(f"MMR sweep missing required columns: {missing}")

    points = points.sort_values(["chain", "sweep_idx"]).reset_index(drop=True)
    chain_values = pd.to_numeric(points["chain"], errors="coerce")
    sweep_values = pd.to_numeric(points["sweep_idx"], errors="coerce")
    objective_values = pd.to_numeric(points["objective_scalar"], errors="coerce")
    if chain_values.isna().any() or sweep_values.isna().any() or objective_values.isna().any():
        raise ValueError("MMR sweep requires numeric chain, sweep_idx, and objective_scalar columns.")

    score_matrix = points[score_cols].to_numpy(dtype=float)
    norm_matrix = points[norm_cols].to_numpy(dtype=float)
    scorer_llr = Scorer(
        {tf: pwms[tf] for tf in tf_list},
        bidirectional=bool(bidirectional),
        scale="llr",
        pseudocounts=float(pwm_pseudocounts),
        log_odds_clip=log_odds_clip,
    )

    candidates: list[MmrCandidate] = []
    core_maps: dict[str, dict[str, np.ndarray]] = {}
    for idx in range(len(points)):
        seq_arr = _encode_sequence(str(points.at[idx, "sequence"]), row_idx=idx)
        chain_id = int(chain_values.iloc[idx])
        sweep_idx = int(sweep_values.iloc[idx])
        candidate_id = f"{chain_id}:{sweep_idx}"
        per_tf_map = {tf: float(score_matrix[idx, tf_idx]) for tf_idx, tf in enumerate(tf_list)}
        norm_map = {tf: float(norm_matrix[idx, tf_idx]) for tf_idx, tf in enumerate(tf_list)}
        candidates.append(
            MmrCandidate(
                seq_arr=seq_arr,
                chain_id=chain_id,
                draw_idx=sweep_idx,
                combined_score=float(objective_values.iloc[idx]),
                min_norm=float(np.min(norm_matrix[idx, :])),
                sum_norm=float(np.sum(norm_matrix[idx, :])),
                per_tf_map=per_tf_map,
                norm_map=norm_map,
            )
        )
        core_maps[candidate_id] = tfbs_cores_from_scorer(seq_arr, scorer=scorer_llr, tf_names=tf_list)

    if not candidates:
        return pd.DataFrame()
    sequence_length = int(candidates[0].seq_arr.size)
    first_core_map = next(iter(core_maps.values()))
    core_width = int(sum(int(np.asarray(first_core_map[tf]).size) for tf in tf_list))

    baseline_ids: set[str] = set()
    if elites_df is not None and not elites_df.empty and {"chain", "draw_idx"}.issubset(set(elites_df.columns)):
        chain = pd.to_numeric(elites_df["chain"], errors="coerce")
        draw = pd.to_numeric(elites_df["draw_idx"], errors="coerce")
        mask = chain.notna() & draw.notna()
        baseline_ids = {f"{int(c)}:{int(d)}" for c, d in zip(chain[mask], draw[mask])}

    pool_grid: list[str | int] = []
    for value in _ordered_unique(pool_size_values):
        if value in {"auto", "all"}:
            pool_grid.append(str(value))
            continue
        pool_grid.append(int(value))
    diversity_grid = [float(value) for value in _ordered_unique(diversity_values)]
    relevance = "min_tf_score"
    distance_metric = "hybrid"
    constraint_policy = "relax"

    rows: list[dict[str, object]] = []

    def _pool_matches_baseline(pool_input: str | int, pool_resolved: int | None) -> bool:
        if baseline_pool_size is None:
            return False
        if baseline_pool_size == "auto":
            return pool_input == "auto"
        if baseline_pool_size == "all":
            return pool_input == "all"
        if pool_input in {"auto", "all"}:
            return False
        if pool_resolved is None:
            return False
        try:
            return int(pool_resolved) == int(baseline_pool_size)
        except (TypeError, ValueError, OverflowError):
            return False

    for pool_size_value in pool_grid:
        candidate_count = len(candidates)
        if candidate_count == 0:
            for diversity in diversity_grid:
                score_only = float(diversity) <= 0.0
                rows.append(
                    {
                        "score_weight": float(max(0.0, 1.0 - float(diversity))),
                        "diversity_weight": (0.0 if score_only else float(diversity)),
                        "diversity": float(diversity),
                        "distance_metric": ("none" if score_only else distance_metric),
                        "constraint_policy": ("disabled" if score_only else constraint_policy),
                        "min_hamming_bp_requested": None,
                        "min_hamming_bp_final": None,
                        "min_core_hamming_bp_requested": None,
                        "min_core_hamming_bp_final": None,
                        "relax_steps_used": None,
                        "selection_error": None,
                        "pool_size_input": pool_size_value,
                        "pool_size_resolved": None,
                        "candidate_count": 0,
                        "selected_count": 0,
                        "median_relevance_raw": None,
                        "mean_pairwise_core_distance": None,
                        "min_pairwise_core_distance": None,
                        "mean_pairwise_full_distance": None,
                        "min_pairwise_full_distance": None,
                        "median_nn_full_distance": None,
                        "median_min_tf_score_selected": None,
                        "median_joint_score_selected": None,
                        "jaccard_vs_current_elites": None,
                        "is_current_config": False,
                    }
                )
            continue
        pool_size_resolved = _resolve_pool_size(
            pool_size_value,
            elite_k=elite_k,
            candidate_count=candidate_count,
        )
        for diversity in diversity_grid:
            score_only = float(diversity) <= 0.0
            score_weight = 1.0 if score_only else float(max(0.0, 1.0 - float(diversity)))
            diversity_weight = 0.0 if score_only else float(diversity)
            effective_distance_metric = "none" if score_only else distance_metric
            min_hamming_bp = (
                int(round(float(diversity) * max(0, int(sequence_length) // 4))) if not score_only else None
            )
            min_core_hamming_bp = (
                int(round(float(diversity) * max(0, int(core_width) // 4))) if not score_only else None
            )
            selection_error = None
            result = None
            try:
                if score_only:
                    result = select_score_elites(
                        candidates,
                        k=int(elite_k),
                        pool_size=int(pool_size_resolved),
                        dsdna=bool(bidirectional),
                    )
                else:
                    result = select_mmr_elites(
                        candidates,
                        k=int(elite_k),
                        pool_size=int(pool_size_resolved),
                        alpha=score_weight,
                        relevance=relevance,
                        dsdna=bool(bidirectional),
                        tf_names=tf_list,
                        pwms=pwms,
                        core_maps=core_maps,
                        distance_metric=effective_distance_metric,
                        min_hamming_bp=min_hamming_bp,
                        min_core_hamming_bp=min_core_hamming_bp,
                        constraint_policy=constraint_policy,
                        relax_step_bp=1,
                        relax_min_bp=0,
                    )
            except ValueError as exc:
                selection_error = str(exc)
            selected = result.selected if result is not None else []
            selected_ids = {f"{cand.chain_id}:{cand.draw_idx}" for cand in selected}
            selected_min_norm = [float(cand.min_norm) for cand in selected]
            selected_joint_score = [float(cand.combined_score) for cand in selected]
            full_mean, full_min, full_nn_median = _full_distance_metrics(selected)
            jaccard = None
            if baseline_ids:
                union = baseline_ids | selected_ids
                if union:
                    jaccard = float(len(baseline_ids & selected_ids)) / float(len(union))
            rows.append(
                {
                    "score_weight": score_weight,
                    "diversity_weight": diversity_weight,
                    "diversity": float(diversity),
                    "distance_metric": effective_distance_metric,
                    "constraint_policy": ("disabled" if score_only else constraint_policy),
                    "min_hamming_bp_requested": min_hamming_bp,
                    "min_hamming_bp_final": (result.min_hamming_bp_final if result is not None else None),
                    "min_core_hamming_bp_requested": min_core_hamming_bp,
                    "min_core_hamming_bp_final": (result.min_core_hamming_bp_final if result is not None else None),
                    "relax_steps_used": result.relax_steps_used if result is not None else None,
                    "selection_error": selection_error,
                    "pool_size_input": pool_size_value,
                    "pool_size_resolved": int(pool_size_resolved),
                    "candidate_count": int(candidate_count),
                    "selected_count": int(len(selected)),
                    "median_relevance_raw": (result.median_relevance_raw if result is not None else None),
                    "mean_pairwise_core_distance": (result.mean_pairwise_distance if result is not None else None),
                    "min_pairwise_core_distance": (result.min_pairwise_distance if result is not None else None),
                    "mean_pairwise_full_distance": full_mean,
                    "min_pairwise_full_distance": full_min,
                    "median_nn_full_distance": full_nn_median,
                    "median_min_tf_score_selected": (
                        float(np.median(np.asarray(selected_min_norm, dtype=float))) if selected_min_norm else None
                    ),
                    "median_joint_score_selected": (
                        float(np.median(np.asarray(selected_joint_score, dtype=float)))
                        if selected_joint_score
                        else None
                    ),
                    "jaccard_vs_current_elites": jaccard,
                    "is_current_config": (
                        _pool_matches_baseline(pool_size_value, int(pool_size_resolved))
                        and (
                            (baseline_diversity is None and abs(float(diversity)) <= 1.0e-12)
                            or (
                                baseline_diversity is not None
                                and abs(float(baseline_diversity) - float(diversity)) <= 1.0e-12
                            )
                        )
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Keep a stable scalar type for parquet serialization even when values include "auto" and integers.
    out["pool_size_input"] = out["pool_size_input"].map(lambda value: None if value is None else str(value))
    return out
