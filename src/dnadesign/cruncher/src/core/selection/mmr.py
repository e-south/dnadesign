"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/selection/mmr.py

MMR-based elite selection utilities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.selection.mmr_distance import (
    compute_core_distance,
    compute_position_weights,
)
from dnadesign.cruncher.core.selection.mmr_distance import (
    core_hamming_bp as _core_hamming_bp,
)
from dnadesign.cruncher.core.selection.mmr_distance import (
    full_sequence_distance as _full_sequence_distance,
)
from dnadesign.cruncher.core.selection.mmr_distance import (
    full_sequence_distance_bp as _full_sequence_distance_bp,
)
from dnadesign.cruncher.core.selection.mmr_distance import (
    tfbs_cores_from_hits as _tfbs_cores_from_hits,
)
from dnadesign.cruncher.core.selection.mmr_distance import (
    tfbs_cores_from_scorer as _tfbs_cores_from_scorer,
)
from dnadesign.cruncher.core.sequence import canon_int
from dnadesign.cruncher.core.state import SequenceState

tfbs_cores_from_hits = _tfbs_cores_from_hits
tfbs_cores_from_scorer = _tfbs_cores_from_scorer


@dataclass(frozen=True)
class MmrCandidate:
    seq_arr: np.ndarray
    chain_id: int
    draw_idx: int
    combined_score: float
    min_norm: float
    sum_norm: float
    per_tf_map: dict[str, float]
    norm_map: dict[str, float]


@dataclass(frozen=True)
class MmrSelectionResult:
    selected: list[MmrCandidate]
    meta: list[dict[str, object]]
    pool_size: int
    k: int
    alpha: float
    median_relevance_raw: float | None
    mean_pairwise_distance: float | None
    min_pairwise_distance: float | None
    min_hamming_bp_requested: int | None = None
    min_hamming_bp_final: int | None = None
    min_core_hamming_bp_requested: int | None = None
    min_core_hamming_bp_final: int | None = None
    constraint_policy: str | None = None
    relax_steps_used: int = 0


def _candidate_id(candidate: MmrCandidate) -> str:
    return f"{int(candidate.chain_id)}:{int(candidate.draw_idx)}"


def _sequence_string(seq_arr: np.ndarray) -> str:
    return SequenceState(seq_arr).to_string()


def _canonical_string(seq_arr: np.ndarray) -> str:
    return SequenceState(canon_int(seq_arr)).to_string()


def _percentile_ranks(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j)
        pct = float(avg_rank) / float(n - 1)
        ranks[order[i : j + 1]] = pct
        i = j + 1
    return ranks.tolist()


def _pairwise_distances(
    selected: list[MmrCandidate],
    *,
    tf_names: Sequence[str],
    weights_by_tf: dict[str, np.ndarray],
    core_maps: dict[str, dict[str, np.ndarray]],
    distance_metric: str,
    hybrid_full_weight: float,
    hybrid_core_weight: float,
) -> list[float]:
    if len(selected) < 2:
        return []
    distances: list[float] = []
    for idx, cand_a in enumerate(selected):
        for cand_b in selected[idx + 1 :]:
            core_a = core_maps[_candidate_id(cand_a)]
            core_b = core_maps[_candidate_id(cand_b)]
            core_dist = compute_core_distance(core_a, core_b, weights=weights_by_tf, tf_names=tf_names)
            full_dist = _full_sequence_distance(cand_a.seq_arr, cand_b.seq_arr)
            if distance_metric == "full":
                dist = full_dist
            elif distance_metric == "hybrid":
                denom = hybrid_full_weight + hybrid_core_weight
                dist = ((hybrid_full_weight * full_dist) + (hybrid_core_weight * core_dist)) / denom
            else:
                dist = core_dist
            distances.append(dist)
    return distances


def _dedupe_pool(
    pool: Sequence[MmrCandidate],
    *,
    dsdna: bool,
) -> list[MmrCandidate]:
    deduped: list[MmrCandidate] = []
    seen: set[str] = set()
    for cand in pool:
        key = _canonical_string(cand.seq_arr) if dsdna else _sequence_string(cand.seq_arr)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    return deduped


def select_score_elites(
    candidates: Sequence[MmrCandidate],
    *,
    k: int,
    pool_size: int,
    dsdna: bool,
) -> MmrSelectionResult:
    if k < 0:
        raise ValueError("k must be >= 0")
    if pool_size < 1:
        raise ValueError("pool_size must be >= 1")
    if k == 0 or not candidates:
        return MmrSelectionResult(
            selected=[],
            meta=[],
            pool_size=pool_size,
            k=k,
            alpha=1.0,
            median_relevance_raw=None,
            mean_pairwise_distance=None,
            min_pairwise_distance=None,
            min_hamming_bp_requested=None,
            min_hamming_bp_final=None,
            min_core_hamming_bp_requested=None,
            min_core_hamming_bp_final=None,
            constraint_policy="disabled",
            relax_steps_used=0,
        )

    candidates_sorted = sorted(
        candidates,
        key=lambda cand: (cand.combined_score, _candidate_id(cand)),
        reverse=True,
    )
    effective_pool_size = max(pool_size, k)
    pool = _dedupe_pool(candidates_sorted[:effective_pool_size], dsdna=dsdna)
    if not pool:
        return MmrSelectionResult(
            selected=[],
            meta=[],
            pool_size=pool_size,
            k=k,
            alpha=1.0,
            median_relevance_raw=None,
            mean_pairwise_distance=None,
            min_pairwise_distance=None,
            min_hamming_bp_requested=None,
            min_hamming_bp_final=None,
            min_core_hamming_bp_requested=None,
            min_core_hamming_bp_final=None,
            constraint_policy="disabled",
            relax_steps_used=0,
        )

    selected = pool[:k]
    relevance_raw = [float(cand.combined_score) for cand in pool]
    relevance_scaled = _percentile_ranks(relevance_raw)
    index_by_id = {_candidate_id(cand): idx for idx, cand in enumerate(pool)}
    meta_rows: list[dict[str, object]] = []
    for rank, cand in enumerate(selected, start=1):
        idx = index_by_id[_candidate_id(cand)]
        meta_rows.append(
            {
                "elite_rank": rank,
                "candidate_id": _candidate_id(cand),
                "sequence": _sequence_string(cand.seq_arr),
                "canonical_sequence": _canonical_string(cand.seq_arr),
                "utility": float(cand.combined_score),
                "relevance_raw": relevance_raw[idx],
                "relevance_norm": relevance_scaled[idx],
                "nearest_selected_id": None,
                "nearest_distance": None,
                "nearest_similarity": None,
                "nearest_distance_full": None,
                "nearest_distance_core": None,
                "nearest_distance_core_bp": None,
            }
        )

    pairwise: list[float] = []
    for i, cand_a in enumerate(selected):
        for cand_b in selected[i + 1 :]:
            pairwise.append(_full_sequence_distance(cand_a.seq_arr, cand_b.seq_arr))
    selected_scores = [float(cand.combined_score) for cand in selected]
    return MmrSelectionResult(
        selected=selected,
        meta=meta_rows,
        pool_size=pool_size,
        k=k,
        alpha=1.0,
        median_relevance_raw=float(np.median(np.asarray(selected_scores, dtype=float))) if selected_scores else None,
        mean_pairwise_distance=float(np.mean(pairwise)) if pairwise else None,
        min_pairwise_distance=float(np.min(pairwise)) if pairwise else None,
        min_hamming_bp_requested=None,
        min_hamming_bp_final=None,
        min_core_hamming_bp_requested=None,
        min_core_hamming_bp_final=None,
        constraint_policy="disabled",
        relax_steps_used=0,
    )


@dataclass(frozen=True)
class _MmrChoice:
    idx: int
    utility: float
    nearest_id: str | None
    nearest_distance: float
    nearest_similarity: float
    nearest_full_distance: float
    nearest_core_distance: float
    nearest_core_bp: int


def _empty_mmr_result(
    *,
    pool_size: int,
    k: int,
    alpha: float,
    min_hamming_bp_requested: int | None,
    min_hamming_bp_final: int | None,
    min_core_hamming_bp_requested: int | None,
    min_core_hamming_bp_final: int | None,
    constraint_policy: str | None,
    relax_steps_used: int = 0,
) -> MmrSelectionResult:
    return MmrSelectionResult(
        selected=[],
        meta=[],
        pool_size=pool_size,
        k=k,
        alpha=alpha,
        median_relevance_raw=None,
        mean_pairwise_distance=None,
        min_pairwise_distance=None,
        min_hamming_bp_requested=min_hamming_bp_requested,
        min_hamming_bp_final=min_hamming_bp_final,
        min_core_hamming_bp_requested=min_core_hamming_bp_requested,
        min_core_hamming_bp_final=min_core_hamming_bp_final,
        constraint_policy=constraint_policy,
        relax_steps_used=relax_steps_used,
    )


def _resolve_mmr_metric_and_policy(
    *,
    distance_metric: str | None,
    constraint_policy: str,
) -> tuple[str, str]:
    resolved_metric = str(distance_metric or "core").strip().lower()
    if resolved_metric not in {"full", "core", "hybrid"}:
        raise ValueError(f"Unknown distance_metric '{distance_metric}'.")
    policy = str(constraint_policy).strip().lower()
    if policy not in {"relax", "strict"}:
        raise ValueError("constraint_policy must be 'relax' or 'strict'.")
    return resolved_metric, policy


def _validate_mmr_runtime_inputs(
    *,
    relevance: str,
    distance_metric: str | None,
    constraint_policy: str,
    min_hamming_bp: int | None,
    min_core_hamming_bp: int | None,
    relax_step_bp: int,
    relax_min_bp: int,
    hybrid_full_weight: float,
    hybrid_core_weight: float,
) -> tuple[str, str]:
    if relevance not in {"min_tf_score", "joint_score"}:
        raise ValueError(f"Unknown relevance '{relevance}'.")
    resolved_metric, policy = _resolve_mmr_metric_and_policy(
        distance_metric=distance_metric,
        constraint_policy=constraint_policy,
    )
    if min_hamming_bp is not None and int(min_hamming_bp) < 0:
        raise ValueError("min_hamming_bp must be >= 0 when provided.")
    if min_core_hamming_bp is not None and int(min_core_hamming_bp) < 0:
        raise ValueError("min_core_hamming_bp must be >= 0 when provided.")
    if relax_step_bp < 1:
        raise ValueError("relax_step_bp must be >= 1.")
    if relax_min_bp < 0:
        raise ValueError("relax_min_bp must be >= 0.")
    if hybrid_full_weight <= 0 or hybrid_core_weight <= 0:
        raise ValueError("hybrid_full_weight and hybrid_core_weight must be > 0.")
    return resolved_metric, policy


def _seed_meta_row(
    *,
    seed: MmrCandidate,
    seed_idx: int,
    seed_id: str,
    alpha: float,
    relevance_raw: Sequence[float],
    relevance_scaled: Sequence[float],
    core_maps: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "elite_rank": 1,
        "candidate_id": seed_id,
        "sequence": _sequence_string(seed.seq_arr),
        "canonical_sequence": _canonical_string(seed.seq_arr),
        "utility": alpha * relevance_scaled[seed_idx],
        "relevance_raw": relevance_raw[seed_idx],
        "relevance_norm": relevance_scaled[seed_idx],
        "nearest_selected_id": None,
        "nearest_distance": None,
        "nearest_similarity": None,
    }
    for tf, core in core_maps[seed_id].items():
        row[f"core_{tf}"] = SequenceState(core).to_string()
    return row


def _remaining_candidate_indices(*, pool: Sequence[MmrCandidate], selected_ids: set[str]) -> list[int]:
    return [idx for idx, cand in enumerate(pool) if _candidate_id(cand) not in selected_ids]


def _best_choice_for_remaining(
    *,
    remaining_indices: Sequence[int],
    selected_indices: Sequence[int],
    id_by_index: Sequence[str],
    pair_metrics: Callable[[int, int], tuple[float, float, float, int, int]],
    alpha: float,
    relevance_raw: Sequence[float],
    relevance_scaled: Sequence[float],
    min_hamming_bp: int | None,
    min_core_hamming_bp: int | None,
) -> _MmrChoice | None:
    best_choice: _MmrChoice | None = None
    for idx in remaining_indices:
        choice = _evaluate_candidate_choice(
            idx=idx,
            selected_indices=selected_indices,
            id_by_index=id_by_index,
            pair_metrics=pair_metrics,
            alpha=alpha,
            relevance_scaled=relevance_scaled,
            min_hamming_bp=min_hamming_bp,
            min_core_hamming_bp=min_core_hamming_bp,
        )
        if choice is None:
            continue
        if _is_better_choice(
            candidate=choice,
            current_best=best_choice,
            relevance_raw=relevance_raw,
            id_by_index=id_by_index,
        ):
            best_choice = choice
    return best_choice


def _append_choice_selection(
    *,
    choice: _MmrChoice,
    pool: Sequence[MmrCandidate],
    id_by_index: Sequence[str],
    core_maps: dict[str, dict[str, np.ndarray]],
    alpha: float,
    relevance_raw: Sequence[float],
    relevance_scaled: Sequence[float],
    selected: list[MmrCandidate],
    selected_indices: list[int],
    selected_ids: set[str],
    meta_rows: list[dict[str, object]],
) -> None:
    cand = pool[choice.idx]
    cand_id = id_by_index[choice.idx]
    selected.append(cand)
    selected_indices.append(choice.idx)
    selected_ids.add(cand_id)
    row: dict[str, object] = {
        "elite_rank": len(selected),
        "candidate_id": cand_id,
        "sequence": _sequence_string(cand.seq_arr),
        "canonical_sequence": _canonical_string(cand.seq_arr),
        "utility": choice.utility,
        "relevance_raw": relevance_raw[choice.idx],
        "relevance_norm": relevance_scaled[choice.idx],
        "nearest_selected_id": choice.nearest_id,
        "nearest_distance": choice.nearest_distance,
        "nearest_similarity": choice.nearest_similarity,
        "nearest_distance_full": choice.nearest_full_distance,
        "nearest_distance_core": choice.nearest_core_distance,
        "nearest_distance_core_bp": choice.nearest_core_bp,
    }
    for tf, core in core_maps[cand_id].items():
        row[f"core_{tf}"] = SequenceState(core).to_string()
    meta_rows.append(row)


def _resolve_mmr_pool(
    *,
    candidates: Sequence[MmrCandidate],
    k: int,
    pool_size: int,
    relevance: str,
    dsdna: bool,
) -> list[MmrCandidate]:
    candidates_sorted = sorted(
        candidates,
        key=lambda cand: (
            cand.min_norm if relevance == "min_tf_score" else cand.combined_score,
            _candidate_id(cand),
        ),
        reverse=True,
    )
    effective_pool_size = max(pool_size, k)
    return _dedupe_pool(candidates_sorted[:effective_pool_size], dsdna=dsdna)


def _resolve_relevance_scores(
    *,
    pool: Sequence[MmrCandidate],
    relevance: str,
) -> tuple[list[float], list[float]]:
    relevance_raw: list[float] = []
    for cand in pool:
        if relevance == "joint_score":
            relevance_raw.append(float(cand.combined_score))
        else:
            relevance_raw.append(float(cand.min_norm))
    return relevance_raw, _percentile_ranks(relevance_raw)


def _resolve_weights_by_tf(
    *,
    tf_names: Sequence[str] | None,
    pwms: dict[str, PWM] | None,
    core_maps: dict[str, dict[str, np.ndarray]] | None,
) -> tuple[Sequence[str], dict[str, np.ndarray]]:
    tf_names_resolved: Sequence[str] = tf_names or []
    if tf_names is None or pwms is None:
        raise ValueError("tf_names and pwms are required for TFBS core distances.")
    if core_maps is None:
        raise ValueError("core_maps are required for TFBS core distances.")
    weights_by_tf: dict[str, np.ndarray] = {}
    for tf in tf_names_resolved:
        pwm = pwms.get(tf)
        if pwm is None:
            raise ValueError(f"Missing PWM for TF '{tf}'.")
        weights_by_tf[tf] = compute_position_weights(pwm)
    return tf_names_resolved, weights_by_tf


def _select_seed_index(
    *,
    pool: Sequence[MmrCandidate],
    relevance_raw: Sequence[float],
    relevance_scaled: Sequence[float],
) -> int:
    return max(
        range(len(pool)),
        key=lambda idx: (relevance_scaled[idx], relevance_raw[idx], _candidate_id(pool[idx])),
    )


def _evaluate_candidate_choice(
    *,
    idx: int,
    selected_indices: Sequence[int],
    id_by_index: Sequence[str],
    pair_metrics: Callable[[int, int], tuple[float, float, float, int, int]],
    alpha: float,
    relevance_scaled: Sequence[float],
    min_hamming_bp: int | None,
    min_core_hamming_bp: int | None,
) -> _MmrChoice | None:
    nearest_distance: float | None = None
    nearest_id: str | None = None
    nearest_full_distance: float | None = None
    nearest_core_distance: float | None = None
    nearest_core_bp: int | None = None
    for sel_idx in selected_indices:
        metric_distance, full_distance, core_distance, core_bp, full_bp = pair_metrics(idx, sel_idx)
        if min_hamming_bp is not None and full_bp < min_hamming_bp:
            return None
        if min_core_hamming_bp is not None and core_bp < min_core_hamming_bp:
            return None
        if nearest_distance is None or metric_distance < nearest_distance:
            nearest_distance = float(metric_distance)
            nearest_id = id_by_index[sel_idx]
        nearest_full_distance = (
            float(full_distance) if nearest_full_distance is None else min(nearest_full_distance, float(full_distance))
        )
        nearest_core_distance = (
            float(core_distance) if nearest_core_distance is None else min(nearest_core_distance, float(core_distance))
        )
        nearest_core_bp = int(core_bp) if nearest_core_bp is None else min(nearest_core_bp, int(core_bp))

    nearest_distance_value = float(nearest_distance if nearest_distance is not None else 0.0)
    nearest_full_value = float(nearest_full_distance if nearest_full_distance is not None else 0.0)
    nearest_core_value = float(nearest_core_distance if nearest_core_distance is not None else 0.0)
    nearest_core_bp_value = int(nearest_core_bp if nearest_core_bp is not None else 0)
    nearest_similarity = 1.0 - nearest_distance_value
    utility = alpha * relevance_scaled[idx] - (1.0 - alpha) * nearest_similarity
    return _MmrChoice(
        idx=idx,
        utility=float(utility),
        nearest_id=nearest_id,
        nearest_distance=nearest_distance_value,
        nearest_similarity=float(nearest_similarity),
        nearest_full_distance=nearest_full_value,
        nearest_core_distance=nearest_core_value,
        nearest_core_bp=nearest_core_bp_value,
    )


def _is_better_choice(
    *,
    candidate: _MmrChoice,
    current_best: _MmrChoice | None,
    relevance_raw: Sequence[float],
    id_by_index: Sequence[str],
) -> bool:
    if current_best is None:
        return True
    if candidate.utility != current_best.utility:
        return candidate.utility > current_best.utility
    candidate_relevance = relevance_raw[candidate.idx]
    best_relevance = relevance_raw[current_best.idx]
    if candidate_relevance != best_relevance:
        return candidate_relevance > best_relevance
    if candidate.nearest_full_distance != current_best.nearest_full_distance:
        return candidate.nearest_full_distance > current_best.nearest_full_distance
    return id_by_index[candidate.idx] < id_by_index[current_best.idx]


def _relax_constraints(
    *,
    current_min_hamming_bp: int | None,
    current_min_core_hamming_bp: int | None,
    relax_step_bp: int,
    relax_min_bp: int,
) -> tuple[int | None, int | None, bool]:
    relaxed = False
    next_min_hamming_bp = current_min_hamming_bp
    next_min_core_hamming_bp = current_min_core_hamming_bp
    if next_min_hamming_bp is not None and next_min_hamming_bp > relax_min_bp:
        next_min_hamming_bp = max(relax_min_bp, next_min_hamming_bp - relax_step_bp)
        relaxed = True
    if next_min_core_hamming_bp is not None and next_min_core_hamming_bp > relax_min_bp:
        next_min_core_hamming_bp = max(relax_min_bp, next_min_core_hamming_bp - relax_step_bp)
        relaxed = True
    return next_min_hamming_bp, next_min_core_hamming_bp, relaxed


def _run_mmr_selection(
    *,
    pool: Sequence[MmrCandidate],
    k: int,
    alpha: float,
    relevance_raw: Sequence[float],
    relevance_scaled: Sequence[float],
    id_by_index: Sequence[str],
    core_maps: dict[str, dict[str, np.ndarray]],
    pair_metrics: Callable[[int, int], tuple[float, float, float, int, int]],
    policy: str,
    min_hamming_bp: int | None,
    min_core_hamming_bp: int | None,
    relax_step_bp: int,
    relax_min_bp: int,
) -> tuple[list[MmrCandidate], list[int], list[dict[str, object]], int | None, int | None, int]:
    selected: list[MmrCandidate] = []
    selected_indices: list[int] = []
    selected_ids: set[str] = set()
    meta_rows: list[dict[str, object]] = []

    seed_idx = _select_seed_index(
        pool=pool,
        relevance_raw=relevance_raw,
        relevance_scaled=relevance_scaled,
    )
    seed = pool[seed_idx]
    seed_id = _candidate_id(seed)
    seed_meta = _seed_meta_row(
        seed=seed,
        seed_idx=seed_idx,
        seed_id=seed_id,
        alpha=alpha,
        relevance_raw=relevance_raw,
        relevance_scaled=relevance_scaled,
        core_maps=core_maps,
    )
    selected.append(seed)
    selected_indices.append(seed_idx)
    selected_ids.add(seed_id)
    meta_rows.append(seed_meta)

    current_min_hamming_bp = int(min_hamming_bp) if min_hamming_bp is not None else None
    current_min_core_hamming_bp = int(min_core_hamming_bp) if min_core_hamming_bp is not None else None
    relax_steps_used = 0

    while len(selected) < k:
        remaining_indices = _remaining_candidate_indices(pool=pool, selected_ids=selected_ids)
        if not remaining_indices:
            break
        best_choice = _best_choice_for_remaining(
            remaining_indices=remaining_indices,
            selected_indices=selected_indices,
            id_by_index=id_by_index,
            pair_metrics=pair_metrics,
            alpha=alpha,
            relevance_raw=relevance_raw,
            relevance_scaled=relevance_scaled,
            min_hamming_bp=current_min_hamming_bp,
            min_core_hamming_bp=current_min_core_hamming_bp,
        )
        if best_choice is None:
            if current_min_hamming_bp is None and current_min_core_hamming_bp is None:
                break
            if policy == "strict":
                raise ValueError(
                    "Strict constrained MMR could not select "
                    f"{k} elites with min_hamming_bp={current_min_hamming_bp} "
                    f"and min_core_hamming_bp={current_min_core_hamming_bp}; "
                    f"stopped at {len(selected)}."
                )
            current_min_hamming_bp, current_min_core_hamming_bp, relaxed = _relax_constraints(
                current_min_hamming_bp=current_min_hamming_bp,
                current_min_core_hamming_bp=current_min_core_hamming_bp,
                relax_step_bp=relax_step_bp,
                relax_min_bp=relax_min_bp,
            )
            if not relaxed:
                break
            relax_steps_used += 1
            continue
        _append_choice_selection(
            choice=best_choice,
            pool=pool,
            id_by_index=id_by_index,
            core_maps=core_maps,
            alpha=alpha,
            relevance_raw=relevance_raw,
            relevance_scaled=relevance_scaled,
            selected=selected,
            selected_indices=selected_indices,
            selected_ids=selected_ids,
            meta_rows=meta_rows,
        )
    return (
        selected,
        selected_indices,
        meta_rows,
        current_min_hamming_bp,
        current_min_core_hamming_bp,
        relax_steps_used,
    )


def _build_pair_metrics(
    *,
    pool: Sequence[MmrCandidate],
    id_by_index: Sequence[str],
    core_maps: dict[str, dict[str, np.ndarray]],
    tf_names_resolved: Sequence[str],
    weights_by_tf: dict[str, np.ndarray],
    resolved_metric: str,
    hybrid_full_weight: float,
    hybrid_core_weight: float,
) -> Callable[[int, int], tuple[float, float, float, int, int]]:
    pair_metric_cache: dict[tuple[int, int], tuple[float, float, float, int, int]] = {}

    def _pair_metrics(
        left_idx: int,
        right_idx: int,
    ) -> tuple[float, float, float, int, int]:
        if left_idx == right_idx:
            return 0.0, 0.0, 0.0, 0, 0
        a_idx, b_idx = (left_idx, right_idx) if left_idx < right_idx else (right_idx, left_idx)
        cache_key = (a_idx, b_idx)
        cached = pair_metric_cache.get(cache_key)
        if cached is not None:
            return cached

        cand_a = pool[a_idx]
        cand_b = pool[b_idx]
        if cand_a.seq_arr.size != cand_b.seq_arr.size:
            raise ValueError("MMR candidates must share sequence length for distance comparisons.")
        cand_a_id = id_by_index[a_idx]
        cand_b_id = id_by_index[b_idx]
        core_a = core_maps[cand_a_id]
        core_b = core_maps[cand_b_id]
        core_dist = compute_core_distance(core_a, core_b, weights=weights_by_tf, tf_names=tf_names_resolved)
        full_bp = _full_sequence_distance_bp(cand_a.seq_arr, cand_b.seq_arr)
        full_dist = float(full_bp) / float(cand_a.seq_arr.size)
        core_bp = _core_hamming_bp(core_a, core_b, tf_names=tf_names_resolved)
        if resolved_metric == "full":
            metric_dist = full_dist
        elif resolved_metric == "hybrid":
            denom = hybrid_full_weight + hybrid_core_weight
            metric_dist = ((hybrid_full_weight * full_dist) + (hybrid_core_weight * core_dist)) / denom
        else:
            metric_dist = core_dist
        payload = (float(metric_dist), float(full_dist), float(core_dist), int(core_bp), int(full_bp))
        pair_metric_cache[cache_key] = payload
        return payload

    return _pair_metrics


def _summarize_selected_distances(
    *,
    selected_indices: Sequence[int],
    pair_metrics: Callable[[int, int], tuple[float, float, float, int, int]],
) -> tuple[float | None, float | None]:
    pairwise: list[float] = []
    for idx, left_idx in enumerate(selected_indices):
        for right_idx in selected_indices[idx + 1 :]:
            metric_distance, _, _, _, _ = pair_metrics(left_idx, right_idx)
            pairwise.append(metric_distance)
    mean_pairwise_distance = float(np.mean(pairwise)) if pairwise else None
    min_pairwise_distance = float(np.min(pairwise)) if pairwise else None
    return mean_pairwise_distance, min_pairwise_distance


def _selected_relevance_values(
    *,
    selected: Sequence[MmrCandidate],
    index_by_id: dict[str, int],
    relevance_raw: Sequence[float],
) -> list[float]:
    values: list[float] = []
    for item in selected:
        idx = index_by_id[_candidate_id(item)]
        values.append(relevance_raw[idx])
    return values


def select_mmr_elites(
    candidates: Sequence[MmrCandidate],
    *,
    k: int,
    pool_size: int,
    alpha: float,
    relevance: str,
    dsdna: bool,
    tf_names: Sequence[str] | None = None,
    pwms: dict[str, PWM] | None = None,
    core_maps: dict[str, dict[str, np.ndarray]] | None = None,
    distance_metric: str | None = None,
    min_hamming_bp: int | None = None,
    min_core_hamming_bp: int | None = None,
    constraint_policy: str = "relax",
    relax_step_bp: int = 1,
    relax_min_bp: int = 0,
    hybrid_full_weight: float = 1.0,
    hybrid_core_weight: float = 2.0,
) -> MmrSelectionResult:
    if k < 0:
        raise ValueError("k must be >= 0")
    if pool_size < 1:
        raise ValueError("pool_size must be >= 1")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0, 1]")
    if k == 0 or not candidates:
        return _empty_mmr_result(
            pool_size=pool_size,
            k=k,
            alpha=alpha,
            min_hamming_bp_requested=min_hamming_bp,
            min_hamming_bp_final=min_hamming_bp,
            min_core_hamming_bp_requested=min_core_hamming_bp,
            min_core_hamming_bp_final=min_core_hamming_bp,
            constraint_policy=constraint_policy,
        )

    resolved_metric, policy = _validate_mmr_runtime_inputs(
        relevance=relevance,
        distance_metric=distance_metric,
        constraint_policy=constraint_policy,
        min_hamming_bp=min_hamming_bp,
        min_core_hamming_bp=min_core_hamming_bp,
        relax_step_bp=relax_step_bp,
        relax_min_bp=relax_min_bp,
        hybrid_full_weight=hybrid_full_weight,
        hybrid_core_weight=hybrid_core_weight,
    )

    pool = _resolve_mmr_pool(
        candidates=candidates,
        k=k,
        pool_size=pool_size,
        relevance=relevance,
        dsdna=dsdna,
    )
    if not pool:
        return _empty_mmr_result(
            pool_size=pool_size,
            k=k,
            alpha=alpha,
            min_hamming_bp_requested=min_hamming_bp,
            min_hamming_bp_final=min_hamming_bp,
            min_core_hamming_bp_requested=min_core_hamming_bp,
            min_core_hamming_bp_final=min_core_hamming_bp,
            constraint_policy=policy,
        )

    relevance_raw, relevance_scaled = _resolve_relevance_scores(pool=pool, relevance=relevance)
    tf_names_resolved, weights_by_tf = _resolve_weights_by_tf(
        tf_names=tf_names,
        pwms=pwms,
        core_maps=core_maps,
    )
    assert core_maps is not None

    index_by_id = {_candidate_id(cand): idx for idx, cand in enumerate(pool)}
    id_by_index = [_candidate_id(cand) for cand in pool]
    _pair_metrics = _build_pair_metrics(
        pool=pool,
        id_by_index=id_by_index,
        core_maps=core_maps,
        tf_names_resolved=tf_names_resolved,
        weights_by_tf=weights_by_tf,
        resolved_metric=resolved_metric,
        hybrid_full_weight=hybrid_full_weight,
        hybrid_core_weight=hybrid_core_weight,
    )

    (
        selected,
        selected_indices,
        meta_rows,
        current_min_hamming_bp,
        current_min_core_hamming_bp,
        relax_steps_used,
    ) = _run_mmr_selection(
        pool=pool,
        k=k,
        alpha=alpha,
        relevance_raw=relevance_raw,
        relevance_scaled=relevance_scaled,
        id_by_index=id_by_index,
        core_maps=core_maps,
        pair_metrics=_pair_metrics,
        policy=policy,
        min_hamming_bp=min_hamming_bp,
        min_core_hamming_bp=min_core_hamming_bp,
        relax_step_bp=relax_step_bp,
        relax_min_bp=relax_min_bp,
    )

    mean_pairwise_distance, min_pairwise_distance = _summarize_selected_distances(
        selected_indices=selected_indices,
        pair_metrics=_pair_metrics,
    )
    selected_relevance = _selected_relevance_values(
        selected=selected,
        index_by_id=index_by_id,
        relevance_raw=relevance_raw,
    )
    median_relevance_raw = float(np.median(selected_relevance)) if selected_relevance else None

    return MmrSelectionResult(
        selected=selected,
        meta=meta_rows,
        pool_size=pool_size,
        k=k,
        alpha=alpha,
        median_relevance_raw=median_relevance_raw,
        mean_pairwise_distance=mean_pairwise_distance,
        min_pairwise_distance=min_pairwise_distance,
        min_hamming_bp_requested=min_hamming_bp,
        min_hamming_bp_final=current_min_hamming_bp,
        min_core_hamming_bp_requested=min_core_hamming_bp,
        min_core_hamming_bp_final=current_min_core_hamming_bp,
        constraint_policy=policy,
        relax_steps_used=relax_steps_used,
    )
