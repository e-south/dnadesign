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
from typing import Sequence

import numpy as np

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.sequence import canon_int, revcomp_int
from dnadesign.cruncher.core.state import SequenceState


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


def compute_position_weights(pwm: PWM) -> np.ndarray:
    matrix = np.asarray(pwm.matrix, dtype=float)
    p = matrix + 1.0e-9
    info = 2.0 + np.sum(p * np.log2(p), axis=1)
    min_info = float(np.min(info))
    max_info = float(np.max(info))
    if max_info - min_info <= 0:
        info_norm = np.zeros_like(info, dtype=float)
    else:
        info_norm = (info - min_info) / (max_info - min_info)
    return 1.0 - info_norm


def compute_core_distance(
    cores_a: dict[str, np.ndarray],
    cores_b: dict[str, np.ndarray],
    *,
    weights: dict[str, np.ndarray],
    tf_names: Sequence[str],
) -> float:
    if not tf_names:
        return 0.0
    distances: list[float] = []
    for tf in tf_names:
        core_a = cores_a[tf]
        core_b = cores_b[tf]
        w = weights[tf]
        if core_a.shape != core_b.shape or core_a.shape != w.shape:
            raise ValueError(f"Core/weight shape mismatch for TF '{tf}'.")
        mismatches = (core_a != core_b).astype(float)
        denom = float(np.sum(w))
        if denom <= 0:
            raise ValueError(f"Non-positive weight sum for TF '{tf}'.")
        tf_distance = float(np.sum(w * mismatches) / denom)
        if tf_distance < -1.0e-12 or tf_distance > 1.0 + 1.0e-12:
            raise ValueError(f"TF core distance for '{tf}' must be in [0, 1], got {tf_distance}.")
        distances.append(float(np.clip(tf_distance, 0.0, 1.0)))
    value = float(np.mean(distances))
    if value < -1.0e-12 or value > 1.0 + 1.0e-12:
        raise ValueError(f"Core distance must be in [0, 1], got {value}.")
    return float(np.clip(value, 0.0, 1.0))


def core_from_hit(seq_arr: np.ndarray, *, offset: int, width: int, strand: str) -> np.ndarray:
    if width < 1:
        raise ValueError("core_from_hit requires width >= 1")
    if offset < 0:
        raise ValueError("core_from_hit requires offset >= 0")
    window = np.asarray(seq_arr, dtype=np.int8)[offset : offset + width]
    if window.size != width:
        raise ValueError("core_from_hit window is out of bounds for sequence length")
    if strand == "-":
        return revcomp_int(window)
    return window


def tfbs_cores_from_hits(
    seq_arr: np.ndarray,
    *,
    per_tf_hits: dict[str, dict[str, object]],
    tf_names: Sequence[str],
) -> dict[str, np.ndarray]:
    cores: dict[str, np.ndarray] = {}
    for tf in tf_names:
        hit = per_tf_hits.get(tf)
        if not isinstance(hit, dict):
            raise ValueError(f"Missing TF hit data for '{tf}'.")
        offset = hit.get("offset")
        width = hit.get("width")
        strand = hit.get("strand")
        if not isinstance(offset, int) or not isinstance(width, int) or not isinstance(strand, str):
            raise ValueError(f"Invalid TF hit data for '{tf}'.")
        cores[tf] = core_from_hit(seq_arr, offset=offset, width=width, strand=strand)
    return cores


def tfbs_cores_from_scorer(
    seq_arr: np.ndarray,
    *,
    scorer: object,
    tf_names: Sequence[str],
) -> dict[str, np.ndarray]:
    cores: dict[str, np.ndarray] = {}
    seq_length = int(np.asarray(seq_arr).size)
    if seq_length < 1:
        raise ValueError("Core extraction requires non-empty sequences.")
    for tf in tf_names:
        raw_llr, offset, strand = scorer.best_llr(seq_arr, tf)
        _ = raw_llr
        width = int(scorer.pwm_width(tf))
        if width > seq_length:
            width = seq_length
            offset = 0
        if strand == "-":
            rev = revcomp_int(seq_arr)
            core = rev[offset : offset + width]
        else:
            core = seq_arr[offset : offset + width]
        if core.size != width:
            raise ValueError(f"Core extraction failed for '{tf}'.")
        cores[tf] = core
    return cores


def _candidate_id(candidate: MmrCandidate) -> str:
    return f"{int(candidate.chain_id)}:{int(candidate.draw_idx)}"


def _sequence_string(seq_arr: np.ndarray) -> str:
    return SequenceState(seq_arr).to_string()


def _canonical_string(seq_arr: np.ndarray) -> str:
    return SequenceState(canon_int(seq_arr)).to_string()


def _full_sequence_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.int8)
    b_arr = np.asarray(b, dtype=np.int8)
    if a_arr.shape != b_arr.shape:
        raise ValueError("Full-sequence distance requires equal-length sequence arrays.")
    if a_arr.size == 0:
        raise ValueError("Full-sequence distance requires non-empty sequence arrays.")
    value = float(np.count_nonzero(a_arr != b_arr)) / float(a_arr.size)
    if value < -1.0e-12 or value > 1.0 + 1.0e-12:
        raise ValueError(f"Full-sequence distance must be in [0, 1], got {value}.")
    return float(np.clip(value, 0.0, 1.0))


def _full_sequence_distance_bp(a: np.ndarray, b: np.ndarray) -> int:
    a_arr = np.asarray(a, dtype=np.int8)
    b_arr = np.asarray(b, dtype=np.int8)
    if a_arr.shape != b_arr.shape:
        raise ValueError("Full-sequence distance requires equal-length sequence arrays.")
    if a_arr.size == 0:
        raise ValueError("Full-sequence distance requires non-empty sequence arrays.")
    return int(np.count_nonzero(a_arr != b_arr))


def _core_hamming_bp(
    cores_a: dict[str, np.ndarray],
    cores_b: dict[str, np.ndarray],
    *,
    tf_names: Sequence[str],
) -> int:
    total = 0
    for tf in tf_names:
        core_a = np.asarray(cores_a[tf], dtype=np.int8)
        core_b = np.asarray(cores_b[tf], dtype=np.int8)
        if core_a.shape != core_b.shape:
            raise ValueError(f"Core shape mismatch for TF '{tf}'.")
        total += int(np.count_nonzero(core_a != core_b))
    return total


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
        return MmrSelectionResult(
            selected=[],
            meta=[],
            pool_size=pool_size,
            k=k,
            alpha=alpha,
            median_relevance_raw=None,
            mean_pairwise_distance=None,
            min_pairwise_distance=None,
            min_hamming_bp_requested=min_hamming_bp,
            min_hamming_bp_final=min_hamming_bp,
            min_core_hamming_bp_requested=min_core_hamming_bp,
            min_core_hamming_bp_final=min_core_hamming_bp,
            constraint_policy=constraint_policy,
            relax_steps_used=0,
        )

    if relevance not in {"min_tf_score", "joint_score"}:
        raise ValueError(f"Unknown relevance '{relevance}'.")
    resolved_metric = str(distance_metric or "core").strip().lower()
    if resolved_metric not in {"full", "core", "hybrid"}:
        raise ValueError(f"Unknown distance_metric '{distance_metric}'.")
    policy = str(constraint_policy).strip().lower()
    if policy not in {"relax", "strict"}:
        raise ValueError("constraint_policy must be 'relax' or 'strict'.")
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

    candidates_sorted = sorted(
        candidates,
        key=lambda cand: (
            cand.min_norm if relevance == "min_tf_score" else cand.combined_score,
            _candidate_id(cand),
        ),
        reverse=True,
    )
    effective_pool_size = max(pool_size, k)
    pool = candidates_sorted[:effective_pool_size]

    pool = _dedupe_pool(pool, dsdna=dsdna)

    if not pool:
        return MmrSelectionResult(
            selected=[],
            meta=[],
            pool_size=pool_size,
            k=k,
            alpha=alpha,
            median_relevance_raw=None,
            mean_pairwise_distance=None,
            min_pairwise_distance=None,
            min_hamming_bp_requested=min_hamming_bp,
            min_hamming_bp_final=min_hamming_bp,
            min_core_hamming_bp_requested=min_core_hamming_bp,
            min_core_hamming_bp_final=min_core_hamming_bp,
            constraint_policy=policy,
            relax_steps_used=0,
        )

    relevance_raw: list[float] = []
    for cand in pool:
        if relevance == "joint_score":
            relevance_raw.append(float(cand.combined_score))
        elif relevance == "min_tf_score":
            relevance_raw.append(float(cand.min_norm))

    relevance_scaled = _percentile_ranks(relevance_raw)

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

    selected: list[MmrCandidate] = []
    selected_ids: set[str] = set()
    meta_rows: list[dict[str, object]] = []
    index_by_id = {_candidate_id(cand): idx for idx, cand in enumerate(pool)}
    current_min_hamming_bp = int(min_hamming_bp) if min_hamming_bp is not None else None
    current_min_core_hamming_bp = int(min_core_hamming_bp) if min_core_hamming_bp is not None else None
    relax_steps_used = 0

    seed_idx = max(
        range(len(pool)),
        key=lambda idx: (relevance_scaled[idx], relevance_raw[idx], _candidate_id(pool[idx])),
    )
    seed = pool[seed_idx]
    seed_id = _candidate_id(seed)
    seed_meta = {
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
    if core_maps is not None:
        for tf, core in core_maps[seed_id].items():
            seed_meta[f"core_{tf}"] = SequenceState(core).to_string()
    selected.append(seed)
    selected_ids.add(seed_id)
    meta_rows.append(seed_meta)

    while len(selected) < k:
        remaining_indices = [idx for idx, cand in enumerate(pool) if _candidate_id(cand) not in selected_ids]
        if not remaining_indices:
            break

        feasible_indices: list[int] = []
        for idx in remaining_indices:
            cand = pool[idx]
            cand_id = _candidate_id(cand)
            core_cand = core_maps[cand_id]
            feasible = True
            for sel in selected:
                sel_id = _candidate_id(sel)
                if current_min_hamming_bp is not None:
                    full_bp = _full_sequence_distance_bp(cand.seq_arr, sel.seq_arr)
                    if full_bp < current_min_hamming_bp:
                        feasible = False
                        break
                if current_min_core_hamming_bp is not None:
                    core_bp = _core_hamming_bp(core_cand, core_maps[sel_id], tf_names=tf_names_resolved)
                    if core_bp < current_min_core_hamming_bp:
                        feasible = False
                        break
            if feasible:
                feasible_indices.append(idx)

        if not feasible_indices:
            if current_min_hamming_bp is None and current_min_core_hamming_bp is None:
                break
            if policy == "strict":
                raise ValueError(
                    "Strict constrained MMR could not select "
                    f"{k} elites with min_hamming_bp={current_min_hamming_bp} "
                    f"and min_core_hamming_bp={current_min_core_hamming_bp}; "
                    f"stopped at {len(selected)}."
                )
            relaxed = False
            if current_min_hamming_bp is not None and current_min_hamming_bp > relax_min_bp:
                current_min_hamming_bp = max(relax_min_bp, current_min_hamming_bp - relax_step_bp)
                relaxed = True
            if current_min_core_hamming_bp is not None and current_min_core_hamming_bp > relax_min_bp:
                current_min_core_hamming_bp = max(relax_min_bp, current_min_core_hamming_bp - relax_step_bp)
                relaxed = True
            if not relaxed:
                break
            relax_steps_used += 1
            continue

        best_idx = None
        best_utility = None
        best_nearest = None
        best_nearest_full = None
        best_nearest_core = None
        best_nearest_core_bp = None
        best_nearest_id = None
        for idx in feasible_indices:
            cand = pool[idx]
            cand_id = _candidate_id(cand)
            core_cand = core_maps[_candidate_id(cand)]
            distances: list[float] = []
            full_distances: list[float] = []
            core_distances: list[float] = []
            core_bp_distances: list[int] = []
            nearest_id = None
            nearest_dist = None
            for sel in selected:
                sel_id = _candidate_id(sel)
                dist = compute_core_distance(
                    core_cand,
                    core_maps[sel_id],
                    weights=weights_by_tf,
                    tf_names=tf_names_resolved,
                )
                full_dist = _full_sequence_distance(cand.seq_arr, sel.seq_arr)
                if resolved_metric == "full":
                    metric_distance = full_dist
                elif resolved_metric == "hybrid":
                    denom = hybrid_full_weight + hybrid_core_weight
                    metric_distance = ((hybrid_full_weight * full_dist) + (hybrid_core_weight * dist)) / denom
                else:
                    metric_distance = dist
                distances.append(float(metric_distance))
                full_distances.append(float(full_dist))
                core_distances.append(float(dist))
                core_bp_distances.append(_core_hamming_bp(core_cand, core_maps[sel_id], tf_names=tf_names_resolved))
                if nearest_dist is None or metric_distance < nearest_dist:
                    nearest_dist = float(metric_distance)
                    nearest_id = sel_id
            nearest_distance = min(distances) if distances else 0.0
            nearest_full_distance = min(full_distances) if full_distances else 0.0
            nearest_core_distance = min(core_distances) if core_distances else 0.0
            nearest_core_bp = min(core_bp_distances) if core_bp_distances else 0
            nearest_similarity = 1.0 - nearest_distance
            utility = alpha * relevance_scaled[idx] - (1.0 - alpha) * nearest_similarity
            if best_utility is None or utility > best_utility:
                best_utility = utility
                best_idx = idx
                best_nearest = (nearest_distance, nearest_similarity)
                best_nearest_full = nearest_full_distance
                best_nearest_core = nearest_core_distance
                best_nearest_core_bp = nearest_core_bp
                best_nearest_id = nearest_id
            elif utility == best_utility and best_idx is not None:
                if relevance_raw[idx] > relevance_raw[best_idx]:
                    best_idx = idx
                    best_nearest = (nearest_distance, nearest_similarity)
                    best_nearest_full = nearest_full_distance
                    best_nearest_core = nearest_core_distance
                    best_nearest_core_bp = nearest_core_bp
                    best_nearest_id = nearest_id
                elif relevance_raw[idx] == relevance_raw[best_idx]:
                    current_best_full = best_nearest_full if best_nearest_full is not None else -1.0
                    if nearest_full_distance > current_best_full:
                        best_idx = idx
                        best_nearest = (nearest_distance, nearest_similarity)
                        best_nearest_full = nearest_full_distance
                        best_nearest_core = nearest_core_distance
                        best_nearest_core_bp = nearest_core_bp
                        best_nearest_id = nearest_id
                    elif nearest_full_distance == current_best_full:
                        best_id = _candidate_id(pool[best_idx])
                        if cand_id < best_id:
                            best_idx = idx
                            best_nearest = (nearest_distance, nearest_similarity)
                            best_nearest_full = nearest_full_distance
                            best_nearest_core = nearest_core_distance
                            best_nearest_core_bp = nearest_core_bp
                            best_nearest_id = nearest_id

        if best_idx is None:
            break

        cand = pool[best_idx]
        cand_id = _candidate_id(cand)
        selected.append(cand)
        selected_ids.add(cand_id)
        nearest_distance, nearest_similarity = best_nearest if best_nearest else (None, None)
        row = {
            "elite_rank": len(selected),
            "candidate_id": cand_id,
            "sequence": _sequence_string(cand.seq_arr),
            "canonical_sequence": _canonical_string(cand.seq_arr),
            "utility": best_utility,
            "relevance_raw": relevance_raw[best_idx],
            "relevance_norm": relevance_scaled[best_idx],
            "nearest_selected_id": best_nearest_id,
            "nearest_distance": nearest_distance,
            "nearest_similarity": nearest_similarity,
            "nearest_distance_full": best_nearest_full,
            "nearest_distance_core": best_nearest_core,
            "nearest_distance_core_bp": best_nearest_core_bp,
        }
        if core_maps is not None:
            for tf, core in core_maps[cand_id].items():
                row[f"core_{tf}"] = SequenceState(core).to_string()
        meta_rows.append(row)

    pairwise = _pairwise_distances(
        selected,
        tf_names=tf_names_resolved,
        weights_by_tf=weights_by_tf,
        core_maps=core_maps,
        distance_metric=resolved_metric,
        hybrid_full_weight=hybrid_full_weight,
        hybrid_core_weight=hybrid_core_weight,
    )
    selected_relevance = []
    for sel in selected:
        idx = index_by_id[_candidate_id(sel)]
        selected_relevance.append(relevance_raw[idx])
    median_relevance_raw = float(np.median(selected_relevance)) if selected_relevance else None
    mean_pairwise_distance = float(np.mean(pairwise)) if pairwise else None
    min_pairwise_distance = float(np.min(pairwise)) if pairwise else None

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
