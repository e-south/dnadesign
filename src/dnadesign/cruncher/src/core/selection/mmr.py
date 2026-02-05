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
from dnadesign.cruncher.core.sequence import canon_int, dsdna_hamming, hamming_distance, revcomp_int
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


def compute_sequence_distance(a: np.ndarray, b: np.ndarray, *, dsdna: bool) -> float:
    if dsdna:
        dist = dsdna_hamming(a, b)
    else:
        dist = hamming_distance(a, b)
    denom = max(int(len(a)), int(len(b)), 1)
    return float(dist) / float(denom)


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
        distances.append(float(np.sum(w * mismatches) / denom))
    return float(np.mean(distances))


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
    for tf in tf_names:
        raw_llr, offset, strand = scorer.best_llr(seq_arr, tf)
        _ = raw_llr
        width = scorer.pwm_width(tf)
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
) -> list[float]:
    if len(selected) < 2:
        return []
    distances: list[float] = []
    for idx, cand_a in enumerate(selected):
        for cand_b in selected[idx + 1 :]:
            core_a = core_maps[_candidate_id(cand_a)]
            core_b = core_maps[_candidate_id(cand_b)]
            dist = compute_core_distance(core_a, core_b, weights=weights_by_tf, tf_names=tf_names)
            distances.append(dist)
    return distances


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
) -> MmrSelectionResult:
    if k < 0:
        raise ValueError("k must be >= 0")
    if pool_size < 1:
        raise ValueError("pool_size must be >= 1")
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be in (0, 1]")
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
        )

    candidates_sorted = sorted(
        candidates,
        key=lambda cand: (
            cand.min_norm if relevance == "min_per_tf_norm" else cand.combined_score,
            _candidate_id(cand),
        ),
        reverse=True,
    )
    effective_pool_size = max(pool_size, k)
    pool = candidates_sorted[:effective_pool_size]

    deduped: list[MmrCandidate] = []
    seen: set[str] = set()
    for cand in pool:
        key = _canonical_string(cand.seq_arr) if dsdna else _sequence_string(cand.seq_arr)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    pool = deduped

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
        )

    relevance_raw: list[float] = []
    for cand in pool:
        if relevance == "combined_score_final":
            relevance_raw.append(float(cand.combined_score))
        elif relevance == "min_per_tf_norm":
            relevance_raw.append(float(cand.min_norm))
        else:
            raise ValueError(f"Unknown relevance '{relevance}'.")

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
        best_idx = None
        best_utility = None
        best_nearest = None
        best_nearest_id = None
        for idx, cand in enumerate(pool):
            cand_id = _candidate_id(cand)
            if cand_id in selected_ids:
                continue
            core_cand = core_maps[_candidate_id(cand)]
            distances = []
            nearest_id = None
            nearest_dist = None
            for sel in selected:
                dist = compute_core_distance(
                    core_cand,
                    core_maps[_candidate_id(sel)],
                    weights=weights_by_tf,
                    tf_names=tf_names_resolved,
                )
                distances.append(dist)
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = _candidate_id(sel)
            nearest_distance = min(distances) if distances else 0.0
            nearest_similarity = 1.0 - nearest_distance
            utility = alpha * relevance_scaled[idx] - (1.0 - alpha) * nearest_similarity
            if best_utility is None or utility > best_utility:
                best_utility = utility
                best_idx = idx
                best_nearest = (nearest_distance, nearest_similarity)
                best_nearest_id = nearest_id
            elif utility == best_utility and best_idx is not None:
                if relevance_raw[idx] > relevance_raw[best_idx]:
                    best_idx = idx
                    best_nearest = (nearest_distance, nearest_similarity)
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
    )
