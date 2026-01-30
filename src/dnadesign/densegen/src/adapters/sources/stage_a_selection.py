"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a_selection.py

Stage-A selection and core deduplication helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np

from .stage_a_encoding import CoreEncodingStore, encode_cores
from .stage_a_types import SelectionMeta


class _CandidateLike(Protocol):
    seq: str
    score: float
    matched_sequence: Optional[str]


@dataclass(frozen=True)
class SelectionDiagnostics:
    shortlist_k: int
    shortlist_target: int
    shortlist_target_met: bool
    tier_fraction_used: float | None
    tier_limit: int
    pool_source: str

    def __post_init__(self) -> None:
        if int(self.shortlist_k) < 0:
            raise ValueError("Selection shortlist_k must be >= 0.")
        if int(self.shortlist_target) < 0:
            raise ValueError("Selection shortlist_target must be >= 0.")
        if int(self.tier_limit) < 0:
            raise ValueError("Selection tier_limit must be >= 0.")
        if self.pool_source not in {"shortlist_k", "tier_limit", "eligible_unique"}:
            raise ValueError(
                f"Selection pool_source must be shortlist_k/tier_limit/eligible_unique, got {self.pool_source}."
            )

    def pool_size(self) -> int:
        if self.pool_source == "shortlist_k":
            return int(self.shortlist_k)
        if self.pool_source == "tier_limit":
            return int(self.tier_limit)
        return int(self.tier_limit)

    def to_dict(self) -> dict[str, object]:
        return {
            "shortlist_k": int(self.shortlist_k),
            "shortlist_target": int(self.shortlist_target),
            "shortlist_target_met": bool(self.shortlist_target_met),
            "tier_fraction_used": float(self.tier_fraction_used) if self.tier_fraction_used is not None else None,
            "tier_limit": int(self.tier_limit),
            "pool_source": str(self.pool_source),
        }


def _core_sequence(candidate: _CandidateLike) -> str:
    if candidate.matched_sequence:
        return str(candidate.matched_sequence)
    raise ValueError("FIMO matched_sequence is required to derive the TFBS core.")


def _collapse_by_core_identity(ranked: Sequence[_CandidateLike]) -> tuple[list[_CandidateLike], int]:
    best_by_core: dict[str, _CandidateLike] = {}
    for cand in ranked:
        core = _core_sequence(cand)
        prev = best_by_core.get(core)
        if prev is None or cand.score > prev.score or (cand.score == prev.score and cand.seq < prev.seq):
            best_by_core[core] = cand
    collapsed = max(0, len(ranked) - len(best_by_core))
    return sorted(best_by_core.values(), key=lambda cand: (-cand.score, cand.seq)), int(collapsed)


def _pwm_tolerant_weights(matrix: Sequence[dict[str, float]]) -> np.ndarray:
    if not matrix:
        raise ValueError("PWM matrix is required to compute PWM-tolerant weights.")
    weights: list[float] = []
    for row in matrix:
        probs = [float(row.get(base, 0.0)) for base in ("A", "C", "G", "T")]
        if any(p < 0 for p in probs):
            raise ValueError("PWM probabilities must be >= 0 to compute information content.")
        total = float(sum(probs))
        if total <= 0:
            raise ValueError("PWM probabilities must sum to > 0 to compute information content.")
        probs = [p / total for p in probs]
        entropy = float(-sum(p * np.log2(p) for p in probs if p > 0))
        info_bits = 2.0 - entropy
        info_norm = min(1.0, max(0.0, info_bits / 2.0))
        weights.append(1.0 - info_norm)
    return np.asarray(weights, dtype=float)


def _score_norm(values: Sequence[float]) -> dict[float, float]:
    if not values:
        return {}
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 1:
        return {vals[0]: 1.0}
    if min(vals) == max(vals):
        return {float(v): 1.0 for v in vals}
    sorted_vals = sorted(vals)
    bounds: dict[float, list[int]] = {}
    for idx, val in enumerate(sorted_vals):
        bucket = bounds.get(val)
        if bucket is None:
            bounds[val] = [idx, idx]
        else:
            bucket[1] = idx
    norm: dict[float, float] = {}
    denom = float(n - 1)
    for val, (first, last) in bounds.items():
        avg_rank = (float(first) + float(last)) / 2.0 + 1.0
        norm[float(val)] = (avg_rank - 1.0) / denom if denom > 0 else 1.0
    return norm


def _similarity_from_distance(distance: float) -> float:
    return 1.0 / (1.0 + float(distance))


def _select_diversity_baseline_candidates(
    ranked: Sequence[_CandidateLike],
    *,
    selection_policy: str,
    selection_diag: SelectionDiagnostics | None,
    n_sites: int,
) -> list[_CandidateLike]:
    candidate_slice = _select_diversity_candidate_pool(
        ranked,
        selection_policy=selection_policy,
        selection_diag=selection_diag,
    )
    target_n = min(int(n_sites), len(candidate_slice))
    return candidate_slice[:target_n]


def _select_diversity_global_candidates(
    ranked: Sequence[_CandidateLike],
    *,
    n_sites: int,
) -> list[_CandidateLike]:
    target_n = min(int(n_sites), len(ranked))
    return list(ranked[:target_n])


def _select_diversity_candidate_pool(
    ranked: Sequence[_CandidateLike],
    *,
    selection_policy: str,
    selection_diag: SelectionDiagnostics | None,
) -> list[_CandidateLike]:
    candidate_slice: list[_CandidateLike] = list(ranked)
    if selection_policy == "mmr" and selection_diag is not None:
        tier_limit = int(selection_diag.tier_limit)
        if tier_limit > 0:
            candidate_slice = candidate_slice[:tier_limit]
        shortlist_k = int(selection_diag.shortlist_k)
        if shortlist_k > 0:
            candidate_slice = candidate_slice[:shortlist_k]
    return candidate_slice


def _select_diversity_upper_bound_candidates(
    ranked: Sequence[_CandidateLike],
    *,
    selection_policy: str,
    selection_diag: SelectionDiagnostics | None,
    n_sites: int,
    weights: Sequence[float] | None = None,
    encoding_store: CoreEncodingStore | None = None,
) -> list[_CandidateLike]:
    candidate_slice = _select_diversity_candidate_pool(
        ranked,
        selection_policy=selection_policy,
        selection_diag=selection_diag,
    )
    if not candidate_slice or n_sites <= 0:
        return []
    target_n = min(int(n_sites), len(candidate_slice))
    if target_n <= 0:
        return []
    cores = [_core_sequence(cand) for cand in candidate_slice]
    length = len(cores[0])
    if any(len(core) != length for core in cores):
        raise ValueError("Core length mismatch in max-diversity selection pool.")
    weights_arr = None
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape[0] != length:
            raise ValueError("Weighted Hamming requires weights matching core length.")
    else:
        weights_arr = np.ones(length, dtype=float)
    encoded = encoding_store.encode(cores) if encoding_store is not None else encode_cores(cores)
    scores = [float(cand.score) for cand in candidate_slice]
    seqs = [cand.seq for cand in candidate_slice]

    freqs = np.zeros((length, 4), dtype=float)
    for base_idx in range(4):
        freqs[:, base_idx] = (encoded == base_idx).mean(axis=0)
    expected = np.zeros(len(candidate_slice), dtype=float)
    for idx in range(len(candidate_slice)):
        base_freq = freqs[np.arange(length), encoded[idx]]
        expected[idx] = float(np.sum(weights_arr * (1.0 - base_freq)))

    def _pick_best(indices: Sequence[int], values: np.ndarray) -> int:
        best_idx = None
        best_val = None
        best_score = None
        best_core = None
        best_seq = None
        for idx in indices:
            val = float(values[idx])
            score = float(scores[idx])
            core = cores[idx]
            seq = seqs[idx]
            if best_idx is None:
                best_idx = idx
                best_val = val
                best_score = score
                best_core = core
                best_seq = seq
                continue
            if val > float(best_val):
                pass
            elif val == float(best_val):
                if score > float(best_score):
                    pass
                elif score == float(best_score):
                    if core < str(best_core):
                        pass
                    elif core == str(best_core):
                        if seq >= str(best_seq):
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                continue
            best_idx = idx
            best_val = val
            best_score = score
            best_core = core
            best_seq = seq
        if best_idx is None:
            raise ValueError("Max-diversity selection could not choose a seed.")
        return int(best_idx)

    all_indices = list(range(len(candidate_slice)))
    seed_idx = _pick_best(all_indices, expected)
    selected_indices: list[int] = []
    selected_mask = np.zeros(len(candidate_slice), dtype=bool)
    min_dist = np.full(len(candidate_slice), np.inf, dtype=float)

    def _update_dist(idx: int) -> None:
        diff = encoded != encoded[idx]
        dist = (diff * weights_arr).sum(axis=1)
        min_dist[:] = np.minimum(min_dist, dist)
        min_dist[idx] = 0.0

    selected_indices.append(seed_idx)
    selected_mask[seed_idx] = True
    _update_dist(seed_idx)
    while len(selected_indices) < target_n:
        remaining = [idx for idx in all_indices if not selected_mask[idx]]
        if not remaining:
            break
        next_idx = _pick_best(remaining, min_dist)
        selected_indices.append(next_idx)
        selected_mask[next_idx] = True
        _update_dist(next_idx)
    return [candidate_slice[idx] for idx in selected_indices]


def _select_by_mmr(
    ranked: Sequence[_CandidateLike],
    *,
    matrix: Sequence[dict[str, float]],
    n_sites: int,
    alpha: float,
    shortlist_min: int,
    shortlist_factor: int,
    shortlist_max: Optional[int],
    tier_widening: Optional[Sequence[float]],
    encoding_store: CoreEncodingStore | None = None,
) -> tuple[list[_CandidateLike], dict[str, SelectionMeta], SelectionDiagnostics]:
    if not ranked or n_sites <= 0:
        return (
            [],
            {},
            SelectionDiagnostics(
                shortlist_k=0,
                shortlist_target=0,
                shortlist_target_met=False,
                tier_fraction_used=None,
                tier_limit=0,
                pool_source="tier_limit",
            ),
        )
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("selection.alpha must be in (0, 1].")
    core_by_seq = {cand.seq: _core_sequence(cand) for cand in ranked}
    weights = _pwm_tolerant_weights(matrix)
    tier_fractions = list(tier_widening) if tier_widening else [1.0]
    total = len(ranked)
    shortlist_target = max(int(shortlist_min), int(shortlist_factor) * int(n_sites))

    def _lex_ranks(values: Sequence[str]) -> np.ndarray:
        order = {val: idx for idx, val in enumerate(sorted(set(values)))}
        return np.array([order[val] for val in values], dtype=int)

    def _select_from_slice(
        candidates: Sequence[_CandidateLike],
    ) -> tuple[list[_CandidateLike], dict[str, SelectionMeta], int]:
        if not candidates:
            return [], {}, 0
        max_k = len(candidates) if shortlist_max is None else min(len(candidates), int(shortlist_max))
        k = min(max_k, max(1, int(shortlist_target)))
        if k <= 0:
            k = min(len(candidates), int(n_sites))
        shortlist = list(candidates[:k])
        target_n = min(int(n_sites), len(shortlist))
        if target_n <= 0:
            return [], {}, int(k)
        scores_arr = np.array([float(cand.score) for cand in shortlist], dtype=float)
        scores_norm_map = _score_norm(scores_arr.tolist())
        scores_norm = np.array([scores_norm_map.get(float(cand.score), 1.0) for cand in shortlist], dtype=float)
        score_weight = float(alpha)
        diversity_weight = 1.0 - score_weight
        cores = [core_by_seq[cand.seq] for cand in shortlist]
        if len(weights) != len(cores[0]):
            raise ValueError("PWM weights length must match TFBS core length.")
        encoded = encoding_store.encode(cores) if encoding_store is not None else encode_cores(cores)
        seqs = [cand.seq for cand in shortlist]
        core_ranks = _lex_ranks(cores)
        seq_ranks = _lex_ranks(seqs)

        def _run_mmr() -> tuple[list[_CandidateLike], dict[str, SelectionMeta]]:
            selected_mask = np.zeros(len(shortlist), dtype=bool)
            min_dist = np.full(len(shortlist), np.inf, dtype=float)
            selected: list[_CandidateLike] = []
            meta: dict[str, SelectionMeta] = {}
            any_selected = False
            while len(selected) < target_n:
                if not any_selected:
                    max_sim = np.zeros(len(shortlist), dtype=float)
                else:
                    max_sim = 1.0 / (1.0 + min_dist)
                utility = score_weight * scores_norm - diversity_weight * max_sim
                utility[selected_mask] = -np.inf
                best_val = float(np.max(utility))
                candidate_idx = np.flatnonzero(utility == best_val)
                if candidate_idx.size == 0:
                    raise ValueError("MMR selection failed to identify a candidate.")
                if candidate_idx.size == 1:
                    best_idx = int(candidate_idx[0])
                else:
                    cand_scores = scores_arr[candidate_idx]
                    cand_core = core_ranks[candidate_idx]
                    cand_seq = seq_ranks[candidate_idx]
                    order = np.lexsort((cand_seq, cand_core, -cand_scores))
                    best_idx = int(candidate_idx[order[0]])
                chosen = shortlist[best_idx]
                selected.append(chosen)
                selected_mask[best_idx] = True
                any_selected = True
                meta[chosen.seq] = SelectionMeta(
                    selection_rank=len(selected),
                    selection_utility=float(utility[best_idx]),
                    nearest_selected_similarity=float(max_sim[best_idx]),
                )
                dists = ((encoded != encoded[best_idx]) * weights).sum(axis=1)
                min_dist = np.minimum(min_dist, dists)
                min_dist[best_idx] = 0.0
            return selected, meta

        selected, meta = _run_mmr()
        if len(selected) < int(n_sites) and k < max_k:
            shortlist = list(candidates[:max_k])
            target_n = min(int(n_sites), len(shortlist))
            scores_arr = np.array([float(cand.score) for cand in shortlist], dtype=float)
            scores_norm_map = _score_norm(scores_arr.tolist())
            scores_norm = np.array([scores_norm_map.get(float(cand.score), 1.0) for cand in shortlist], dtype=float)
            cores = [core_by_seq[cand.seq] for cand in shortlist]
            if len(weights) != len(cores[0]):
                raise ValueError("PWM weights length must match TFBS core length.")
            encoded = encoding_store.encode(cores) if encoding_store is not None else encode_cores(cores)
            seqs = [cand.seq for cand in shortlist]
            core_ranks = _lex_ranks(cores)
            seq_ranks = _lex_ranks(seqs)
            selected, meta = _run_mmr()
            return selected, meta, int(max_k)
        return selected, meta, int(k)

    last_selected: list[_CandidateLike] = []
    last_meta: dict[str, dict] = {}
    last_shortlist = 0
    fraction_used = None
    tier_limit = total
    shortlist_target_met = False
    widen_for_shortlist = bool(tier_widening)
    for fraction in tier_fractions:
        if fraction <= 0:
            continue
        tier_limit = min(total, max(1, int(np.floor(float(fraction) * total))))
        subset = ranked[:tier_limit]
        selected, meta, shortlist_k = _select_from_slice(subset)
        last_selected = selected
        last_meta = meta
        last_shortlist = shortlist_k
        fraction_used = float(fraction)
        shortlist_target_met = int(shortlist_k) >= int(shortlist_target)
        if len(selected) >= int(n_sites):
            if widen_for_shortlist and not shortlist_target_met:
                continue
            break
    pool_source = "shortlist_k" if int(last_shortlist) > 0 else "tier_limit"
    diag = SelectionDiagnostics(
        shortlist_k=int(last_shortlist),
        shortlist_target=int(shortlist_target),
        shortlist_target_met=bool(shortlist_target_met),
        tier_fraction_used=fraction_used,
        tier_limit=int(tier_limit),
        pool_source=pool_source,
    )
    return last_selected, last_meta, diag
