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

from ...core.score_tiers import normalize_tier_fractions
from .stage_a_encoding import CoreEncodingStore, encode_cores
from .stage_a_sampling_utils import normalize_background
from .stage_a_types import SelectionMeta


class _CandidateLike(Protocol):
    seq: str
    score: float
    matched_sequence: Optional[str]


@dataclass(frozen=True)
class SelectionDiagnostics:
    selection_pool_size_final: int
    selection_pool_rung_fraction_used: float | None
    selection_pool_min_score_norm_used: float | None
    selection_pool_capped: bool
    selection_pool_cap_value: int | None

    def __post_init__(self) -> None:
        if int(self.selection_pool_size_final) < 0:
            raise ValueError("Selection pool size must be >= 0.")
        if self.selection_pool_rung_fraction_used is not None:
            value = float(self.selection_pool_rung_fraction_used)
            if value <= 0.0 or value > 1.0:
                raise ValueError("Selection pool rung fraction must be in (0, 1].")
        if self.selection_pool_min_score_norm_used is not None:
            value = float(self.selection_pool_min_score_norm_used)
            if value <= 0.0 or value > 1.0:
                raise ValueError("Selection pool min score norm must be in (0, 1].")
        if self.selection_pool_cap_value is not None and int(self.selection_pool_cap_value) <= 0:
            raise ValueError("Selection pool cap value must be > 0 when set.")

    def pool_size(self) -> int:
        return int(self.selection_pool_size_final)

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_pool_size_final": int(self.selection_pool_size_final),
            "selection_pool_rung_fraction_used": (
                float(self.selection_pool_rung_fraction_used)
                if self.selection_pool_rung_fraction_used is not None
                else None
            ),
            "selection_pool_min_score_norm_used": (
                float(self.selection_pool_min_score_norm_used)
                if self.selection_pool_min_score_norm_used is not None
                else None
            ),
            "selection_pool_capped": bool(self.selection_pool_capped),
            "selection_pool_cap_value": int(self.selection_pool_cap_value) if self.selection_pool_cap_value else None,
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


def _pwm_tolerant_weights(
    matrix: Sequence[dict[str, float]],
    *,
    background: dict[str, float],
) -> np.ndarray:
    if not matrix:
        raise ValueError("PWM matrix is required to compute PWM-tolerant weights.")
    bg = normalize_background(background)
    bg_vals = np.array([float(bg.get(base, 0.0)) for base in ("A", "C", "G", "T")], dtype=float)
    if np.any(bg_vals <= 0):
        raise ValueError("Background probabilities must be > 0 to compute relative entropy.")
    max_info_bits = float(np.log2(1.0 / float(bg_vals.min())))
    if max_info_bits <= 0.0:
        raise ValueError("Background probabilities must yield positive information content.")
    weights: list[float] = []
    for row in matrix:
        probs = np.array([float(row.get(base, 0.0)) for base in ("A", "C", "G", "T")], dtype=float)
        if np.any(probs < 0):
            raise ValueError("PWM probabilities must be >= 0 to compute information content.")
        total = float(probs.sum())
        if total <= 0:
            raise ValueError("PWM probabilities must sum to > 0 to compute information content.")
        probs = probs / total
        mask = probs > 0
        info_bits = float(np.sum(probs[mask] * np.log2(probs[mask] / bg_vals[mask])))
        info_norm = min(1.0, max(0.0, info_bits / max_info_bits))
        weights.append(1.0 - info_norm)
    weights_arr = np.asarray(weights, dtype=float)
    if float(weights_arr.sum()) <= 1e-6:
        weights_arr = np.ones(len(weights_arr), dtype=float)
    return weights_arr


def _score_percentile_norm(values: Sequence[float]) -> dict[float, float]:
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


def _select_diversity_top_candidates(
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
        pool_size = int(selection_diag.selection_pool_size_final)
        if pool_size > 0:
            candidate_slice = candidate_slice[:pool_size]
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
    background: dict[str, float],
    n_sites: int,
    alpha: float,
    pool_min_score_norm: float | None,
    pool_max_candidates: int | None,
    relevance_norm: str,
    tier_fractions: Optional[Sequence[float]],
    pwm_theoretical_max_score: float | None,
    encoding_store: CoreEncodingStore | None = None,
) -> tuple[list[_CandidateLike], dict[str, SelectionMeta], SelectionDiagnostics]:
    if not ranked or n_sites <= 0:
        return (
            [],
            {},
            SelectionDiagnostics(
                selection_pool_size_final=0,
                selection_pool_rung_fraction_used=None,
                selection_pool_min_score_norm_used=None,
                selection_pool_capped=False,
                selection_pool_cap_value=None,
            ),
        )
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("selection.alpha must be in (0, 1].")
    relevance_norm = str(relevance_norm or "minmax_raw_score").lower()
    if relevance_norm not in {"percentile", "minmax_raw_score"}:
        raise ValueError("selection.pool.relevance_norm must be 'percentile' or 'minmax_raw_score'.")
    pool_min_score_norm_value = float(pool_min_score_norm) if pool_min_score_norm is not None else None
    if pool_min_score_norm_value is not None:
        if pool_min_score_norm_value <= 0.0 or pool_min_score_norm_value > 1.0:
            raise ValueError("selection.pool.min_score_norm must be in (0, 1].")
    pool_max_candidates_value = int(pool_max_candidates) if pool_max_candidates is not None else None
    if pool_max_candidates_value is not None and pool_max_candidates_value <= 0:
        raise ValueError("selection.pool.max_candidates must be > 0 when set.")
    core_by_seq = {cand.seq: _core_sequence(cand) for cand in ranked}
    weights = _pwm_tolerant_weights(matrix, background=background)
    weight_sum = float(weights.sum())
    tier_fractions = list(normalize_tier_fractions(tier_fractions))
    ladder = list(tier_fractions)
    if not ladder or ladder[-1] < 1.0:
        ladder.append(1.0)
    total = len(ranked)
    if pwm_theoretical_max_score is None:
        raise ValueError("pwm_theoretical_max_score is required for score normalization in Stage-A MMR.")
    score_norm_denominator = float(pwm_theoretical_max_score)
    if score_norm_denominator <= 0.0:
        raise ValueError("pwm_theoretical_max_score must be > 0 for Stage-A MMR score normalization.")

    def _lex_ranks(values: Sequence[str]) -> np.ndarray:
        order = {val: idx for idx, val in enumerate(sorted(set(values)))}
        return np.array([order[val] for val in values], dtype=int)

    def _select_from_pool(
        candidates: Sequence[_CandidateLike],
    ) -> tuple[list[_CandidateLike], dict[str, SelectionMeta]]:
        if not candidates:
            return [], {}
        target_n = min(int(n_sites), len(candidates))
        if target_n <= 0:
            return [], {}
        scores_arr = np.array([float(cand.score) for cand in candidates], dtype=float)
        if relevance_norm == "percentile":
            scores_norm_map = _score_percentile_norm(scores_arr.tolist())
            scores_norm = np.array([scores_norm_map.get(float(cand.score), 1.0) for cand in candidates], dtype=float)
        else:
            scores_norm = np.array([float(cand.score) / score_norm_denominator for cand in candidates], dtype=float)
        score_norm_for_meta = np.array([float(cand.score) / score_norm_denominator for cand in candidates], dtype=float)
        score_weight = float(alpha)
        diversity_weight = 1.0 - score_weight
        cores = [core_by_seq[cand.seq] for cand in candidates]
        if len(weights) != len(cores[0]):
            raise ValueError("PWM weights length must match TFBS core length.")
        encoded = encoding_store.encode(cores) if encoding_store is not None else encode_cores(cores)
        seqs = [cand.seq for cand in candidates]
        core_ranks = _lex_ranks(cores)
        seq_ranks = _lex_ranks(seqs)

        def _run_mmr() -> tuple[list[_CandidateLike], dict[str, SelectionMeta]]:
            selected_mask = np.zeros(len(candidates), dtype=bool)
            min_dist = np.full(len(candidates), np.inf, dtype=float)
            selected: list[_CandidateLike] = []
            meta: dict[str, SelectionMeta] = {}
            any_selected = False
            while len(selected) < target_n:
                if not any_selected:
                    max_sim = np.zeros(len(candidates), dtype=float)
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
                chosen = candidates[best_idx]
                selected.append(chosen)
                selected_mask[best_idx] = True
                had_selected = any_selected
                nearest_distance = None
                nearest_distance_norm = None
                if had_selected:
                    nearest_distance = float(min_dist[best_idx])
                    nearest_distance_norm = float(nearest_distance / weight_sum) if weight_sum > 0 else None
                meta[chosen.seq] = SelectionMeta(
                    selection_rank=len(selected),
                    selection_utility=float(utility[best_idx]),
                    selection_score_norm=float(score_norm_for_meta[best_idx]),
                    nearest_selected_similarity=float(max_sim[best_idx]) if had_selected else None,
                    nearest_selected_distance=nearest_distance,
                    nearest_selected_distance_norm=nearest_distance_norm,
                )
                any_selected = True
                dists = ((encoded != encoded[best_idx]) * weights).sum(axis=1)
                min_dist = np.minimum(min_dist, dists)
                min_dist[best_idx] = 0.0
            return selected, meta

        selected, meta = _run_mmr()
        return selected, meta

    pool_candidates: list[_CandidateLike] = []
    fraction_used = None
    for fraction in ladder:
        if float(fraction) <= 0:
            continue
        tier_limit = min(total, max(1, int(np.ceil(float(fraction) * total))))
        subset = list(ranked[:tier_limit])
        pool_candidates = subset
        fraction_used = float(fraction)
        if len(pool_candidates) >= int(n_sites):
            break
    if fraction_used is None:
        fraction_used = 1.0
        pool_candidates = list(ranked)
    pool_capped = False
    pool_cap_value = None
    if pool_max_candidates_value is not None and len(pool_candidates) > pool_max_candidates_value:
        pool_capped = True
        pool_cap_value = int(pool_max_candidates_value)
        pool_candidates = sorted(
            pool_candidates,
            key=lambda cand: (-float(cand.score), core_by_seq[cand.seq], cand.seq),
        )[:pool_max_candidates_value]
    if len(pool_candidates) <= int(n_sites):
        import logging

        details = []
        if fraction_used is not None:
            details.append(f"rung={float(fraction_used) * 100:.3f}%")
        if pool_capped and pool_cap_value is not None:
            details.append(f"cap={int(pool_cap_value)}")
        detail_label = f" ({', '.join(details)})" if details else ""
        logging.getLogger(__name__).warning(
            "Stage-A MMR degenerate: pool size %d <= n_sites=%d%s; returning pool in score order.",
            len(pool_candidates),
            int(n_sites),
            detail_label,
        )
    if pool_max_candidates_value is None and len(pool_candidates) > 100_000:
        import logging

        logging.getLogger(__name__).warning(
            "Stage-A MMR pool size is %d candidates; consider setting selection.pool.max_candidates.",
            len(pool_candidates),
        )
    selected, meta = _select_from_pool(pool_candidates)
    diag = SelectionDiagnostics(
        selection_pool_size_final=int(len(pool_candidates)),
        selection_pool_rung_fraction_used=fraction_used,
        selection_pool_min_score_norm_used=pool_min_score_norm_value,
        selection_pool_capped=bool(pool_capped),
        selection_pool_cap_value=int(pool_cap_value) if pool_cap_value is not None else None,
    )
    return selected, meta, diag
