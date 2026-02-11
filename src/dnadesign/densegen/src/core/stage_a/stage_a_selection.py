"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/stage_a/stage_a_selection.py

Stage-A selection and core deduplication helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np

from ..score_tiers import normalize_tier_fractions
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
    selection_pool_target_size: int | None = None
    selection_pool_degenerate: bool | None = None
    selection_score_norm_max_raw: float | None = None
    selection_score_norm_clipped: bool | None = None
    selection_pool_sequences: tuple[str, ...] | None = None

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
        if self.selection_pool_target_size is not None and int(self.selection_pool_target_size) <= 0:
            raise ValueError("Selection pool target size must be > 0 when set.")
        if self.selection_pool_target_size is not None and int(self.selection_pool_size_final) > int(
            self.selection_pool_target_size
        ):
            raise ValueError("Selection pool size cannot exceed target size.")
        if self.selection_score_norm_max_raw is not None:
            value = float(self.selection_score_norm_max_raw)
            if not np.isfinite(value):
                raise ValueError("Selection score norm max must be finite when set.")
        if self.selection_pool_sequences is not None:
            if len(self.selection_pool_sequences) != len(set(self.selection_pool_sequences)):
                raise ValueError("Selection pool sequences must be unique when set.")
            if any(not str(seq).strip() for seq in self.selection_pool_sequences):
                raise ValueError("Selection pool sequences must be non-empty when set.")

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
            "selection_pool_target_size": (
                int(self.selection_pool_target_size) if self.selection_pool_target_size is not None else None
            ),
            "selection_pool_degenerate": (
                bool(self.selection_pool_degenerate) if self.selection_pool_degenerate is not None else None
            ),
            "selection_score_norm_max_raw": (
                float(self.selection_score_norm_max_raw) if self.selection_score_norm_max_raw is not None else None
            ),
            "selection_score_norm_clipped": (
                bool(self.selection_score_norm_clipped) if self.selection_score_norm_clipped is not None else None
            ),
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
        if selection_diag.selection_pool_sequences:
            ranked_by_seq = {cand.seq: cand for cand in ranked}
            resolved = [ranked_by_seq[seq] for seq in selection_diag.selection_pool_sequences if seq in ranked_by_seq]
            if resolved:
                return resolved
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
    score_norm_denominator_by_seq: dict[str, float] | None = None,
    rank_by: str = "score",
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
    rank_by = str(rank_by or "score").lower()
    if rank_by not in {"score", "score_norm"}:
        raise ValueError("selection.rank_by must be 'score' or 'score_norm'.")
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
    ladder = [
        tier_fractions[0],
        tier_fractions[0] + tier_fractions[1],
        tier_fractions[0] + tier_fractions[1] + tier_fractions[2],
    ]
    if ladder[-1] < 1.0:
        ladder.append(1.0)
    total = len(ranked)
    score_norm_denominator_by_seq = (
        dict(score_norm_denominator_by_seq) if score_norm_denominator_by_seq is not None else None
    )
    if score_norm_denominator_by_seq is None:
        if pwm_theoretical_max_score is None:
            raise ValueError("pwm_theoretical_max_score is required for score normalization in Stage-A MMR.")
        score_norm_denominator = float(pwm_theoretical_max_score)
        if score_norm_denominator <= 0.0:
            raise ValueError("pwm_theoretical_max_score must be > 0 for Stage-A MMR score normalization.")

    def _lex_ranks(values: Sequence[str]) -> np.ndarray:
        order = {val: idx for idx, val in enumerate(sorted(set(values)))}
        return np.array([order[val] for val in values], dtype=int)

    score_norm_max_raw: float | None = None
    score_norm_clipped: bool | None = None

    def _score_norm_denominators(
        candidates: Sequence[_CandidateLike],
    ) -> tuple[np.ndarray, float, float]:
        if score_norm_denominator_by_seq is None:
            denom = float(score_norm_denominator)
            return np.full(len(candidates), denom, dtype=float), denom, denom
        values: list[float] = []
        for cand in candidates:
            denom = score_norm_denominator_by_seq.get(cand.seq)
            if denom is None:
                raise ValueError("score_norm_denominator_by_seq missing entry for candidate.")
            denom_val = float(denom)
            if denom_val <= 0.0:
                raise ValueError("score_norm_denominator_by_seq values must be > 0 for Stage-A MMR.")
            values.append(denom_val)
        arr = np.asarray(values, dtype=float)
        return arr, float(arr.min()), float(arr.max())

    def _select_from_pool(
        candidates: Sequence[_CandidateLike],
    ) -> tuple[list[_CandidateLike], dict[str, SelectionMeta]]:
        if not candidates:
            return [], {}
        target_n = min(int(n_sites), len(candidates))
        if target_n <= 0:
            return [], {}
        scores_arr = np.array([float(cand.score) for cand in candidates], dtype=float)
        denom_arr, denom_min, denom_max = _score_norm_denominators(candidates)
        if relevance_norm == "percentile":
            scores_norm_map = _score_percentile_norm(scores_arr.tolist())
            scores_norm = np.array([scores_norm_map.get(float(cand.score), 1.0) for cand in candidates], dtype=float)
        else:
            scores_norm = scores_arr / denom_arr
        score_norm_for_meta = scores_arr / denom_arr
        max_raw = float(np.nanmax(score_norm_for_meta)) if score_norm_for_meta.size else None
        if max_raw is not None:
            nonlocal score_norm_max_raw, score_norm_clipped
            score_norm_max_raw = max_raw
            score_norm_clipped = False
            if max_raw > 1.0 + 1e-6:
                score_norm_clipped = True
                logging.getLogger(__name__).error(
                    "Stage-A score_norm exceeded 1.0 (max_raw=%.6f, denom_range=[%.6f, %.6f]); clipping to 1.0.",
                    max_raw,
                    denom_min,
                    denom_max,
                )
        score_norm_for_meta = np.clip(score_norm_for_meta, 0.0, 1.0)
        if relevance_norm == "minmax_raw_score":
            scores_norm = np.clip(scores_norm, 0.0, 1.0)
        if target_n == len(candidates):
            selected = list(candidates)
            meta: dict[str, SelectionMeta] = {}
            for idx, cand in enumerate(selected):
                meta[cand.seq] = SelectionMeta(
                    selection_rank=idx + 1,
                    selection_utility=None,
                    selection_score_norm=float(score_norm_for_meta[idx]) if score_norm_for_meta.size > idx else None,
                    nearest_selected_similarity=None,
                    nearest_selected_distance=None,
                    nearest_selected_distance_norm=None,
                )
            return selected, meta
        score_weight = float(alpha)
        diversity_weight = 1.0 - score_weight
        cores = [core_by_seq[cand.seq] for cand in candidates]
        core_lengths = {len(core) for core in cores}
        if len(core_lengths) != 1:
            length_list = ", ".join(str(length) for length in sorted(core_lengths))
            raise ValueError(
                "Stage-A MMR requires a uniform core length; got lengths "
                f"[{length_list}]. Ensure pwm.sampling.length.range does not "
                "cross below the motif width or set pwm.sampling.trimming.window_length "
                "to a fixed length."
            )
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

    def _apply_pool_gate(candidates: Sequence[_CandidateLike]) -> list[_CandidateLike]:
        if pool_min_score_norm_value is None or not candidates:
            return list(candidates)
        scores_arr = np.array([float(cand.score) for cand in candidates], dtype=float)
        denom_arr, _, _ = _score_norm_denominators(candidates)
        score_norm = np.clip(scores_arr / denom_arr, 0.0, 1.0)
        keep_idx = np.flatnonzero(score_norm >= pool_min_score_norm_value)
        return [candidates[int(idx)] for idx in keep_idx]

    def _rank_pool_candidates(candidates: Sequence[_CandidateLike]) -> list[_CandidateLike]:
        if not candidates:
            return []
        if rank_by == "score_norm":
            scores_arr = np.array([float(cand.score) for cand in candidates], dtype=float)
            denom_arr, _, _ = _score_norm_denominators(candidates)
            scores_norm = np.clip(scores_arr / denom_arr, 0.0, 1.0)
            seqs = [cand.seq for cand in candidates]
            cores = [core_by_seq[cand.seq] for cand in candidates]
            order = np.lexsort((seqs, cores, -scores_arr, -scores_norm))
            return [candidates[int(idx)] for idx in order]
        return sorted(
            candidates,
            key=lambda cand: (-float(cand.score), core_by_seq[cand.seq], cand.seq),
        )

    target_pool = max(int(n_sites), int(np.ceil(10.0 * int(n_sites))))
    target_pool = min(target_pool, total)
    pool_capped = False
    pool_cap_value = int(pool_max_candidates_value) if pool_max_candidates_value is not None else None
    if pool_cap_value is not None and target_pool > pool_cap_value:
        pool_capped = True
        target_pool = min(target_pool, pool_cap_value)

    pool_candidates: list[_CandidateLike] = []
    fraction_used: float | None = None
    for fraction in ladder:
        if float(fraction) <= 0:
            continue
        tier_limit = min(total, max(1, int(np.ceil(float(fraction) * total))))
        subset = list(ranked[:tier_limit])
        pool_candidates = _apply_pool_gate(subset)
        fraction_used = float(fraction)
        if len(pool_candidates) >= target_pool:
            break
    if fraction_used is None:
        fraction_used = 1.0
        pool_candidates = _apply_pool_gate(list(ranked))

    pool_candidates = _rank_pool_candidates(pool_candidates)
    if len(pool_candidates) > target_pool:
        pool_candidates = pool_candidates[:target_pool]

    pool_degenerate = len(pool_candidates) <= int(n_sites)
    if pool_degenerate:
        logging.getLogger(__name__).warning(
            "Stage-A MMR degenerate: pool_size=%d n_sites=%d target_pool=%d "
            "rung=%.3f min_score_norm=%s capped=%s cap_value=%s",
            len(pool_candidates),
            int(n_sites),
            int(target_pool),
            float(fraction_used),
            float(pool_min_score_norm_value) if pool_min_score_norm_value is not None else None,
            bool(pool_capped),
            int(pool_cap_value) if pool_cap_value is not None else None,
        )
    selected, meta = _select_from_pool(pool_candidates)
    diag = SelectionDiagnostics(
        selection_pool_size_final=int(len(pool_candidates)),
        selection_pool_rung_fraction_used=fraction_used,
        selection_pool_min_score_norm_used=pool_min_score_norm_value,
        selection_pool_capped=bool(pool_capped),
        selection_pool_cap_value=int(pool_cap_value) if pool_cap_value is not None else None,
        selection_pool_target_size=int(target_pool),
        selection_pool_degenerate=bool(pool_degenerate),
        selection_score_norm_max_raw=score_norm_max_raw,
        selection_score_norm_clipped=score_norm_clipped,
        selection_pool_sequences=tuple(cand.seq for cand in pool_candidates),
    )
    return selected, meta, diag
