"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a_selection.py

Stage-A selection and core deduplication helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

import numpy as np


class _CandidateLike(Protocol):
    seq: str
    score: float
    matched_sequence: Optional[str]


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


def _contrib_vector(core: str, log_odds: Sequence[dict[str, float]]) -> np.ndarray:
    if len(core) != len(log_odds):
        raise ValueError("TFBS core length must match PWM log-odds length.")
    contrib = []
    for base, row in zip(core, log_odds):
        val = float(row.get(base, float("nan")))
        if not np.isfinite(val):
            raise ValueError("Non-finite log-odds contribution encountered for core sequence.")
        contrib.append(val)
    return np.asarray(contrib, dtype=float)


def _score_norm(values: Sequence[float]) -> dict[float, float]:
    if not values:
        return {}
    lo = float(min(values))
    hi = float(max(values))
    if hi == lo:
        return {float(v): 1.0 for v in values}
    return {float(v): (float(v) - lo) / (hi - lo) for v in values}


def _similarity_from_distance(distance: float) -> float:
    return 1.0 / (1.0 + float(distance))


def _select_diversity_baseline_candidates(
    ranked: Sequence[_CandidateLike],
    *,
    selection_policy: str,
    selection_diag: dict | None,
    n_sites: int,
) -> list[_CandidateLike]:
    candidate_slice: list[_CandidateLike] = list(ranked)
    if selection_policy == "mmr" and selection_diag is not None:
        tier_limit = selection_diag.get("tier_limit")
        if isinstance(tier_limit, int) and tier_limit > 0:
            candidate_slice = candidate_slice[:tier_limit]
        shortlist_k = selection_diag.get("shortlist_k")
        if isinstance(shortlist_k, int) and shortlist_k > 0:
            candidate_slice = candidate_slice[:shortlist_k]
    target_n = min(int(n_sites), len(candidate_slice))
    return candidate_slice[:target_n]


def _select_diversity_global_candidates(
    ranked: Sequence[_CandidateLike],
    *,
    n_sites: int,
) -> list[_CandidateLike]:
    target_n = min(int(n_sites), len(ranked))
    return list(ranked[:target_n])


def _select_by_mmr(
    ranked: Sequence[_CandidateLike],
    *,
    log_odds: Sequence[dict[str, float]],
    n_sites: int,
    alpha: float,
    shortlist_min: int,
    shortlist_factor: int,
    shortlist_max: Optional[int],
    tier_widening: Optional[Sequence[float]],
    ensure_shortlist_target: bool = False,
) -> tuple[list[_CandidateLike], dict[str, dict], dict]:
    if not ranked or n_sites <= 0:
        return [], {}, {"shortlist_k": 0, "shortlist_target": 0, "tier_fraction_used": None, "tier_limit": 0}
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("selection.alpha must be in (0, 1].")
    core_by_seq = {cand.seq: _core_sequence(cand) for cand in ranked}
    contrib_by_seq = {seq: _contrib_vector(core, log_odds) for seq, core in core_by_seq.items()}
    scores = [cand.score for cand in ranked]
    norm_by_score = _score_norm(scores)
    tier_fractions = list(tier_widening) if tier_widening else [1.0]
    total = len(ranked)
    shortlist_target = max(int(shortlist_min), int(shortlist_factor) * int(n_sites))

    def _best_candidate(
        shortlist: Sequence[_CandidateLike],
        *,
        selected_mask: np.ndarray,
        min_dist: np.ndarray,
        cores: Sequence[str],
        seqs: Sequence[str],
        scores_norm: np.ndarray,
    ) -> tuple[int, float, float]:
        best_idx = None
        best_utility = None
        best_score = None
        best_core = None
        best_seq = None
        best_sim = None
        any_selected = bool(np.any(selected_mask))
        for idx, cand in enumerate(shortlist):
            if selected_mask[idx]:
                continue
            if not any_selected:
                max_sim = 0.0
            else:
                max_sim = _similarity_from_distance(float(min_dist[idx]))
            utility = float(alpha * scores_norm[idx] - (1.0 - alpha) * max_sim)
            cand_core = cores[idx]
            cand_seq = seqs[idx]
            cand_score = float(cand.score)
            if best_idx is None:
                best_idx = idx
                best_utility = utility
                best_score = cand_score
                best_core = cand_core
                best_seq = cand_seq
                best_sim = max_sim
                continue
            if utility > float(best_utility):
                pass
            elif utility == float(best_utility):
                if cand_score > float(best_score):
                    pass
                elif cand_score == float(best_score):
                    if cand_core < str(best_core):
                        pass
                    elif cand_core == str(best_core):
                        if cand_seq >= str(best_seq):
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                continue
            best_idx = idx
            best_utility = utility
            best_score = cand_score
            best_core = cand_core
            best_seq = cand_seq
            best_sim = max_sim
        if best_idx is None or best_utility is None or best_sim is None:
            raise ValueError("MMR selection failed to identify a candidate.")
        return int(best_idx), float(best_utility), float(best_sim)

    def _select_from_slice(candidates: Sequence[_CandidateLike]) -> tuple[list[_CandidateLike], dict[str, dict], int]:
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
        vectors = np.vstack([contrib_by_seq[cand.seq] for cand in shortlist])
        scores_norm = np.array([norm_by_score.get(float(cand.score), 1.0) for cand in shortlist], dtype=float)
        cores = [core_by_seq[cand.seq] for cand in shortlist]
        seqs = [cand.seq for cand in shortlist]
        selected_mask = np.zeros(len(shortlist), dtype=bool)
        min_dist = np.full(len(shortlist), np.inf, dtype=float)
        selected: list[_CandidateLike] = []
        meta: dict[str, dict] = {}
        while len(selected) < target_n:
            idx, utility, max_sim = _best_candidate(
                shortlist,
                selected_mask=selected_mask,
                min_dist=min_dist,
                cores=cores,
                seqs=seqs,
                scores_norm=scores_norm,
            )
            chosen = shortlist[idx]
            selected.append(chosen)
            selected_mask[idx] = True
            meta[chosen.seq] = {
                "selection_rank": len(selected),
                "selection_utility": float(utility),
                "nearest_selected_similarity": float(max_sim),
            }
            vec = vectors[idx]
            dists = np.abs(vectors - vec).sum(axis=1)
            min_dist = np.minimum(min_dist, dists)
            min_dist[idx] = 0.0
        if len(selected) < int(n_sites) and k < max_k:
            shortlist = list(candidates[:max_k])
            target_n = min(int(n_sites), len(shortlist))
            vectors = np.vstack([contrib_by_seq[cand.seq] for cand in shortlist])
            scores_norm = np.array([norm_by_score.get(float(cand.score), 1.0) for cand in shortlist], dtype=float)
            cores = [core_by_seq[cand.seq] for cand in shortlist]
            seqs = [cand.seq for cand in shortlist]
            selected_mask = np.zeros(len(shortlist), dtype=bool)
            min_dist = np.full(len(shortlist), np.inf, dtype=float)
            selected = []
            meta = {}
            while len(selected) < target_n:
                idx, utility, max_sim = _best_candidate(
                    shortlist,
                    selected_mask=selected_mask,
                    min_dist=min_dist,
                    cores=cores,
                    seqs=seqs,
                    scores_norm=scores_norm,
                )
                chosen = shortlist[idx]
                selected.append(chosen)
                selected_mask[idx] = True
                meta[chosen.seq] = {
                    "selection_rank": len(selected),
                    "selection_utility": float(utility),
                    "nearest_selected_similarity": float(max_sim),
                }
                vec = vectors[idx]
                dists = np.abs(vectors - vec).sum(axis=1)
                min_dist = np.minimum(min_dist, dists)
                min_dist[idx] = 0.0
            return selected, meta, int(max_k)
        return selected, meta, int(k)

    last_selected: list[_CandidateLike] = []
    last_meta: dict[str, dict] = {}
    last_shortlist = 0
    fraction_used = None
    tier_limit = total
    shortlist_target_met = False
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
            if ensure_shortlist_target and not shortlist_target_met:
                continue
            break
    diag = {
        "shortlist_k": int(last_shortlist),
        "shortlist_target": int(shortlist_target),
        "shortlist_target_met": bool(shortlist_target_met),
        "tier_fraction_used": fraction_used,
        "tier_limit": int(tier_limit),
    }
    return last_selected, last_meta, diag
