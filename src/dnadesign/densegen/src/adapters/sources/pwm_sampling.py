"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/pwm_sampling.py

Shared Stage-A PWM sampling utilities.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, TextIO, Tuple, Union

import numpy as np
import pandas as pd

from ...core.artifacts.ids import hash_candidate_id
from ...core.score_tiers import score_tier_counts
from ...core.stage_a_constants import FIMO_REPORT_THRESH
from ...utils import logging_utils

SMOOTHING_ALPHA = 1e-6
SCORE_HIST_BINS = 60
log = logging.getLogger(__name__)
_SAFE_LABEL_RE = None


def _format_rate(rate: float) -> str:
    if rate >= 1000.0:
        return f"{rate / 1000.0:.1f}k/s"
    return f"{rate:.1f}/s"


def _format_pwm_progress_line(
    *,
    motif_id: str,
    backend: str,
    generated: int,
    target: int,
    accepted: Optional[int],
    accepted_target: Optional[int],
    batch_index: Optional[int],
    batch_total: Optional[int],
    elapsed: float,
) -> str:
    safe_target = max(1, int(target))
    gen_pct = min(100, int(100 * generated / safe_target))
    parts = [f"PWM {motif_id}", backend, f"gen {gen_pct}% ({generated}/{safe_target})"]
    if batch_index is not None:
        total_label = "-" if batch_total is None else str(int(batch_total))
        parts.append(f"batch {int(batch_index)}/{total_label}")
    elapsed_label = f"{max(0.0, float(elapsed)):.1f}s"
    rate = generated / elapsed if elapsed > 0 else 0.0
    parts.append(elapsed_label)
    parts.append(_format_rate(rate))
    return " | ".join(parts)


@dataclass
class _PwmSamplingProgress:
    motif_id: str
    backend: str
    target: int
    accepted_target: Optional[int]
    stream: TextIO
    min_interval: float = 0.2

    def __post_init__(self) -> None:
        self._enabled = bool(logging_utils.is_progress_enabled()) and bool(
            getattr(self.stream, "isatty", lambda: False)()
        )
        self._start = time.monotonic()
        self._last_update = self._start
        self._last_len = 0
        self._last_state: tuple[int, Optional[int], Optional[int]] | None = None
        self._shown = False
        if self._enabled:
            logging_utils.set_progress_active(True)

    def update(
        self,
        *,
        generated: int,
        accepted: Optional[int],
        batch_index: Optional[int] = None,
        batch_total: Optional[int] = None,
        force: bool = False,
    ) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        if not force and (now - self._last_update) < float(self.min_interval):
            return
        state = (int(generated), batch_index, batch_total)
        if self._shown and state == self._last_state and logging_utils.is_progress_line_visible():
            self._last_update = now
            return
        line = _format_pwm_progress_line(
            motif_id=self.motif_id,
            backend=self.backend,
            generated=int(generated),
            target=int(self.target),
            accepted=accepted,
            accepted_target=self.accepted_target,
            batch_index=batch_index,
            batch_total=batch_total,
            elapsed=now - self._start,
        )
        padded = line.ljust(self._last_len)
        self._last_len = max(self._last_len, len(line))
        self.stream.write(f"\r{padded}")
        self.stream.flush()
        self._last_update = now
        self._last_state = state
        self._shown = True
        logging_utils.mark_progress_line_visible()

    def finish(self) -> None:
        if self._enabled:
            logging_utils.set_progress_active(False)
        if not self._shown:
            return
        self.stream.write("\n")
        self.stream.flush()


def _safe_label(text: str) -> str:
    global _SAFE_LABEL_RE
    if _SAFE_LABEL_RE is None:
        import re

        _SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_.-]+")
    cleaned = _SAFE_LABEL_RE.sub("_", str(text).strip())
    return cleaned or "motif"


def _mining_attr(mining, name: str, default=None):
    if mining is None:
        return default
    if hasattr(mining, name):
        return getattr(mining, name)
    if isinstance(mining, dict):
        return mining.get(name, default)
    return default


def _selection_attr(selection, name: str, default=None):
    if selection is None:
        return default
    if hasattr(selection, name):
        return getattr(selection, name)
    if isinstance(selection, dict):
        return selection.get(name, default)
    return default


def _budget_attr(mining, name: str, default=None):
    if mining is None:
        return default
    budget = None
    if hasattr(mining, "budget"):
        budget = getattr(mining, "budget")
    elif isinstance(mining, dict):
        budget = mining.get("budget")
    if budget is None:
        return default
    if hasattr(budget, name):
        return getattr(budget, name)
    if isinstance(budget, dict):
        return budget.get(name, default)
    return default


def _cfg_attr(obj, name: str, default=None):
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def sampling_kwargs_from_config(sampling: object) -> dict:
    mining = _cfg_attr(sampling, "mining")
    length_cfg = _cfg_attr(sampling, "length", {}) or {}
    trimming_cfg = _cfg_attr(sampling, "trimming", {}) or {}
    uniqueness_cfg = _cfg_attr(sampling, "uniqueness", {}) or {}
    return {
        "strategy": _cfg_attr(sampling, "strategy", "stochastic"),
        "n_sites": int(_cfg_attr(sampling, "n_sites")),
        "mining": mining,
        "bgfile": _cfg_attr(sampling, "bgfile"),
        "keep_all_candidates_debug": bool(_cfg_attr(sampling, "keep_all_candidates_debug", False)),
        "include_matched_sequence": bool(_cfg_attr(sampling, "include_matched_sequence", False)),
        "uniqueness_key": _cfg_attr(uniqueness_cfg, "key"),
        "selection": _cfg_attr(sampling, "selection"),
        "length_policy": _cfg_attr(length_cfg, "policy", "exact"),
        "length_range": _cfg_attr(length_cfg, "range"),
        "trim_window_length": _cfg_attr(trimming_cfg, "window_length"),
        "trim_window_strategy": _cfg_attr(trimming_cfg, "window_strategy", "max_info"),
    }


@dataclass(frozen=True)
class FimoCandidate:
    seq: str
    score: float
    start: int
    stop: int
    strand: str
    matched_sequence: Optional[str] = None


def _core_sequence(candidate: FimoCandidate) -> str:
    if candidate.matched_sequence:
        return str(candidate.matched_sequence)
    raise ValueError("FIMO matched_sequence is required to derive the TFBS core.")


def _hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length sequences.")
    return sum(1 for a, b in zip(left, right) if a != b)


def _evaluate_tier_target(*, n_sites: int, target_tier_fraction: float, eligible_unique: int) -> tuple[int, bool]:
    if target_tier_fraction <= 0 or target_tier_fraction > 1:
        raise ValueError("target_tier_fraction must be in (0, 1].")
    required_unique = int(np.ceil(float(n_sites) / float(target_tier_fraction)))
    return required_unique, int(eligible_unique) >= required_unique


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


def _select_by_mmr(
    ranked: Sequence[FimoCandidate],
    *,
    motif: PWMMotif,
    n_sites: int,
    alpha: float,
    shortlist_min: int,
    shortlist_factor: int,
    shortlist_max: Optional[int],
    tier_widening: Optional[Sequence[float]],
) -> tuple[list[FimoCandidate], dict[str, dict], dict]:
    if not ranked or n_sites <= 0:
        return [], {}, {"shortlist_k": 0, "tier_fraction_used": None, "tier_limit": 0}
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("selection.alpha must be in (0, 1].")
    log_odds = motif.log_odds or build_log_odds(motif.matrix, motif.background)
    core_by_seq = {cand.seq: _core_sequence(cand) for cand in ranked}
    contrib_by_seq = {seq: _contrib_vector(core, log_odds) for seq, core in core_by_seq.items()}
    scores = [cand.score for cand in ranked]
    norm_by_score = _score_norm(scores)
    tier_fractions = list(tier_widening) if tier_widening else [1.0]
    total = len(ranked)

    def _best_candidate(
        candidates: Sequence[FimoCandidate],
        selected: list[FimoCandidate],
    ) -> tuple[FimoCandidate, float, float]:
        best = None
        best_utility = None
        best_score = None
        best_core = None
        best_seq = None
        best_sim = None
        selected_vectors = [contrib_by_seq[c.seq] for c in selected]
        for cand in candidates:
            if cand in selected:
                continue
            score_norm = norm_by_score.get(float(cand.score), 1.0)
            if not selected_vectors:
                max_sim = 0.0
            else:
                dists = [float(np.abs(contrib_by_seq[cand.seq] - vec).sum()) for vec in selected_vectors]
                max_sim = _similarity_from_distance(min(dists))
            utility = float(alpha * score_norm - (1.0 - alpha) * max_sim)
            cand_core = core_by_seq[cand.seq]
            if best is None:
                best = cand
                best_utility = utility
                best_score = cand.score
                best_core = cand_core
                best_seq = cand.seq
                best_sim = max_sim
                continue
            if utility > float(best_utility):
                pass
            elif utility == float(best_utility):
                if cand.score > float(best_score):
                    pass
                elif cand.score == float(best_score):
                    if cand_core < str(best_core):
                        pass
                    elif cand_core == str(best_core):
                        if cand.seq >= str(best_seq):
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                continue
            best = cand
            best_utility = utility
            best_score = cand.score
            best_core = cand_core
            best_seq = cand.seq
            best_sim = max_sim
        if best is None or best_utility is None or best_sim is None:
            raise ValueError("MMR selection failed to identify a candidate.")
        return best, float(best_utility), float(best_sim)

    def _select_from_slice(candidates: Sequence[FimoCandidate]) -> tuple[list[FimoCandidate], dict[str, dict], int]:
        if not candidates:
            return [], {}, 0
        max_k = len(candidates) if shortlist_max is None else min(len(candidates), int(shortlist_max))
        base_k = max(int(shortlist_min), int(shortlist_factor) * int(n_sites))
        k = min(max_k, max(1, base_k))
        if k <= 0:
            k = min(len(candidates), int(n_sites))
        shortlist = list(candidates[:k])
        selected: list[FimoCandidate] = []
        meta: dict[str, dict] = {}
        while shortlist and len(selected) < int(n_sites):
            pick, utility, max_sim = _best_candidate(shortlist, selected)
            selected.append(pick)
            meta[pick.seq] = {
                "selection_rank": len(selected),
                "selection_utility": float(utility),
                "nearest_selected_similarity": float(max_sim),
            }
        if len(selected) < int(n_sites) and k < max_k:
            shortlist = list(candidates[:max_k])
            selected = []
            meta = {}
            while shortlist and len(selected) < int(n_sites):
                pick, utility, max_sim = _best_candidate(shortlist, selected)
                selected.append(pick)
                meta[pick.seq] = {
                    "selection_rank": len(selected),
                    "selection_utility": float(utility),
                    "nearest_selected_similarity": float(max_sim),
                }
            return selected, meta, int(max_k)
        return selected, meta, int(k)

    last_selected: list[FimoCandidate] = []
    last_meta: dict[str, dict] = {}
    last_shortlist = 0
    fraction_used = None
    tier_limit = total
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
        if len(selected) >= int(n_sites):
            break
    diag = {
        "shortlist_k": int(last_shortlist),
        "tier_fraction_used": fraction_used,
        "tier_limit": int(tier_limit),
    }
    return last_selected, last_meta, diag


def _collapse_by_core_identity(ranked: Sequence[FimoCandidate]) -> tuple[list[FimoCandidate], int]:
    best_by_core: dict[str, FimoCandidate] = {}
    for cand in ranked:
        core = _core_sequence(cand)
        prev = best_by_core.get(core)
        if prev is None or cand.score > prev.score or (cand.score == prev.score and cand.seq < prev.seq):
            best_by_core[core] = cand
    collapsed = max(0, len(ranked) - len(best_by_core))
    kept = sorted(best_by_core.values(), key=lambda item: (-item.score, item.seq))
    return kept, collapsed


def _write_candidate_records(
    records: list[dict],
    *,
    debug_output_dir: Path,
    debug_label: str,
    motif_id: str,
    motif_hash: str | None = None,
) -> Path:
    suffix = ""
    if motif_hash:
        suffix = f"__{_safe_label(motif_hash[:10])}"
    safe_label = f"{_safe_label(debug_label or motif_id)}{suffix}"
    debug_output_dir.mkdir(parents=True, exist_ok=True)
    path = debug_output_dir / f"candidates__{safe_label}.parquet"
    df = pd.DataFrame(records)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            if "candidate_id" not in existing.columns or "candidate_id" not in df.columns:
                raise ValueError(
                    f"Candidate append requires candidate_id in {path}. "
                    "Clear outputs/pools/candidates or use --fresh to reset."
                )
            if set(existing.columns) != set(df.columns):
                raise ValueError(
                    f"Candidate schema mismatch for {path}. Clear outputs/pools/candidates or use --fresh to reset."
                )
            df = df[existing.columns]
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["candidate_id"], keep="last")
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise RuntimeError(f"Failed to append candidate records to {path}") from exc
    df.to_parquet(path, index=False)
    return path


@dataclass(frozen=True)
class PWMMotif:
    motif_id: str
    matrix: List[dict[str, float]]  # per-position A/C/G/T probabilities
    background: dict[str, float]
    log_odds: Optional[List[dict[str, float]]] = None


@dataclass(frozen=True)
class PWMSamplingSummary:
    input_name: Optional[str]
    regulator: str
    backend: str
    uniqueness_key: Optional[str]
    collapsed_by_core_identity: Optional[int]
    generated: int
    target: int
    target_sites: Optional[int]
    candidates_with_hit: Optional[int]
    eligible_total: Optional[int]
    eligible: int
    retained: int
    retained_len_min: Optional[int]
    retained_len_median: Optional[float]
    retained_len_mean: Optional[float]
    retained_len_max: Optional[int]
    retained_score_min: Optional[float]
    retained_score_median: Optional[float]
    retained_score_mean: Optional[float]
    retained_score_max: Optional[float]
    eligible_tier_counts: Optional[List[int]]
    retained_tier_counts: Optional[List[int]]
    tier0_score: Optional[float]
    tier1_score: Optional[float]
    tier2_score: Optional[float]
    eligible_score_hist_edges: Optional[List[float]] = None
    eligible_score_hist_counts: Optional[List[int]] = None
    tier_target_fraction: Optional[float] = None
    tier_target_required_unique: Optional[int] = None
    tier_target_met: Optional[bool] = None
    selection_policy: Optional[str] = None
    selection_alpha: Optional[float] = None
    selection_similarity: Optional[str] = None
    selection_shortlist_k: Optional[int] = None
    selection_shortlist_min: Optional[int] = None
    selection_shortlist_factor: Optional[int] = None
    selection_shortlist_max: Optional[int] = None
    selection_tier_fraction_used: Optional[float] = None
    selection_tier_limit: Optional[int] = None
    diversity_nearest_distance_mean: Optional[float] = None
    diversity_nearest_distance_min: Optional[float] = None
    diversity_nearest_similarity_mean: Optional[float] = None


def normalize_background(bg: Optional[dict[str, float]]) -> dict[str, float]:
    if not bg:
        return {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    total = sum(bg.values())
    if total <= 0:
        raise ValueError("Background frequencies must sum to > 0.")
    return {k: float(v) / total for k, v in bg.items()}


def build_log_odds(
    matrix: List[dict[str, float]],
    background: dict[str, float],
    *,
    smoothing_alpha: float = SMOOTHING_ALPHA,
) -> List[dict[str, float]]:
    # Alignment (3): smooth PWM probabilities to avoid -inf log-odds on zeros.
    bg = normalize_background(background)
    log_odds: List[dict[str, float]] = []
    for row in matrix:
        lod_row: dict[str, float] = {}
        for base in ("A", "C", "G", "T"):
            p = float(row.get(base, 0.0))
            b = float(bg.get(base, 0.0))
            if smoothing_alpha > 0.0:
                p = (1.0 - smoothing_alpha) * p + smoothing_alpha * b
            if p <= 0.0 or b <= 0.0:
                lod_row[base] = float("-inf")
            else:
                lod_row[base] = float(np.log(p / b))
        log_odds.append(lod_row)
    return log_odds


def score_sequence(
    seq: str,
    matrix: List[dict[str, float]],
    *,
    log_odds: Optional[List[dict[str, float]]] = None,
    background: Optional[dict[str, float]] = None,
) -> float:
    score = 0.0
    if log_odds is None:
        if background is None:
            background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
        log_odds = build_log_odds(matrix, background)
    for base, probs in zip(seq, log_odds):
        lod = probs.get(base, float("-inf"))
        if lod == float("-inf"):
            return float("-inf")
        score += float(lod)
    return score


def _summarize_lengths(lengths: Sequence[int]) -> tuple[Optional[int], Optional[float], Optional[float], Optional[int]]:
    if not lengths:
        return None, None, None, None
    arr = np.asarray(lengths, dtype=float)
    return int(arr.min()), float(np.median(arr)), float(arr.mean()), int(arr.max())


def _summarize_scores(
    scores: Sequence[float],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not scores:
        return None, None, None, None
    arr = np.asarray(scores, dtype=float)
    return float(arr.min()), float(np.median(arr)), float(arr.mean()), float(arr.max())


def _rank_scored_sequences(scored: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    best_by_seq: dict[str, float] = {}
    for seq, score in scored:
        seq = str(seq)
        val = float(score)
        prev = best_by_seq.get(seq)
        if prev is None or val > prev:
            best_by_seq[seq] = val
    return sorted(best_by_seq.items(), key=lambda item: (-item[1], item[0]))


def _ranked_sequence_positions(ranked: Sequence[tuple[str, float]]) -> dict[str, int]:
    return {seq: idx + 1 for idx, (seq, _score) in enumerate(ranked)}


def _score_tier_counts(total: int) -> tuple[int, int, int, int]:
    return score_tier_counts(total)


def _assign_score_tiers(ranked: Sequence[tuple[str, float]]) -> list[int]:
    total = len(ranked)
    n0, n1, n2, _n3 = _score_tier_counts(total)
    tiers: list[int] = []
    for idx in range(total):
        if idx < n0:
            tiers.append(0)
        elif idx < n0 + n1:
            tiers.append(1)
        elif idx < n0 + n1 + n2:
            tiers.append(2)
        else:
            tiers.append(3)
    return tiers


def _build_score_hist(
    scores: Sequence[float],
    *,
    bins: int = SCORE_HIST_BINS,
) -> tuple[list[float], list[int]]:
    vals = [float(v) for v in scores if v is not None]
    if not vals:
        return [], []
    lo = min(vals)
    hi = max(vals)
    lo = min(lo, 0.0)
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, num=int(bins) + 1)
    counts, _ = np.histogram(np.asarray(vals, dtype=float), bins=edges)
    return [float(v) for v in edges], [int(v) for v in counts]


def _build_summary(
    *,
    generated: int,
    target: int,
    target_sites: Optional[int],
    candidates_with_hit: Optional[int],
    eligible_total: Optional[int],
    eligible: Sequence[str],
    retained: Sequence[str],
    retained_scores: Optional[Sequence[float]] = None,
    uniqueness_key: Optional[str] = None,
    collapsed_by_core_identity: Optional[int] = None,
    eligible_tier_counts: Optional[Sequence[int]] = None,
    retained_tier_counts: Optional[Sequence[int]] = None,
    tier0_score: Optional[float] = None,
    tier1_score: Optional[float] = None,
    tier2_score: Optional[float] = None,
    eligible_score_hist_edges: Optional[Sequence[float]] = None,
    eligible_score_hist_counts: Optional[Sequence[int]] = None,
    tier_target_fraction: Optional[float] = None,
    tier_target_required_unique: Optional[int] = None,
    tier_target_met: Optional[bool] = None,
    selection_policy: Optional[str] = None,
    selection_alpha: Optional[float] = None,
    selection_similarity: Optional[str] = None,
    selection_shortlist_k: Optional[int] = None,
    selection_shortlist_min: Optional[int] = None,
    selection_shortlist_factor: Optional[int] = None,
    selection_shortlist_max: Optional[int] = None,
    selection_tier_fraction_used: Optional[float] = None,
    selection_tier_limit: Optional[int] = None,
    diversity_nearest_distance_mean: Optional[float] = None,
    diversity_nearest_distance_min: Optional[float] = None,
    diversity_nearest_similarity_mean: Optional[float] = None,
    input_name: Optional[str] = None,
    regulator: Optional[str] = None,
    backend: Optional[str] = None,
) -> PWMSamplingSummary:
    lengths = [len(seq) for seq in retained]
    min_len, median_len, mean_len, max_len = _summarize_lengths(lengths)
    score_min, score_median, score_mean, score_max = _summarize_scores(retained_scores or [])
    return PWMSamplingSummary(
        input_name=input_name,
        regulator=str(regulator or ""),
        backend=str(backend or ""),
        uniqueness_key=str(uniqueness_key) if uniqueness_key is not None else None,
        collapsed_by_core_identity=int(collapsed_by_core_identity) if collapsed_by_core_identity is not None else None,
        generated=int(generated),
        target=int(target),
        target_sites=int(target_sites) if target_sites is not None else None,
        candidates_with_hit=int(candidates_with_hit) if candidates_with_hit is not None else None,
        eligible_total=int(eligible_total) if eligible_total is not None else None,
        eligible=int(len(eligible)),
        retained=int(len(retained)),
        retained_len_min=min_len,
        retained_len_median=median_len,
        retained_len_mean=mean_len,
        retained_len_max=max_len,
        retained_score_min=score_min,
        retained_score_median=score_median,
        retained_score_mean=score_mean,
        retained_score_max=score_max,
        eligible_tier_counts=list(eligible_tier_counts) if eligible_tier_counts is not None else None,
        retained_tier_counts=list(retained_tier_counts) if retained_tier_counts is not None else None,
        tier0_score=float(tier0_score) if tier0_score is not None else None,
        tier1_score=float(tier1_score) if tier1_score is not None else None,
        tier2_score=float(tier2_score) if tier2_score is not None else None,
        eligible_score_hist_edges=list(eligible_score_hist_edges) if eligible_score_hist_edges is not None else None,
        eligible_score_hist_counts=list(eligible_score_hist_counts) if eligible_score_hist_counts is not None else None,
        tier_target_fraction=float(tier_target_fraction) if tier_target_fraction is not None else None,
        tier_target_required_unique=int(tier_target_required_unique)
        if tier_target_required_unique is not None
        else None,
        tier_target_met=bool(tier_target_met) if tier_target_met is not None else None,
        selection_policy=str(selection_policy) if selection_policy is not None else None,
        selection_alpha=float(selection_alpha) if selection_alpha is not None else None,
        selection_similarity=str(selection_similarity) if selection_similarity is not None else None,
        selection_shortlist_k=int(selection_shortlist_k) if selection_shortlist_k is not None else None,
        selection_shortlist_min=int(selection_shortlist_min) if selection_shortlist_min is not None else None,
        selection_shortlist_factor=int(selection_shortlist_factor) if selection_shortlist_factor is not None else None,
        selection_shortlist_max=int(selection_shortlist_max) if selection_shortlist_max is not None else None,
        selection_tier_fraction_used=float(selection_tier_fraction_used)
        if selection_tier_fraction_used is not None
        else None,
        selection_tier_limit=int(selection_tier_limit) if selection_tier_limit is not None else None,
        diversity_nearest_distance_mean=float(diversity_nearest_distance_mean)
        if diversity_nearest_distance_mean is not None
        else None,
        diversity_nearest_distance_min=float(diversity_nearest_distance_min)
        if diversity_nearest_distance_min is not None
        else None,
        diversity_nearest_similarity_mean=float(diversity_nearest_similarity_mean)
        if diversity_nearest_similarity_mean is not None
        else None,
    )


def sample_sequence_from_background(rng: np.random.Generator, probs: dict[str, float], length: int) -> str:
    bases = ["A", "C", "G", "T"]
    weights = np.array([probs[b] for b in bases], dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(len(bases), size=length, replace=True, p=weights)
    return "".join(bases[i] for i in idx)


def sample_sequence_from_pwm(rng: np.random.Generator, matrix: List[dict[str, float]]) -> str:
    seq = []
    for probs in matrix:
        bases = ["A", "C", "G", "T"]
        weights = np.array([probs[b] for b in bases], dtype=float)
        weights = weights / weights.sum()
        seq.append(bases[int(rng.choice(len(bases), p=weights))])
    return "".join(seq)


def _matrix_array(matrix: List[dict[str, float]]) -> np.ndarray:
    rows = []
    for row in matrix:
        rows.append(
            [
                float(row.get("A", 0.0)),
                float(row.get("C", 0.0)),
                float(row.get("G", 0.0)),
                float(row.get("T", 0.0)),
            ]
        )
    return np.asarray(rows, dtype=float)


def _information_bits(matrix: np.ndarray) -> float:
    p = matrix + 1e-9
    return float((2 + (p * np.log2(p)).sum(axis=1)).sum())


def _select_pwm_window(
    *,
    matrix: List[dict[str, float]],
    log_odds: List[dict[str, float]],
    length: int,
    strategy: str,
) -> tuple[List[dict[str, float]], List[dict[str, float]], int, float]:
    if length < 1:
        raise ValueError("pwm.sampling.trim_window_length must be >= 1")
    if length > len(matrix):
        raise ValueError(f"pwm.sampling.trim_window_length={length} exceeds motif width {len(matrix)}")
    if strategy != "max_info":
        raise ValueError("pwm.sampling.trim_window_strategy must be 'max_info'")
    if length == len(matrix):
        return matrix, log_odds, 0, 0.0
    arr = _matrix_array(matrix)
    best_start = 0
    best_score = float("-inf")
    last_start = len(matrix) - length
    for start in range(last_start + 1):
        window = arr[start : start + length]
        score = _information_bits(window)
        if score > best_score:
            best_score = score
            best_start = start
    return (
        matrix[best_start : best_start + length],
        log_odds[best_start : best_start + length],
        best_start,
        best_score,
    )


def sample_pwm_sites(
    rng: np.random.Generator,
    motif: PWMMotif,
    *,
    input_name: Optional[str] = None,
    motif_hash: str | None = None,
    run_id: str | None = None,
    strategy: str,
    n_sites: int,
    mining: Optional[object] = None,
    bgfile: Optional[str | Path] = None,
    keep_all_candidates_debug: bool = False,
    include_matched_sequence: bool = False,
    uniqueness_key: str = "sequence",
    selection: Optional[object] = None,
    debug_output_dir: Optional[Path] = None,
    debug_label: Optional[str] = None,
    length_policy: str = "exact",
    length_range: Optional[Sequence[int]] = None,
    trim_window_length: Optional[int] = None,
    trim_window_strategy: str = "max_info",
    return_metadata: bool = False,
    return_summary: bool = False,
) -> Union[
    List[str],
    Tuple[List[str], dict[str, dict]],
    Tuple[List[str], PWMSamplingSummary],
    Tuple[List[str], dict[str, dict], Optional[PWMSamplingSummary]],
]:
    if n_sites <= 0:
        raise ValueError("n_sites must be > 0")
    scoring_backend = "fimo"
    uniqueness_key = str(uniqueness_key or "sequence").lower()
    if uniqueness_key not in {"sequence", "core"}:
        raise ValueError(f"Stage-A PWM sampling uniqueness.key must be 'sequence' or 'core', got '{uniqueness_key}'.")
    if keep_all_candidates_debug and run_id is None:
        raise ValueError("Stage-A PWM sampling keep_all_candidates_debug requires run_id to be set.")
    if strategy == "consensus" and n_sites != 1:
        raise ValueError("Stage-A PWM sampling strategy 'consensus' requires n_sites=1")

    width = len(motif.matrix)
    if width <= 0:
        raise ValueError(f"PWM motif '{motif.motif_id}' has zero width.")
    if length_policy not in {"exact", "range"}:
        raise ValueError(f"Unsupported pwm length.policy: {length_policy}")
    log_odds = motif.log_odds or build_log_odds(motif.matrix, motif.background)
    window_label = "full"
    if trim_window_length is not None:
        matrix, log_odds, window_start, window_score = _select_pwm_window(
            matrix=motif.matrix,
            log_odds=log_odds,
            length=int(trim_window_length),
            strategy=str(trim_window_strategy),
        )
        width = len(matrix)
        window_label = f"{width}@{window_start}"
        log.debug(
            "Stage-A PWM sampling trimmed motif %s to window length %d (start=%d, score=%.3f).",
            motif.motif_id,
            width,
            window_start,
            window_score,
        )
    else:
        matrix = motif.matrix

    score_label = "best_hit_score"
    length_label = str(length_policy)
    if length_policy == "range" and length_range is not None and len(length_range) == 2:
        length_label = f"{length_policy}({length_range[0]}..{length_range[1]})"

    selection_policy = str(_selection_attr(selection, "policy", "top_score") or "top_score").lower()
    if selection_policy not in {"top_score", "mmr"}:
        raise ValueError(f"Stage-A selection.policy must be 'top_score' or 'mmr', got '{selection_policy}'.")
    selection_alpha = _selection_attr(selection, "alpha", 0.9)
    selection_shortlist_min = _selection_attr(selection, "shortlist_min", 50)
    selection_shortlist_factor = _selection_attr(selection, "shortlist_factor", 5)
    selection_shortlist_max = _selection_attr(selection, "shortlist_max", None)
    selection_tier_widening = _selection_attr(selection, "tier_widening", None)
    if selection_policy == "mmr":
        selection_alpha = float(selection_alpha)
        if selection_alpha <= 0.0 or selection_alpha > 1.0:
            raise ValueError("selection.alpha must be in (0, 1].")
        if int(selection_shortlist_min) <= 0:
            raise ValueError("selection.shortlist_min must be > 0.")
        if int(selection_shortlist_factor) <= 0:
            raise ValueError("selection.shortlist_factor must be > 0.")
        if selection_shortlist_max is not None and int(selection_shortlist_max) <= 0:
            raise ValueError("selection.shortlist_max must be > 0 when set.")
        if selection_shortlist_max is not None and int(selection_shortlist_max) < int(selection_shortlist_min):
            raise ValueError("selection.shortlist_max must be >= selection.shortlist_min.")

    if selection_tier_widening is not None:
        if hasattr(selection_tier_widening, "enabled"):
            enabled = bool(getattr(selection_tier_widening, "enabled"))
            ladder = getattr(selection_tier_widening, "ladder", None)
            selection_tier_widening = ladder if enabled else None
        elif isinstance(selection_tier_widening, dict):
            enabled = bool(selection_tier_widening.get("enabled", False))
            selection_tier_widening = selection_tier_widening.get("ladder") if enabled else None

    include_matched_sequence = bool(include_matched_sequence or uniqueness_key == "core" or selection_policy == "mmr")

    budget_mode = str(_budget_attr(mining, "mode", "fixed_candidates") or "fixed_candidates").lower()
    if budget_mode not in {"tier_target", "fixed_candidates"}:
        raise ValueError(
            f"pwm.sampling.mining.budget.mode must be 'tier_target' or 'fixed_candidates', got '{budget_mode}'."
        )
    budget_target_tier_fraction = _budget_attr(mining, "target_tier_fraction", None)
    budget_candidates = _budget_attr(mining, "candidates", None)
    budget_max_candidates = _budget_attr(mining, "max_candidates", None)
    budget_min_candidates = _budget_attr(mining, "min_candidates", None)
    budget_max_seconds = _budget_attr(mining, "max_seconds", None)
    budget_growth_factor = float(_budget_attr(mining, "growth_factor", 1.25))
    if budget_max_candidates is not None and int(budget_max_candidates) <= 0:
        raise ValueError("pwm.sampling.mining.budget.max_candidates must be > 0 when set.")
    if budget_min_candidates is not None and int(budget_min_candidates) <= 0:
        raise ValueError("pwm.sampling.mining.budget.min_candidates must be > 0 when set.")
    if (
        budget_min_candidates is not None
        and budget_max_candidates is not None
        and int(budget_min_candidates) > int(budget_max_candidates)
    ):
        raise ValueError("pwm.sampling.mining.budget.min_candidates must be <= max_candidates.")
    if budget_max_seconds is not None and float(budget_max_seconds) <= 0:
        raise ValueError("pwm.sampling.mining.budget.max_seconds must be > 0 when set.")
    if budget_growth_factor <= 1.0:
        raise ValueError("pwm.sampling.mining.budget.growth_factor must be > 1.0")
    if budget_mode == "fixed_candidates":
        if budget_candidates is None:
            raise ValueError("pwm.sampling.mining.budget.candidates must be set when mode=fixed_candidates.")
        if int(budget_candidates) <= 0:
            raise ValueError("pwm.sampling.mining.budget.candidates must be > 0.")
    else:
        if budget_target_tier_fraction is None:
            raise ValueError("pwm.sampling.mining.budget.target_tier_fraction is required for mode=tier_target.")
        if float(budget_target_tier_fraction) <= 0 or float(budget_target_tier_fraction) > 1:
            raise ValueError("pwm.sampling.mining.budget.target_tier_fraction must be in (0, 1].")
        if budget_max_candidates is None and budget_max_seconds is None:
            raise ValueError("pwm.sampling.mining.budget.mode=tier_target requires max_candidates or max_seconds.")

    def _cap_label(cap_applied: bool, time_limited: bool) -> str:
        cap_label = ""
        if time_limited and budget_max_seconds is not None:
            cap_label = f" (max_seconds={budget_max_seconds})"
        if cap_applied and budget_max_candidates is not None:
            cap_label = (
                f"{cap_label}; max_candidates={budget_max_candidates}"
                if cap_label
                else (f" (max_candidates={budget_max_candidates})")
            )
        return cap_label

    def _context(length_obs: str, cap_applied: bool, requested: int, generated: int, time_limited: bool) -> dict:
        return {
            "motif_id": motif.motif_id,
            "width": width,
            "strategy": strategy,
            "length_label": length_label,
            "window_label": window_label,
            "length_observed": length_obs,
            "score_label": score_label,
            "n_sites": n_sites,
            "budget_mode": budget_mode,
            "target_tier_fraction": budget_target_tier_fraction,
            "requested_candidates": requested,
            "generated_candidates": generated,
            "cap_applied": cap_applied,
            "cap_label": _cap_label(cap_applied, time_limited),
            "time_limited": time_limited,
            "mining_batch_size": _mining_attr(mining, "batch_size"),
            "mining_max_seconds": budget_max_seconds,
            "mining_log_every_batches": _mining_attr(mining, "log_every_batches"),
        }

    def _resolve_length() -> int:
        if length_policy == "exact":
            return width
        if length_range is None or len(length_range) != 2:
            raise ValueError("pwm.sampling.length.range must be provided when length.policy=range")
        lo, hi = int(length_range[0]), int(length_range[1])
        if lo <= 0 or hi <= 0:
            raise ValueError("pwm.sampling.length.range values must be > 0")
        if lo > hi:
            raise ValueError("pwm.sampling.length.range must be min <= max")
        if lo < width:
            raise ValueError(f"pwm.sampling.length.range min must be >= motif width ({width}), got {lo}")
        return int(rng.integers(lo, hi + 1))

    def _embed_with_background(seq: str, target_len: int) -> str:
        if target_len == len(seq):
            return seq
        extra = target_len - len(seq)
        left_len = int(rng.integers(0, extra + 1))
        right_len = extra - left_len
        left = sample_sequence_from_background(rng, motif.background, left_len)
        right = sample_sequence_from_background(rng, motif.background, right_len)
        return f"{left}{seq}{right}"

    progress: _PwmSamplingProgress | None = None

    def _score_with_fimo(
        *,
        n_candidates: int,
        requested: int,
        sequences: Optional[List[str]] = None,
    ) -> tuple[List[str], dict[str, dict]]:
        import tempfile

        from .pwm_fimo import (
            aggregate_best_hits,
            build_candidate_records,
            run_fimo,
            write_candidates_fasta,
            write_minimal_meme_motif,
        )

        mining_batch_size = int(_mining_attr(mining, "batch_size", n_candidates))
        mining_max_seconds = budget_max_seconds
        mining_log_every = int(_mining_attr(mining, "log_every_batches", 1))
        log.info(
            "FIMO mining config for %s: mode=%s target=%d batch=%d max_seconds=%s max_candidates=%s thresh=%s",
            motif.motif_id,
            budget_mode,
            n_candidates,
            mining_batch_size,
            str(mining_max_seconds) if mining_max_seconds is not None else "-",
            str(budget_max_candidates) if budget_max_candidates is not None else "-",
            FIMO_REPORT_THRESH,
            extra={"suppress_stdout": True},
        )
        debug_path: Optional[Path] = None
        debug_dir = debug_output_dir
        if keep_all_candidates_debug:
            if debug_dir is None:
                tmp_dir = tempfile.mkdtemp(prefix="densegen-fimo-")
                debug_dir = Path(tmp_dir)
                log.warning(
                    "Stage-A PWM sampling keep_all_candidates_debug enabled without outputs_root; "
                    "writing FIMO debug TSVs to %s",
                    debug_dir,
                )
            debug_dir.mkdir(parents=True, exist_ok=True)
            label = _safe_label(debug_label or motif.motif_id)
            debug_path = debug_dir / f"{label}__fimo.tsv"

        def _merge_tsv(existing: list[str], text: str) -> None:
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                return
            if not existing:
                existing.extend(lines)
                return
            header_skipped = False
            for ln in lines:
                if ln.lstrip().startswith("#"):
                    continue
                if not header_skipped:
                    header_skipped = True
                    continue
                existing.append(ln)

        def _generate_batch(count: int) -> tuple[list[str], list[int], bool]:
            batch_start = time.monotonic()
            sequences: list[str] = []
            lengths: list[int] = []
            time_limited = False
            for _ in range(count):
                if mining_max_seconds is not None and sequences:
                    if (time.monotonic() - batch_start) >= float(mining_max_seconds):
                        time_limited = True
                        break
                target_len = _resolve_length()
                lengths.append(int(target_len))
                if strategy == "background":
                    core = sample_sequence_from_background(rng, motif.background, width)
                else:
                    core = sample_sequence_from_pwm(rng, matrix)
                full_seq = _embed_with_background(core, target_len)
                sequences.append(full_seq)
            return sequences, lengths, time_limited

        candidates_by_seq: dict[str, FimoCandidate] = {}
        candidates_with_hit = 0
        eligible_total = 0
        lengths_all: list[int] = []
        generated_total = 0
        time_limited = False
        mining_time_limited = False
        cap_applied = False
        batches = 0
        tsv_lines: list[str] = []
        provided_sequences = sequences
        requested_final = int(requested)
        candidate_records: list[dict] | None = [] if keep_all_candidates_debug else None

        def _record_candidate(
            *,
            seq: str,
            hit,
            accepted: bool,
            reject_reason: str | None,
        ) -> None:
            if candidate_records is None:
                return
            resolved_motif_id = motif_hash or motif.motif_id
            candidate_id = hash_candidate_id(
                input_name=input_name,
                motif_id=resolved_motif_id,
                sequence=seq,
                scoring_backend=scoring_backend,
            )
            candidate_records.append(
                {
                    "candidate_id": candidate_id,
                    "run_id": run_id,
                    "input_name": input_name,
                    "motif_id": resolved_motif_id,
                    "motif_label": motif.motif_id,
                    "scoring_backend": scoring_backend,
                    "sequence": seq,
                    "best_hit_score": None if hit is None else hit.score,
                    "start": None if hit is None else hit.start,
                    "stop": None if hit is None else hit.stop,
                    "strand": None if hit is None else hit.strand,
                    "matched_sequence": None if hit is None else hit.matched_sequence,
                    "accepted": bool(accepted),
                    "selected": False,
                    "reject_reason": reject_reason,
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            meme_path = tmp_path / "motif.meme"
            motif_for_fimo = PWMMotif(motif_id=motif.motif_id, matrix=matrix, background=motif.background)
            # FIMO background uses the MEME motif background unless bgfile is provided.
            write_minimal_meme_motif(motif_for_fimo, meme_path)
            if provided_sequences is not None:
                lengths_all = [len(seq) for seq in provided_sequences]
                fasta_path = tmp_path / "candidates.fasta"
                records = build_candidate_records(motif.motif_id, provided_sequences, start_index=0)
                write_candidates_fasta(records, fasta_path)
                rows, raw_tsv = run_fimo(
                    meme_motif_path=meme_path,
                    fasta_path=fasta_path,
                    bgfile=Path(bgfile) if bgfile is not None else None,
                    thresh=FIMO_REPORT_THRESH,
                    norc=True,
                    include_matched_sequence=include_matched_sequence or keep_all_candidates_debug,
                    return_tsv=debug_path is not None,
                )
                if debug_path is not None and raw_tsv is not None:
                    _merge_tsv(tsv_lines, raw_tsv)
                best_hits = aggregate_best_hits(rows)
                for rec_id, seq in records:
                    hit = best_hits.get(rec_id)
                    if hit is None:
                        _record_candidate(
                            seq=seq,
                            hit=None,
                            accepted=False,
                            reject_reason="no_hit",
                        )
                        continue
                    candidates_with_hit += 1
                    if hit.score <= 0:
                        _record_candidate(
                            seq=seq,
                            hit=hit,
                            accepted=False,
                            reject_reason="score_non_positive",
                        )
                        continue
                    eligible_total += 1
                    _record_candidate(
                        seq=seq,
                        hit=hit,
                        accepted=True,
                        reject_reason=None,
                    )
                    prev = candidates_by_seq.get(seq)
                    if (
                        prev is None
                        or hit.score > prev.score
                        or (hit.score == prev.score and (hit.start, hit.stop) < (prev.start, prev.stop))
                    ):
                        candidates_by_seq[seq] = FimoCandidate(
                            seq=seq,
                            score=hit.score,
                            start=hit.start,
                            stop=hit.stop,
                            strand=hit.strand,
                            matched_sequence=hit.matched_sequence,
                        )
                generated_total = len(provided_sequences)
                batches = 1
                if progress is not None:
                    progress.update(generated=generated_total, accepted=len(candidates_by_seq), force=True)
            else:
                mining_start = time.monotonic()
                target_candidates = int(n_candidates)
                cap_applied = False
                core_by_seq: dict[str, str] = {}
                core_counts: dict[str, int] = {}

                def _eligible_unique_count() -> int:
                    if uniqueness_key == "core":
                        return int(len(core_counts))
                    return int(len(candidates_by_seq))

                def _record_core(seq: str, cand: FimoCandidate) -> None:
                    core = _core_sequence(cand)
                    prev_core = core_by_seq.get(seq)
                    if prev_core is not None:
                        core_counts[prev_core] = core_counts.get(prev_core, 0) - 1
                        if core_counts.get(prev_core) == 0:
                            core_counts.pop(prev_core, None)
                    core_by_seq[seq] = core
                    core_counts[core] = core_counts.get(core, 0) + 1

                while True:
                    if budget_max_seconds is not None and (time.monotonic() - mining_start) >= float(
                        budget_max_seconds
                    ):
                        mining_time_limited = True
                        break
                    if generated_total >= target_candidates:
                        if budget_mode == "tier_target":
                            if budget_min_candidates is None or generated_total >= int(budget_min_candidates):
                                if budget_target_tier_fraction is not None:
                                    required_unique = int(np.ceil(float(n_sites) / float(budget_target_tier_fraction)))
                                    if _eligible_unique_count() >= required_unique:
                                        break
                        if budget_max_candidates is not None and generated_total >= int(budget_max_candidates):
                            cap_applied = True
                            break
                        next_target = int(np.ceil(target_candidates * budget_growth_factor))
                        if budget_max_candidates is not None:
                            next_target = min(next_target, int(budget_max_candidates))
                        if next_target <= target_candidates:
                            cap_applied = True
                            break
                        target_candidates = next_target
                        if progress is not None:
                            progress.target = target_candidates
                        continue
                    remaining = int(target_candidates) - generated_total
                    if remaining <= 0:
                        continue
                    batch_target = min(int(mining_batch_size), remaining)
                    sequences, lengths, batch_limited = _generate_batch(batch_target)
                    if batch_limited:
                        time_limited = True
                    if not sequences:
                        break
                    lengths_all.extend(lengths)
                    fasta_path = tmp_path / "candidates.fasta"
                    records = build_candidate_records(motif.motif_id, sequences, start_index=generated_total)
                    write_candidates_fasta(records, fasta_path)
                    rows, raw_tsv = run_fimo(
                        meme_motif_path=meme_path,
                        fasta_path=fasta_path,
                        bgfile=Path(bgfile) if bgfile is not None else None,
                        thresh=FIMO_REPORT_THRESH,
                        norc=True,
                        include_matched_sequence=include_matched_sequence or keep_all_candidates_debug,
                        return_tsv=debug_path is not None,
                    )
                    if debug_path is not None and raw_tsv is not None:
                        _merge_tsv(tsv_lines, raw_tsv)
                    best_hits = aggregate_best_hits(rows)
                    for rec_id, seq in records:
                        hit = best_hits.get(rec_id)
                        if hit is None:
                            _record_candidate(
                                seq=seq,
                                hit=None,
                                accepted=False,
                                reject_reason="no_hit",
                            )
                            continue
                        candidates_with_hit += 1
                        if hit.score <= 0:
                            _record_candidate(
                                seq=seq,
                                hit=hit,
                                accepted=False,
                                reject_reason="score_non_positive",
                            )
                            continue
                        eligible_total += 1
                        _record_candidate(
                            seq=seq,
                            hit=hit,
                            accepted=True,
                            reject_reason=None,
                        )
                        prev = candidates_by_seq.get(seq)
                        if (
                            prev is None
                            or hit.score > prev.score
                            or (hit.score == prev.score and (hit.start, hit.stop) < (prev.start, prev.stop))
                        ):
                            cand = FimoCandidate(
                                seq=seq,
                                score=hit.score,
                                start=hit.start,
                                stop=hit.stop,
                                strand=hit.strand,
                                matched_sequence=hit.matched_sequence,
                            )
                            candidates_by_seq[seq] = cand
                            if uniqueness_key == "core":
                                _record_core(seq, cand)
                    generated_total += len(sequences)
                    batches += 1
                    if progress is not None:
                        progress.update(
                            generated=generated_total,
                            accepted=_eligible_unique_count(),
                            batch_index=batches,
                            batch_total=None,
                        )
                    if mining_log_every > 0 and batches % mining_log_every == 0:
                        log.info(
                            "FIMO mining %s batch %d/%s: generated=%d/%d eligible=%d",
                            motif.motif_id,
                            batches,
                            "-",
                            generated_total,
                            target_candidates,
                            _eligible_unique_count(),
                        )
                    if budget_mode == "tier_target" and budget_target_tier_fraction is not None:
                        if budget_min_candidates is None or generated_total >= int(budget_min_candidates):
                            required_unique = int(np.ceil(float(n_sites) / float(budget_target_tier_fraction)))
                            if _eligible_unique_count() >= required_unique:
                                break
                requested_final = int(target_candidates)

        if debug_path is not None and tsv_lines:
            debug_path.write_text("\n".join(tsv_lines) + "\n")
            log.info("FIMO debug TSV written: %s", debug_path)

        length_obs = "-"
        if lengths_all:
            length_obs = (
                f"{min(lengths_all)}..{max(lengths_all)}"
                if min(lengths_all) != max(lengths_all)
                else str(lengths_all[0])
            )

        context = _context(
            length_obs,
            cap_applied,
            requested_final,
            generated_total,
            time_limited or mining_time_limited,
        )
        context["mining_batch_size"] = mining_batch_size
        context["mining_max_seconds"] = mining_max_seconds
        context["mining_time_limited"] = mining_time_limited
        ranked = sorted(candidates_by_seq.values(), key=lambda cand: (-cand.score, cand.seq))
        collapsed_by_core_identity = 0
        if uniqueness_key == "core":
            ranked, collapsed_by_core_identity = _collapse_by_core_identity(ranked)
        eligible_hits = len(ranked)
        ranked_pairs = [(cand.seq, cand.score) for cand in ranked]
        tiers = _assign_score_tiers(ranked_pairs)
        rank_by_seq = _ranked_sequence_positions(ranked_pairs)
        tier_by_seq = {cand.seq: tiers[idx] for idx, cand in enumerate(ranked)}
        eligible_tier_counts = [0, 0, 0, 0]
        for tier in tiers:
            eligible_tier_counts[tier] += 1
        selection_meta: dict[str, dict] = {}
        selection_diag: dict = {}
        if selection_policy == "mmr":
            picked, selection_meta, selection_diag = _select_by_mmr(
                ranked,
                motif=motif,
                n_sites=int(n_sites),
                alpha=float(selection_alpha),
                shortlist_min=int(selection_shortlist_min),
                shortlist_factor=int(selection_shortlist_factor),
                shortlist_max=int(selection_shortlist_max) if selection_shortlist_max is not None else None,
                tier_widening=selection_tier_widening,
            )
        else:
            picked = ranked[: int(n_sites)]
            for idx, cand in enumerate(picked):
                selection_meta[cand.seq] = {
                    "selection_rank": idx + 1,
                    "selection_utility": None,
                    "nearest_selected_similarity": None,
                }
            selection_diag = {"shortlist_k": None, "tier_fraction_used": None, "tier_limit": None}
        retained_tier_counts = [0, 0, 0, 0]
        for cand in picked:
            retained_tier_counts[tier_by_seq[cand.seq]] += 1
        if len(ranked) < n_sites:
            msg_lines = [
                (
                    "Stage-A PWM sampling shortfall for motif "
                    f"'{context.get('motif_id')}' "
                    f"(width={context.get('width')}, strategy={context.get('strategy')}, "
                    f"length={context.get('length_label')}, window={context.get('window_label')}, "
                    f"score={context.get('score_label')})."
                ),
                (
                    f"Requested n_sites={context.get('n_sites')} "
                    f"-> candidates requested={context.get('requested_candidates')} "
                    f"generated={context.get('generated_candidates')}"
                    f"{context.get('cap_label')}."
                ),
                (f"Eligible unique sequences={len(ranked)} (need {n_sites})."),
            ]
            if context.get("length_observed"):
                msg_lines.append(f"Observed candidate lengths={context.get('length_observed')}.")
            suggestions = [
                "reduce n_sites",
                "increase mining.budget.max_candidates",
            ]
            if context.get("mining_max_seconds") is not None and context.get("mining_time_limited"):
                suggestions.append("increase mining.budget.max_seconds")
            if context.get("width") is not None and int(context.get("width")) <= 6:
                suggestions.append("try length.policy=range with a longer length.range")
            msg_lines.append("Try next: " + "; ".join(suggestions) + ".")
            log.warning(" ".join(msg_lines))
        tier_target_required_unique = None
        tier_target_met = None
        if budget_mode == "tier_target" and budget_target_tier_fraction is not None:
            tier_target_required_unique, tier_target_met = _evaluate_tier_target(
                n_sites=int(n_sites),
                target_tier_fraction=float(budget_target_tier_fraction),
                eligible_unique=len(ranked),
            )
            if not tier_target_met:
                warn_lines = [
                    f"Stage-A tier target unmet for motif '{motif.motif_id}': "
                    f"eligible_unique={len(ranked)} < required_unique={tier_target_required_unique} "
                    f"for target_tier_fraction={budget_target_tier_fraction}.",
                    "Retained set will spill beyond the target tier.",
                    "Try next: increase mining.budget.max_candidates; relax "
                    "mining.budget.target_tier_fraction; reduce n_sites.",
                ]
                log.warning(" ".join(warn_lines))
        n0, n1, n2, _n3 = _score_tier_counts(len(ranked))
        tier0_score = ranked[n0 - 1].score if n0 > 0 else None
        tier1_score = ranked[n0 + n1 - 1].score if n1 > 0 else None
        tier2_score = ranked[n0 + n1 + n2 - 1].score if n2 > 0 else None
        eligible_scores = [cand.score for cand in ranked]
        hist_edges, hist_counts = _build_score_hist(eligible_scores)
        if progress is not None:
            progress.update(
                generated=generated_total,
                accepted=eligible_hits,
                batch_index=batches if batches > 0 else None,
                batch_total=None,
                force=True,
            )
            progress.finish()
        log.info(
            "FIMO yield for motif %s: eligible=%d retained=%d",
            motif.motif_id,
            eligible_hits,
            len(picked),
            extra={"suppress_stdout": True},
        )
        meta_by_seq: dict[str, dict] = {}
        for cand in picked:
            meta = {
                "best_hit_score": cand.score,
                "rank_within_regulator": rank_by_seq[cand.seq],
                "tier": tier_by_seq[cand.seq],
                "fimo_start": cand.start,
                "fimo_stop": cand.stop,
                "fimo_strand": cand.strand,
            }
            meta["tfbs_core"] = _core_sequence(cand)
            if cand.matched_sequence:
                meta["fimo_matched_sequence"] = cand.matched_sequence
            selection_meta_row = selection_meta.get(cand.seq, {})
            meta["selection_rank"] = selection_meta_row.get("selection_rank")
            meta["selection_utility"] = selection_meta_row.get("selection_utility")
            meta["nearest_selected_similarity"] = selection_meta_row.get("nearest_selected_similarity")
            meta["selection_policy"] = selection_policy
            meta["selection_alpha"] = selection_alpha if selection_policy == "mmr" else None
            meta["selection_similarity"] = "contribution_l1" if selection_policy == "mmr" else None
            meta["selection_shortlist_min"] = selection_shortlist_min if selection_policy == "mmr" else None
            meta["selection_shortlist_factor"] = selection_shortlist_factor if selection_policy == "mmr" else None
            meta["selection_shortlist_max"] = selection_shortlist_max if selection_policy == "mmr" else None
            meta["selection_tier_fraction_used"] = selection_diag.get("tier_fraction_used")
            meta["selection_tier_limit"] = selection_diag.get("tier_limit")
            meta["shortlist_k"] = selection_diag.get("shortlist_k")
            meta["tier_target_fraction"] = budget_target_tier_fraction
            meta["tier_target_required_unique"] = tier_target_required_unique
            meta["tier_target_met"] = tier_target_met
            meta_by_seq[cand.seq] = meta
        if candidate_records is not None and debug_dir is not None:
            selected_set = {c.seq for c in picked}
            for row in candidate_records:
                if row.get("sequence") in selected_set:
                    row["selected"] = True
                    row["reject_reason"] = None
                elif row.get("accepted"):
                    row["reject_reason"] = "not_selected"
            try:
                path = _write_candidate_records(
                    candidate_records,
                    debug_output_dir=debug_dir,
                    debug_label=debug_label or motif.motif_id,
                    motif_id=motif.motif_id,
                    motif_hash=motif_hash,
                )
                log.info("FIMO candidate records written: %s", path)
            except Exception:
                log.warning("Failed to write FIMO candidate records.", exc_info=True)
        nearest_sims = [
            float(meta.get("nearest_selected_similarity"))
            for meta in selection_meta.values()
            if meta.get("selection_rank") is not None
            and int(meta.get("selection_rank")) > 1
            and meta.get("nearest_selected_similarity") is not None
        ]
        diversity_nearest_similarity_mean = None
        diversity_nearest_distance_mean = None
        diversity_nearest_distance_min = None
        if nearest_sims:
            diversity_nearest_similarity_mean = float(np.mean(nearest_sims))
            diversity_nearest_distance_mean = float(np.mean([(1.0 / sim) - 1.0 for sim in nearest_sims if sim > 0]))
            diversity_nearest_distance_min = float(min((1.0 / sim) - 1.0 for sim in nearest_sims if sim > 0))
        summary = None
        if return_summary:
            summary = _build_summary(
                generated=generated_total,
                target=requested,
                target_sites=n_sites,
                candidates_with_hit=candidates_with_hit,
                eligible_total=eligible_total,
                eligible=[cand.seq for cand in ranked],
                retained=[c.seq for c in picked],
                retained_scores=[cand.score for cand in picked],
                uniqueness_key=uniqueness_key,
                collapsed_by_core_identity=collapsed_by_core_identity,
                eligible_tier_counts=eligible_tier_counts,
                retained_tier_counts=retained_tier_counts,
                tier0_score=tier0_score,
                tier1_score=tier1_score,
                tier2_score=tier2_score,
                eligible_score_hist_edges=hist_edges,
                eligible_score_hist_counts=hist_counts,
                tier_target_fraction=budget_target_tier_fraction,
                tier_target_required_unique=tier_target_required_unique,
                tier_target_met=tier_target_met,
                selection_policy=selection_policy,
                selection_alpha=selection_alpha if selection_policy == "mmr" else None,
                selection_similarity="contribution_l1" if selection_policy == "mmr" else None,
                selection_shortlist_k=selection_diag.get("shortlist_k"),
                selection_shortlist_min=selection_shortlist_min if selection_policy == "mmr" else None,
                selection_shortlist_factor=selection_shortlist_factor if selection_policy == "mmr" else None,
                selection_shortlist_max=selection_shortlist_max if selection_policy == "mmr" else None,
                selection_tier_fraction_used=selection_diag.get("tier_fraction_used"),
                selection_tier_limit=selection_diag.get("tier_limit"),
                diversity_nearest_similarity_mean=diversity_nearest_similarity_mean,
                diversity_nearest_distance_mean=diversity_nearest_distance_mean,
                diversity_nearest_distance_min=diversity_nearest_distance_min,
                input_name=input_name,
                regulator=motif.motif_id,
                backend=scoring_backend,
            )
        return [c.seq for c in picked], meta_by_seq, summary

    if strategy == "consensus":
        progress = _PwmSamplingProgress(
            motif_id=motif.motif_id,
            backend=scoring_backend,
            target=1,
            accepted_target=n_sites,
            stream=sys.stdout,
        )
        seq = "".join(max(row.items(), key=lambda kv: kv[1])[0] for row in matrix)
        target_len = _resolve_length()
        full_seq = _embed_with_background(seq, target_len)
        selected, meta, summary = _score_with_fimo(
            n_candidates=1,
            requested=1,
            sequences=[full_seq],
        )
        if return_metadata and return_summary:
            return selected, meta, summary
        if return_metadata:
            return selected, meta
        if return_summary:
            return selected, summary
        return selected

    if budget_mode == "fixed_candidates":
        requested_candidates = max(1, int(budget_candidates))
    else:
        base_target = _mining_attr(mining, "batch_size", n_sites)
        if budget_min_candidates is not None:
            base_target = max(int(base_target), int(budget_min_candidates))
        requested_candidates = max(1, int(base_target))
        if budget_max_candidates is not None:
            requested_candidates = min(requested_candidates, int(budget_max_candidates))
    n_candidates = max(1, int(requested_candidates))
    progress = _PwmSamplingProgress(
        motif_id=motif.motif_id,
        backend=scoring_backend,
        target=requested_candidates,
        accepted_target=n_sites,
        stream=sys.stdout,
    )
    selected, meta, summary = _score_with_fimo(
        requested=requested_candidates,
        n_candidates=n_candidates,
    )
    if return_metadata and return_summary:
        return selected, meta, summary
    if return_metadata:
        return selected, meta
    if return_summary:
        return selected, summary
    return selected
