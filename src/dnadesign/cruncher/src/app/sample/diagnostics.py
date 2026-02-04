"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/diagnostics.py

Aggregate diagnostics and scoring summaries for sampling workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.cruncher.config.schema_v2 import SampleConfig
from dnadesign.cruncher.core.evaluator import SequenceEvaluator
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.state import SequenceState

_AUTO_OPT_BOOTSTRAP_SAMPLES = 300
_AUTO_OPT_BOOTSTRAP_PCT = (5.0, 95.0)


@dataclass
class _EliteCandidate:
    seq_arr: np.ndarray
    chain_id: int
    draw_idx: int
    combined_score: float
    min_norm: float
    sum_norm: float
    per_tf_map: dict[str, float]
    norm_map: dict[str, float]


def _norm_map_for_elites(
    seq_arr: np.ndarray,
    per_tf_map: dict[str, float],
    *,
    scorer: Scorer,
    score_scale: str,
) -> dict[str, float]:
    if score_scale.lower() == "normalized-llr":
        return {tf: float(per_tf_map.get(tf, 0.0)) for tf in scorer.tf_names}
    return scorer.normalized_llr_map(seq_arr)


def _elite_filter_passes(
    *,
    norm_map: dict[str, float],
    min_norm: float,
    sum_norm: float,
    min_per_tf_norm: float | None,
    require_all_tfs_over_min_norm: bool,
    pwm_sum_min: float,
) -> bool:
    if min_per_tf_norm is not None:
        if require_all_tfs_over_min_norm:
            if not all(score >= min_per_tf_norm for score in norm_map.values()):
                return False
        else:
            if min_norm < min_per_tf_norm:
                return False
    if pwm_sum_min > 0 and sum_norm < pwm_sum_min:
        return False
    return True


def _filter_elite_candidates(
    candidates: list[_EliteCandidate],
    *,
    min_per_tf_norm: float | None,
    require_all_tfs_over_min_norm: bool,
    pwm_sum_min: float,
) -> list[_EliteCandidate]:
    filtered: list[_EliteCandidate] = []
    for cand in candidates:
        if _elite_filter_passes(
            norm_map=cand.norm_map,
            min_norm=cand.min_norm,
            sum_norm=cand.sum_norm,
            min_per_tf_norm=min_per_tf_norm,
            require_all_tfs_over_min_norm=require_all_tfs_over_min_norm,
            pwm_sum_min=pwm_sum_min,
        ):
            filtered.append(cand)
    return filtered


def _elite_rank_key(combined_score: float, min_norm: float, sum_norm: float) -> tuple[float, float, float]:
    return (combined_score, min_norm, sum_norm)


def resolve_dsdna_mode(*, elites_cfg: object, bidirectional: bool) -> bool:
    selection = getattr(elites_cfg, "selection", None)
    policy = getattr(selection, "policy", "top_score") if selection is not None else "top_score"
    if policy == "top_score":
        return bool(getattr(elites_cfg, "dsDNA_canonicalize", False))
    distance = getattr(selection, "distance", None)
    mode = getattr(distance, "dsDNA", "auto") if distance is not None else "auto"
    if mode == "auto":
        return bool(bidirectional)
    if mode == "true":
        return True
    if mode == "false":
        return False
    raise ValueError(f"Unknown dsDNA mode '{mode}'.")


def dsdna_equivalence_enabled(sample_cfg: SampleConfig) -> bool:
    return resolve_dsdna_mode(
        elites_cfg=sample_cfg.elites,
        bidirectional=sample_cfg.objective.bidirectional,
    )


def _draw_scores_from_sequences(seq_df: pd.DataFrame) -> np.ndarray:
    if "combined_score_final" not in seq_df.columns:
        return np.array([], dtype=float)
    df = seq_df
    if "phase" in df.columns:
        df = df[df["phase"] == "draw"]
    series = pd.to_numeric(df["combined_score_final"], errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return np.array([], dtype=float)
    return series.to_numpy(dtype=float)


def _best_score_final_from_sequences(seq_df: pd.DataFrame) -> float | None:
    scores = _draw_scores_from_sequences(seq_df)
    if scores.size == 0:
        return None
    return float(np.max(scores))


def _top_k_median_from_scores(scores: np.ndarray, k: int) -> float | None:
    if scores.size == 0:
        return None
    k = max(1, min(int(k), int(scores.size)))
    if k >= scores.size:
        return float(np.median(scores))
    topk = np.partition(scores, scores.size - k)[scores.size - k :]
    return float(np.median(topk))


def _top_k_median_from_sequences(seq_df: pd.DataFrame, *, k: int) -> float | None:
    scores = _draw_scores_from_sequences(seq_df)
    return _top_k_median_from_scores(scores, k)


def _bootstrap_top_k_ci(
    scores: np.ndarray,
    *,
    k: int,
    rng: np.random.Generator,
    n_boot: int = _AUTO_OPT_BOOTSTRAP_SAMPLES,
) -> tuple[float, float] | None:
    if scores.size == 0:
        return None
    if scores.size == 1:
        value = float(scores[0])
        return value, value
    n_boot = max(1, int(n_boot))
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, scores.size, size=scores.size)
        sample = scores[idx]
        boot[i] = _top_k_median_from_scores(sample, k) or float("nan")
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return None
    low, high = np.percentile(boot, _AUTO_OPT_BOOTSTRAP_PCT)
    return float(low), float(high)


def _bootstrap_seed_payload(payload: dict[str, object]) -> int:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _bootstrap_seed(
    *,
    manifest: dict[str, object],
    run_dir: Path,
    kind: str,
) -> int:
    _ = run_dir  # unused but retained for call sites; keep signature stable
    auto_meta = manifest.get("auto_opt") or {}
    regulator_set = manifest.get("regulator_set") or {}
    payload = {
        "seed": manifest.get("seed"),
        "kind": kind,
        "attempt": auto_meta.get("attempt"),
        "candidate": auto_meta.get("candidate"),
        "budget": auto_meta.get("budget"),
        "replicate": auto_meta.get("replicate"),
        "length": manifest.get("sequence_length"),
        "tfs": regulator_set.get("tfs"),
    }
    return _bootstrap_seed_payload(payload)


def _pooled_bootstrap_seed(
    *,
    manifests: list[dict[str, object]],
    kind: str,
    length: int | None,
    budget: int | None,
) -> int | None:
    if not manifests:
        return None
    replicates: list[dict[str, object]] = []
    for manifest in manifests:
        auto_meta = manifest.get("auto_opt") or {}
        replicates.append(
            {
                "seed": manifest.get("seed"),
                "attempt": auto_meta.get("attempt"),
                "candidate": auto_meta.get("candidate"),
                "budget": auto_meta.get("budget"),
                "replicate": auto_meta.get("replicate"),
                "length": manifest.get("sequence_length"),
            }
        )
    payload = {
        "kind": kind,
        "length": length,
        "budget": budget,
        "replicates": sorted(replicates, key=lambda item: json.dumps(item, sort_keys=True)),
    }
    return _bootstrap_seed_payload(payload)


def _polish_sequence(
    seq_arr: np.ndarray,
    *,
    evaluator: SequenceEvaluator,
    beta_softmin_final: float | None,
    max_rounds: int,
    improvement_tol: float,
    max_evals: int | None,
) -> np.ndarray:
    seq = seq_arr.copy()
    evals = 0

    def _score() -> float:
        nonlocal evals
        evals += 1
        return evaluator.combined(SequenceState(seq), beta=beta_softmin_final)

    best_score = _score()
    for _ in range(max_rounds):
        improved = False
        for i in range(seq.size):
            old_base = seq[i]
            best_base = old_base
            best_local = best_score
            for b in range(4):
                if b == old_base:
                    continue
                seq[i] = b
                score = _score()
                if score > best_local + improvement_tol:
                    best_local = score
                    best_base = b
                if max_evals is not None and evals >= max_evals:
                    seq[i] = best_base
                    return seq
            seq[i] = best_base
            if best_base != old_base:
                best_score = best_local
                improved = True
            if max_evals is not None and evals >= max_evals:
                return seq
        if not improved:
            break
    return seq
