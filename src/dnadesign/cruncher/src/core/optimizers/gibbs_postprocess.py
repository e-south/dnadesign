"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/gibbs_postprocess.py

Provide Gibbs optimiser postprocess helpers for trace assembly and elite picking.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
from typing import Callable

import numpy as np

from dnadesign.cruncher.core.sequence import hamming_distance
from dnadesign.cruncher.core.state import SequenceState


def build_trace_idata(*, build_trace: bool, chain_scores: list[list[float]]) -> object | None:
    if not build_trace:
        return None
    if chain_scores:
        max_len = max(len(scores) for scores in chain_scores)
        for scores in chain_scores:
            if len(scores) < max_len:
                scores.extend([float("nan")] * (max_len - len(scores)))
    score_arr = np.asarray(chain_scores, dtype=float)
    az = importlib.import_module("arviz")
    return az.from_dict(posterior={"score": score_arr})


def select_diverse_elites(
    *,
    top_k: int,
    min_dist: int,
    all_samples: list[np.ndarray],
    all_scores: list[dict[str, float]],
    all_meta: list[tuple[int, int]],
    beta_softmin_final: float | None,
    combined_from_scores: Callable[[dict[str, float], float | None, int], float],
) -> tuple[list[SequenceState], list[tuple[int, int]]]:
    if top_k <= 0:
        return [], []
    ranked: list[tuple[float, np.ndarray, int]] = []
    for idx, (seq, per_tf_map) in enumerate(zip(all_samples, all_scores)):
        val = combined_from_scores(per_tf_map, beta_softmin_final, int(seq.size))
        ranked.append((val, seq.copy(), idx))
    ranked.sort(key=lambda item: item[0], reverse=True)
    elites: list[SequenceState] = []
    picked_idx: list[int] = []
    for _, seq, idx in ranked:
        if len(elites) >= top_k:
            break
        if any(hamming_distance(seq, elite.seq) < min_dist for elite in elites):
            continue
        elites.append(SequenceState(seq))
        picked_idx.append(idx)
    elites_meta = [all_meta[idx] for idx in picked_idx]
    return elites, elites_meta


def make_move_stat_entry(
    *,
    sweep_idx: int,
    phase: str,
    chain_index: int,
    move_kind: str,
    accepted: bool,
    move_detail_payload: dict[str, object],
) -> dict[str, object]:
    return {
        "sweep_idx": int(sweep_idx),
        "phase": str(phase),
        "chain": int(chain_index),
        "move_kind": move_kind,
        "attempted": 1,
        "accepted": int(bool(accepted)),
        "delta": move_detail_payload.get("delta"),
        "score_old": move_detail_payload.get("score_old"),
        "score_new": move_detail_payload.get("score_new"),
        "delta_hamming": move_detail_payload.get("delta_hamming"),
        "gibbs_changed": move_detail_payload.get("gibbs_changed"),
    }
