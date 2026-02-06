"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/diversity.py

Compute TFBS-core diversity metrics for elites.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from dnadesign.cruncher.core.pwm import PWM


def _info_norm_weights(pwm: PWM) -> np.ndarray:
    matrix = np.asarray(pwm.matrix, dtype=float)
    p = matrix + 1e-9
    info = 2.0 + (p * np.log2(p)).sum(axis=1)
    max_info = float(np.max(info)) if info.size else 0.0
    if max_info <= 0:
        return np.ones_like(info, dtype=float)
    info_norm = info / max_info
    weights = 1.0 - info_norm
    if float(weights.sum()) <= 0:
        return np.ones_like(info, dtype=float)
    return weights


def _weighted_hamming(a: str, b: str, weights: np.ndarray) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    mismatches = np.fromiter((c0 != c1 for c0, c1 in zip(a, b)), dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    return float(np.dot(weights, mismatches) / total)


def _tfbs_core_map(
    hits_df: pd.DataFrame,
    tf_names: Iterable[str],
) -> dict[str, dict[str, str]]:
    tf_list = list(tf_names)
    core_by_elite: dict[str, dict[str, str]] = {}
    if hits_df is None or hits_df.empty:
        return core_by_elite
    for _, row in hits_df.iterrows():
        elite_id = str(row.get("elite_id") or "")
        tf_name = row.get("tf")
        if not elite_id or tf_name not in tf_list:
            continue
        core_seq = row.get("best_core_seq")
        if not isinstance(core_seq, str) or not core_seq:
            continue
        core_by_elite.setdefault(elite_id, {})[str(tf_name)] = core_seq
    return core_by_elite


def compute_elite_distance_matrix(
    hits_df: pd.DataFrame,
    tf_names: Iterable[str],
    pwms: dict[str, PWM],
) -> tuple[list[str], np.ndarray]:
    tf_list = list(tf_names)
    core_by_elite = _tfbs_core_map(hits_df, tf_list)
    elite_ids = sorted(core_by_elite.keys())
    n = len(elite_ids)
    dist = np.full((n, n), np.nan, dtype=float)
    if n == 0:
        return elite_ids, dist

    weights_by_tf: dict[str, np.ndarray] = {}
    for tf in tf_list:
        pwm = pwms.get(tf)
        if pwm is None:
            continue
        weights_by_tf[tf] = _info_norm_weights(pwm)

    for i, elite_i in enumerate(elite_ids):
        dist[i, i] = 0.0
        for j in range(i + 1, n):
            elite_j = elite_ids[j]
            per_tf = []
            for tf in tf_list:
                core_i = core_by_elite[elite_i].get(tf)
                core_j = core_by_elite[elite_j].get(tf)
                weights = weights_by_tf.get(tf)
                if core_i is None or core_j is None or weights is None:
                    continue
                per_tf.append(_weighted_hamming(core_i, core_j, weights))
            if not per_tf:
                value = float("nan")
            else:
                value = float(np.nanmean(per_tf))
            dist[i, j] = value
            dist[j, i] = value
    return elite_ids, dist


def _representative_by_identity(
    identity_by_elite_id: dict[str, str],
    rank_by_elite_id: dict[str, int] | None,
) -> dict[str, str]:
    rep_by_identity: dict[str, str] = {}
    for elite_id, identity in identity_by_elite_id.items():
        rep = rep_by_identity.get(identity)
        if rep is None:
            rep_by_identity[identity] = elite_id
            continue
        if rank_by_elite_id is None:
            if elite_id < rep:
                rep_by_identity[identity] = elite_id
            continue
        rank_new = rank_by_elite_id.get(elite_id)
        rank_old = rank_by_elite_id.get(rep)
        if rank_new is None or rank_old is None:
            if elite_id < rep:
                rep_by_identity[identity] = elite_id
            continue
        if rank_new < rank_old:
            rep_by_identity[identity] = elite_id
    return rep_by_identity


def representative_elite_ids(
    identity_by_elite_id: dict[str, str],
    rank_by_elite_id: dict[str, int] | None = None,
) -> dict[str, str]:
    if not identity_by_elite_id:
        return {}
    return _representative_by_identity(identity_by_elite_id, rank_by_elite_id)


def compute_elites_nn_distance_table(
    hits_df: pd.DataFrame,
    tf_names: Iterable[str],
    pwms: dict[str, PWM],
    *,
    identity_mode: str,
    identity_by_elite_id: dict[str, str] | None = None,
    rank_by_elite_id: dict[str, int] | None = None,
) -> pd.DataFrame:
    rep_by_identity: dict[str, str] | None = None
    if identity_by_elite_id:
        rep_by_identity = _representative_by_identity(identity_by_elite_id, rank_by_elite_id)
        keep_ids = set(rep_by_identity.values())
        hits_df = hits_df[hits_df["elite_id"].isin(keep_ids)].copy()

    elite_ids, dist = compute_elite_distance_matrix(hits_df, tf_names, pwms)
    if dist.size == 0:
        return pd.DataFrame(columns=["elite_id", "nn_dist", "mean_dist", "min_dist", "identity_mode"])

    rows: list[dict[str, object]] = []
    for i, elite_id in enumerate(elite_ids):
        row_dist = dist[i, :].astype(float)
        other = np.delete(row_dist, i)
        if other.size == 0:
            nn = None
            mean = None
            min_val = None
        else:
            nn = float(np.nanmin(other))
            mean = float(np.nanmean(other))
            min_val = nn
        rows.append(
            {
                "elite_id": elite_id,
                "nn_dist": nn,
                "mean_dist": mean,
                "min_dist": min_val,
                "identity_mode": identity_mode,
            }
        )
    nn_df = pd.DataFrame(rows)
    if not identity_by_elite_id or not rep_by_identity:
        return nn_df
    metrics_by_rep = {str(row["elite_id"]): row for row in nn_df.to_dict(orient="records")}
    expanded_rows: list[dict[str, object]] = []
    for elite_id, identity in identity_by_elite_id.items():
        rep_id = rep_by_identity.get(identity)
        if rep_id is None:
            continue
        metrics = metrics_by_rep.get(str(rep_id))
        if metrics is None:
            continue
        row = dict(metrics)
        row["elite_id"] = elite_id
        expanded_rows.append(row)
    return pd.DataFrame(expanded_rows)


def summarize_elite_distances(dist: np.ndarray) -> dict[str, float | None]:
    if dist.size == 0:
        return {
            "mean_pairwise_distance": None,
            "min_pairwise_distance": None,
        }
    upper = dist[np.triu_indices(dist.shape[0], k=1)]
    upper = upper[np.isfinite(upper)]
    if upper.size == 0:
        return {
            "mean_pairwise_distance": None,
            "min_pairwise_distance": None,
        }
    return {
        "mean_pairwise_distance": float(np.mean(upper)),
        "min_pairwise_distance": float(np.min(upper)),
    }
