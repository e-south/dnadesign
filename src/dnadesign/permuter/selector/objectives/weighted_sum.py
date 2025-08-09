"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/selector/objectives/weighted_sum.py

Weighted-sum objective over normalized metrics.

Defaults:
  - normalization: rank-based in [0,1], scope=round
  - goal-aware: if goal=min → invert (1 - norm)
  - fallback: if <5 unique raw values, use z2cdf (median/MAD → Φ(z))

Writes a per-round sidecar `norm_stats_r{N}.json` in output_dir.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import Objective


def _avg_rank_0_1(values: np.ndarray) -> np.ndarray:
    """
    Average ranks in [0,1] with ties, stable and deterministic.
    For n==1 returns 0.0.
    """
    n = len(values)
    if n == 0:
        return values.astype(float)
    if n == 1:
        return np.array([0.0], dtype=float)

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        vi = values[order[i]]
        while j < n and values[order[j]] == vi:
            j += 1
        # average rank (1-based)
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    # scale to [0,1]
    return (ranks - 1.0) / (n - 1.0)


def _z2cdf(values: np.ndarray) -> np.ndarray:
    """
    Robust z-score via median/MAD → map to Φ(z). Clip to [0.01, 0.99].
    """
    if values.size == 0:
        return values.astype(float)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        z = np.zeros_like(values, dtype=float)
    else:
        # 1.4826 ≈ consistency constant to approximate std from MAD
        z = (values - med) / (1.4826 * mad)
    # standard normal CDF via erf
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    return np.clip(cdf, 0.01, 0.99)


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin == 0:
        return np.full_like(values, 0.5, dtype=float)
    return (values - vmin) / (vmax - vmin)


def _identity(values: np.ndarray) -> np.ndarray:
    # Assumes already comparable/bounded; still clip defensively.
    return np.clip(values.astype(float), 0.0, 1.0)


_METHODS = {
    "rank": _avg_rank_0_1,
    "z2cdf": _z2cdf,
    "minmax": _minmax,
    "identity": _identity,
}


class WeightedSumObjective(Objective):
    def compute(
        self,
        variants: List[Dict],
        *,
        metrics_cfg: List[Dict],
        objective_cfg: Dict,
        job_ctx: Optional[Dict] = None,
    ) -> None:
        weights: Dict[str, float] = objective_cfg.get("weights", {})
        # strict: weights keys must equal declared metric ids
        metric_ids = [m["id"] for m in metrics_cfg]
        if set(weights.keys()) != set(metric_ids):
            raise ValueError(
                f"objective.weights keys must match metric ids exactly; "
                f"got {sorted(weights)} expected {sorted(metric_ids)}"
            )
        # normalize weights to sum 1.0 (but reject negatives)
        w_arr = np.array([float(weights[mid]) for mid in metric_ids], dtype=float)
        if np.any(w_arr < 0):
            raise ValueError("objective.weights must be non-negative")
        if not np.any(w_arr > 0):  # all zeros
            raise ValueError("objective.weights cannot be all zeros")
        w_arr = w_arr / float(np.sum(w_arr))
        weights = {mid: float(w) for mid, w in zip(metric_ids, w_arr)}

        # collect raw arrays
        raw_by_metric: Dict[str, np.ndarray] = {}
        for mid in metric_ids:
            vals = []
            for v in variants:
                if "metrics" not in v or mid not in v["metrics"]:
                    raise ValueError(
                        f"Missing raw metric '{mid}' for variant {v.get('var_id','<no-id>')}"
                    )
                val = float(v["metrics"][mid])
                if not math.isfinite(val):
                    raise ValueError(
                        f"Non-finite raw metric '{mid}' for variant {v.get('var_id','<no-id>')}"
                    )
                vals.append(val)
            raw_by_metric[mid] = np.asarray(vals, dtype=float)

        # prepare normalization settings per metric
        # defaults: method=rank, scope=round
        norm_used: Dict[str, Dict[str, object]] = {}
        norm_matrix = np.zeros((len(variants), len(metric_ids)), dtype=float)

        for j, m in enumerate(metrics_cfg):
            mid = m["id"]
            goal = str(m.get("goal", "max")).lower()
            norm_cfg = m.get("norm", {})
            method = str(norm_cfg.get("method", "rank")).lower()
            scope = str(norm_cfg.get("scope", "round")).lower()
            arr = raw_by_metric[mid]
            tie_fallback = False

            # choose normalization
            func = _METHODS.get(method)
            if func is None:
                raise ValueError(
                    f"Unknown normalization method '{method}' for metric '{mid}'"
                )

            # rank fallback if too few unique values
            if method == "rank":
                unique = int(np.unique(arr).size)
                if unique < 5:
                    func = _METHODS["z2cdf"]
                    tie_fallback = True

            # apply
            norm_vals = func(arr)

            # direction handling
            if goal == "min":
                norm_vals = 1.0 - norm_vals
            elif goal != "max":
                raise ValueError(f"Invalid goal '{goal}' for metric '{mid}'")

            norm_matrix[:, j] = norm_vals

            norm_used[mid] = {
                "method": "z2cdf" if tie_fallback else method,
                "scope": scope,
                "unique": int(np.unique(arr).size),
                "tie_fallback": bool(tie_fallback),
            }

        # compute weighted sum
        weight_vec = np.array([weights[mid] for mid in metric_ids], dtype=float)
        scores = norm_matrix @ weight_vec

        # write sidecar norm_stats for this round (if job_ctx provided)
        norm_stats_id = None
        if job_ctx and job_ctx.get("output_dir") and job_ctx.get("round") is not None:
            out_dir = Path(job_ctx["output_dir"])
            rnd = int(job_ctx["round"])
            sidecar = {
                "round": rnd,
                "metrics": norm_used,
            }
            text = json.dumps(sidecar, separators=(",", ":"), sort_keys=True)
            # simple deterministic id
            norm_stats_id = f"r{rnd}_{abs(hash(text)) & 0xFFFFFF:06X}"
            sidecar_path = out_dir / f"norm_stats_r{rnd}.json"
            sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        # mutate variants
        for i, v in enumerate(variants):
            v["norm_metrics"] = {
                mid: float(norm_matrix[i, j]) for j, mid in enumerate(metric_ids)
            }
            v["objective_score"] = float(scores[i])
            v["objective_meta"] = {
                "type": "weighted_sum",
                "weights": weights,
                "norm_scope": "round",  # default; per-metric details in sidecar
                "norm_stats_id": norm_stats_id or "",
            }
