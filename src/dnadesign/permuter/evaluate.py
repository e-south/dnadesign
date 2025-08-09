"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/evaluate.py

Thin façade so that the rest of the pipeline needn't import evaluator internals.
Provides helpers to score batches across multiple metrics.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .evaluators import get_evaluator


def evaluate(
    sequences: List[str],
    *,
    evaluator_name: str,
    metric: str,
    evaluator_params: dict,
    ref_sequence: str | None = None,
) -> List[float]:
    """
    Score a batch of sequences according to the chosen evaluator.

    Notes:
      - The evaluator is responsible for handling any reference-specific
        requirements (e.g., computing an internal ref embedding) when
        `ref_sequence` is provided.

    Returns:
      A list of floats (higher is better).
    """
    evaluator = get_evaluator(evaluator_name, **(evaluator_params or {}))
    return evaluator.score(
        sequences,
        metric=metric,
        ref_sequence=ref_sequence,
        ref_embedding=None,
    )


def _freeze_params(params: dict | None) -> Tuple:
    """Make evaluator params hashable for caching evaluator instances."""
    if not params:
        return tuple()
    # naive freeze for flat dicts; extend if you pass nested structures
    return tuple(sorted(params.items()))


def evaluate_many(
    sequences: List[str],
    *,
    metrics_cfg: List[dict],
    ref_sequence: str | None,
) -> Dict[str, List[float]]:
    """
    Compute multiple metrics for the same list of sequences.

    Args:
      sequences: aligned list of sequences to score.
      metrics_cfg: list of metric configs (id, name, evaluator, params, ...).
      ref_sequence: reference sequence for ratio/distance metrics.

    Returns:
      dict metric_id -> list[float] (same order as `sequences`).

    Raises:
      ValueError if any evaluator produces non-finite values.
    """
    results: Dict[str, List[float]] = {}
    # cache evaluator instances keyed by (evaluator, params)
    cache: Dict[Tuple[str, Tuple], object] = {}

    for mc in metrics_cfg:
        mid = mc["id"]
        metric_name = mc["name"]
        evaluator_name = mc["evaluator"]
        params = mc.get("params", {}) or {}
        key = (evaluator_name, _freeze_params(params))
        if key not in cache:
            cache[key] = get_evaluator(evaluator_name, **params)
        evaluator = cache[key]
        # Each concrete evaluator implements .score()
        scores = evaluator.score(
            sequences,
            metric=metric_name,
            ref_sequence=ref_sequence,
            ref_embedding=None,
        )
        # sanity: all finite numbers
        arr = np.asarray(scores, dtype=float)
        if not np.isfinite(arr).all():
            bad_idx = int(np.where(~np.isfinite(arr))[0][0])
            raise ValueError(
                f"Non-finite value from evaluator '{evaluator_name}' for metric '{metric_name}' "
                f"at index {bad_idx}"
            )
        results[mid] = arr.tolist()
    return results
