"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/evaluate.py

Thin façade so that the rest of the pipeline needn't import evaluator internals.
Provides a single `evaluate()` call that handles both direct metrics and
embedding-based distances (with optional caching of the reference embedding).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .evaluators import get_evaluator


def evaluate(
    sequences: List[str],
    *,
    evaluator_name: str,
    metric: str,
    evaluator_params: dict,
    ref_sequence: str | None = None,
    ref_embedding_cache: dict | None = None,
) -> List[float]:
    """
    Score a batch of sequences according to the chosen evaluator.

    Args:
      sequences: list of DNA/protein strings to score.
      evaluator_name: key in our registry (e.g. "placeholder").
      metric: which metric to compute (e.g. "log_likelihood_ratio").
      evaluator_params: any kwargs for the evaluator constructor.
      ref_sequence: required for ratio-based metrics.
      ref_embedding_cache: optional dict to store/reuse the reference embedding
                           when using embedding_distance.

    Returns:
      A list of floats, one per input sequence, where higher is better.
    """
    # Instantiate (and cache) the concrete evaluator class
    evaluator = get_evaluator(evaluator_name, **evaluator_params)

    ref_embed: Optional[List[float]] = None

    # If embedding_distance, we need to compute the reference embedding once
    if metric == "embedding_distance":
        # If not yet cached, compute via the evaluator’s stub
        if ref_embedding_cache is None or "embed" not in ref_embedding_cache:
            # We call the stub with metric="log_likelihood" to reuse code
            # (PlaceholderEvaluator returns deterministic floats)
            emb = evaluator.score([ref_sequence], metric="log_likelihood")
            ref_embed = emb[0]
            if ref_embedding_cache is not None:
                ref_embedding_cache["embed"] = ref_embed
        else:
            ref_embed = ref_embedding_cache["embed"]

    # Delegate scoring to the evaluator
    return evaluator.score(
        sequences,
        metric=metric,
        ref_sequence=ref_sequence,
        ref_embedding=np.asarray(ref_embed) if ref_embed is not None else None,
    )
