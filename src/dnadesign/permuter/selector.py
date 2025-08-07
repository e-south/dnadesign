"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/selector.py

<dnadesign project>
dnadesign/permuter/selector.py

Core selection logic: from a pool of scored variants, pick the “elites”
according to a user-specified strategy (e.g. top-k or threshold).

This module is invoked after each evaluation round:
  1. Collect all variant dicts, each with a numeric "score" field.
  2. Apply the selection rule to choose a subset for the next round.
  3. Return the list of elite variants, preserving their full metadata.

Module Author(s): Eric J. South
Dunlop Lab

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def select(variants: List[Dict], cfg: Dict) -> List[Dict]:
    """
    Choose a subset of variants ("elites") based on the supplied configuration.

    Args:
      variants: list of variant dicts, each must contain a numeric "score" key.
      cfg: selection configuration dict with keys:
        - strategy: "top_k" or "threshold"
        - k: (int) number of elites when strategy == "top_k"
        - include_ties: (bool, optional) whether to include all tied at the k-th score
        - threshold: (float) minimum score when strategy == "threshold"

    Returns:
      A list of variant dicts that meet the selection criterion, in their
      original order.

    Raises:
      ValueError: if an unsupported strategy is provided.
    """
    strategy = cfg["strategy"]
    # Extract scores into a NumPy array for easy sorting and comparison
    scores = np.array([v["score"] for v in variants])

    if strategy == "top_k":
        # Select the top k variants by score (highest first).
        k = cfg["k"]
        # argsort(-scores) gives descending order of indices
        order = np.argsort(-scores)
        # Determine the cutoff score: the k-th best score (or min if fewer variants)
        cutoff_score = scores[order[k - 1]] if len(scores) >= k else scores.min()
        # Build a boolean mask of all variants with score >= cutoff_score
        if cfg.get("include_ties", True):
            mask = scores >= cutoff_score
        else:
            # Include exactly the first k indices
            mask = np.zeros_like(scores, dtype=bool)
            mask[order[:k]] = True

    elif strategy == "threshold":
        # Select all variants whose score meets or exceeds the threshold
        thr = cfg["threshold"]
        mask = scores >= thr

    else:
        # Defensive: reject unknown strategies early
        raise ValueError(f"Unknown selection strategy: {strategy!r}")

    # Return only the variants marked as True in the mask
    elites = [variant for variant, keep in zip(variants, mask) if keep]
    return elites
