"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/selector/strategies/top_k.py

Top-K selection strategy, with optional tie inclusion.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import Strategy


class TopKStrategy(Strategy):
    def select(self, variants: List[Dict], *, cfg: Dict) -> List[Dict]:
        if not variants:
            return []
        if "k" not in cfg or int(cfg["k"]) <= 0:
            raise ValueError("top_k strategy requires positive `k`")
        include_ties = bool(cfg.get("include_ties", True))
        k = int(cfg["k"])

        scores = np.array([float(v["objective_score"]) for v in variants], dtype=float)
        order = np.argsort(-scores, kind="mergesort")  # stable desc
        if include_ties:
            if len(scores) >= k:
                kth = scores[order[k - 1]]
            else:
                kth = float(np.min(scores))
            mask = scores >= kth
        else:
            mask = np.zeros_like(scores, dtype=bool)
            mask[order[:k]] = True

        elites = [v for v, keep in zip(variants, mask) if keep]
        return elites
