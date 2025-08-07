"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/selector.py

Select top variants by strategy.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from typing import Dict, List

import numpy as np


def select(variants: List[Dict], selector_cfg: Dict) -> List[Dict]:
    strategy = selector_cfg["strategy"]
    scores = np.array([v["score"] for v in variants])
    if strategy == "top_k":
        k = selector_cfg.get("k")
        assert k and k > 0, "k>0 required for top_k"
        idx = np.argsort(-scores)[:k]
    elif strategy == "threshold":
        thr = selector_cfg.get("threshold")
        assert thr is not None, "threshold required"
        idx = np.where(scores >= thr)[0]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return [variants[i] for i in idx]
