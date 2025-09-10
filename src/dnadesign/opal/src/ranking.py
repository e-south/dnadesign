"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/ranking.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np


def selection_threshold_for_top_k(scores_desc: np.ndarray, k: int) -> float:
    if scores_desc.size == 0 or k <= 0:
        return float("inf")
    idx = min(k - 1, scores_desc.size - 1)
    return float(scores_desc[idx])
