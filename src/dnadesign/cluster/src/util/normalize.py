"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/util/normalize.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np


def normalize_array(arr: np.ndarray, method: str = "robust") -> np.ndarray:
    if method == "z":
        mean = np.mean(arr)
        std = np.std(arr)
        return (arr - mean) / (std if std else 1.0)
    if method == "robust":
        median = np.median(arr)
        q75, q25 = np.percentile(arr, 75), np.percentile(arr, 25)
        iqr = q75 - q25
        return (arr - median) / (iqr if iqr else 1.0)
    return arr
