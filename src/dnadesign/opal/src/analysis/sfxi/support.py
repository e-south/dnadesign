"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/sfxi/support.py

Support and extrapolation diagnostics for SFXI logic space.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .state_order import require_state_order


def _validate_matrix(arr: np.ndarray, *, name: str, n_cols: int) -> np.ndarray:
    mat = np.asarray(arr, dtype=float)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.shape[1] != n_cols:
        raise ValueError(f"{name} must have shape (n, {n_cols}).")
    if not np.all(np.isfinite(mat)):
        raise ValueError(f"{name} must be finite.")
    return mat


def _min_l2_distance(
    candidates: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    if labels.shape[0] == 0:
        raise ValueError("labels must be non-empty for distance computation.")
    n = candidates.shape[0]
    out = np.empty(n, dtype=float)
    bs = int(batch_size)
    if bs <= 0:
        raise ValueError("batch_size must be >= 1.")
    for start in range(0, n, bs):
        stop = min(start + bs, n)
        batch = candidates[start:stop]
        diffs = batch[:, None, :] - labels[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        out[start:stop] = dists.min(axis=1)
    return out


def dist_to_labeled_logic(
    candidates: np.ndarray,
    labels: np.ndarray,
    *,
    state_order: Sequence[str] | None = None,
    batch_size: int = 2048,
) -> np.ndarray:
    require_state_order(state_order)
    cand = _validate_matrix(candidates, name="candidates", n_cols=4)
    lab = _validate_matrix(labels, name="labels", n_cols=4)
    return _min_l2_distance(cand, lab, batch_size=batch_size)
