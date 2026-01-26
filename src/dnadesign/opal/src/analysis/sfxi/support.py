"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/sfxi/support.py

Support and extrapolation diagnostics for SFXI logic and embedding spaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .state_order import STATE_ORDER, assert_state_order


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
    state_order: Sequence[str] = STATE_ORDER,
    batch_size: int = 2048,
) -> np.ndarray:
    assert_state_order(state_order)
    cand = _validate_matrix(candidates, name="candidates", n_cols=4)
    lab = _validate_matrix(labels, name="labels", n_cols=4)
    return _min_l2_distance(cand, lab, batch_size=batch_size)


def dist_to_labeled_x(
    candidates: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    cand_raw = np.asarray(candidates)
    lab_raw = np.asarray(labels)
    cand_cols = cand_raw.shape[0] if cand_raw.ndim == 1 else cand_raw.shape[1]
    lab_cols = lab_raw.shape[0] if lab_raw.ndim == 1 else lab_raw.shape[1]
    cand = _validate_matrix(candidates, name="candidates", n_cols=int(cand_cols))
    lab = _validate_matrix(labels, name="labels", n_cols=int(lab_cols))
    if cand.shape[1] != lab.shape[1]:
        raise ValueError("candidates and labels must have matching dimensions.")
    return _min_l2_distance(cand, lab, batch_size=batch_size)
