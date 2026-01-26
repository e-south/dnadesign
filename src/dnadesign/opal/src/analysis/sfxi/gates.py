"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/sfxi/gates.py

Truth-table library and nearest-gate assignment for SFXI logic vectors.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from .state_order import STATE_ORDER, assert_state_order


@dataclass(frozen=True)
class GateSpec:
    code: str
    vector: np.ndarray


_TRUTH_TABLES = np.asarray(list(product([0.0, 1.0], repeat=4)), dtype=float)
_GATE_CLASSES = ["".join(str(int(v)) for v in row.tolist()) for row in _TRUTH_TABLES]
GATE_LIBRARY = tuple(GateSpec(code=cls, vector=row.copy()) for cls, row in zip(_GATE_CLASSES, _TRUTH_TABLES))


def truth_table_vectors() -> np.ndarray:
    return _TRUTH_TABLES.copy()


def nearest_gate(
    v: np.ndarray,
    *,
    state_order: Sequence[str] = STATE_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    assert_state_order(state_order)
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 4:
        raise ValueError("nearest_gate expects vectors of shape (n, 4).")
    diffs = arr[:, None, :] - _TRUTH_TABLES[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    idx = np.argmin(dists, axis=1)
    classes = np.asarray([_GATE_CLASSES[i] for i in idx], dtype=object)
    min_dists = dists[np.arange(dists.shape[0]), idx]
    return classes, min_dists
