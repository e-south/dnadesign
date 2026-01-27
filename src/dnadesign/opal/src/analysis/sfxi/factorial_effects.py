"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/sfxi/factorial_effects.py

Factorial effects (A, B, interaction) for logic vectors.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .state_order import require_state_order


def compute_factorial_effects(
    v: np.ndarray,
    *,
    state_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_state_order(state_order)
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 4:
        raise ValueError("factorial_effects expects vectors of shape (n, 4).")
    v00, v10, v01, v11 = (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
    a_effect = ((v10 + v11) - (v00 + v01)) / 2.0
    b_effect = ((v01 + v11) - (v00 + v10)) / 2.0
    ab_interaction = ((v11 + v00) - (v10 + v01)) / 2.0
    return a_effect, b_effect, ab_interaction
