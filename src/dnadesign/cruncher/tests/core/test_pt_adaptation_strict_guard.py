"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_pt_adaptation_strict_guard.py

Ensures strict swap-adaptation guard fails PT runs when ladder adaptation is
persistently saturated.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dnadesign.cruncher.core.optimizers.pt import PTGibbsOptimizer
from dnadesign.cruncher.core.state import SequenceState


class _DummyEvaluator:
    def __call__(self, state: SequenceState) -> dict[str, float]:
        return {"tf": float(state.seq.sum())}

    def combined_from_scores(self, per_tf: dict[str, float], beta: float | None = None, length: int | None = None):
        return float(next(iter(per_tf.values())))

    def evaluate(self, state: SequenceState, beta: float | None = None, *, length: int | None = None):
        per_tf = self(state)
        return per_tf, float(next(iter(per_tf.values())))


def test_pt_strict_swap_adaptation_guard_raises() -> None:
    cfg = {
        "draws": 2,
        "tune": 6,
        "chains": 3,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 4,
        "swap_prob": 1.0,
        "bidirectional": False,
        "record_tune": False,
        "progress_bar": False,
        "progress_every": 0,
        "early_stop": {},
        "block_len_range": (1, 1),
        "multi_k_range": (1, 1),
        "slide_max_shift": 1,
        "swap_len_range": (1, 1),
        "move_probs": {"S": 1.0, "B": 0.0, "M": 0.0, "L": 0.0, "W": 0.0, "I": 0.0},
        "kind": "geometric",
        "beta": [0.2, 0.5, 1.0],
        "softmin": {"enabled": False},
        "target_worst_tf_prob": 0.0,
        "target_window_pad": 0,
        "adaptive_swap": {
            "enabled": True,
            "target_swap": 0.25,
            "window": 1,
            "k": 2.0,
            "min_scale": 0.25,
            "max_scale": 1.0,
            "stop_after_tune": True,
            "strict": True,
            "saturation_windows": 2,
        },
    }
    init_cfg = SimpleNamespace(kind="random", length=4, pad_with="background", regulator=None)
    optimizer = PTGibbsOptimizer(
        evaluator=_DummyEvaluator(),
        cfg=cfg,
        rng=np.random.default_rng(0),
        pwms={},
        init_cfg=init_cfg,
    )
    with pytest.raises(RuntimeError, match="tuning-limited"):
        optimizer.optimise()
