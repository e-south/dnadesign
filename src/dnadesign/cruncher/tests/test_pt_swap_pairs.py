"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_pt_swap_pairs.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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


def test_pt_swap_pairs_tracked() -> None:
    cfg = {
        "draws": 1,
        "tune": 0,
        "chains": 3,
        "min_dist": 0,
        "top_k": 1,
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
        "beta": [1.0, 2.0, 3.0],
        "softmin": {"enabled": False},
        "target_worst_tf_prob": 0.0,
        "target_window_pad": 0,
    }
    init_cfg = SimpleNamespace(kind="random", length=4, pad_with="background", regulator=None)
    optimizer = PTGibbsOptimizer(
        evaluator=_DummyEvaluator(),
        cfg=cfg,
        rng=np.random.default_rng(0),
        pwms={},
        init_cfg=init_cfg,
    )
    optimizer.optimise()
    stats = optimizer.stats()
    assert stats["swap_attempts_by_pair"] == [1, 1]
    assert len(stats["swap_attempts_by_pair"]) == optimizer.chains - 1
