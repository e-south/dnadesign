"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_early_stop_unique_successes.py

Validates diversity-aware early-stop gating.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dnadesign.cruncher.core.optimizers.pt import GibbsAnnealOptimizer
from dnadesign.cruncher.core.state import SequenceState


class _ConstantEvaluator:
    def __call__(self, state: SequenceState) -> dict[str, float]:
        _ = state
        return {"tf": 0.0}

    def combined(self, state: SequenceState, beta: float | None = None) -> float:
        _ = state
        _ = beta
        return 0.0

    def combined_from_scores(
        self, per_tf: dict[str, float], beta: float | None = None, *, length: int | None = None
    ) -> float:
        _ = per_tf
        _ = beta
        _ = length
        return 0.0

    def evaluate(self, state: SequenceState, beta: float | None = None, *, length: int | None = None):
        _ = state
        _ = beta
        _ = length
        return {"tf": 0.0}, 0.0


def test_pt_early_stop_waits_for_unique_successes() -> None:
    rng = np.random.default_rng(0)
    evaluator = _ConstantEvaluator()
    cfg = {
        "draws": 5,
        "tune": 0,
        "chains": 1,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 4,
        "swap_prob": 0.0,
        "bidirectional": False,
        "dsdna_canonicalize": False,
        "score_scale": "normalized-llr",
        "record_tune": False,
        "progress_bar": False,
        "progress_every": 0,
        "early_stop": {
            "enabled": True,
            "patience": 1,
            "min_delta": 0.0,
            "require_min_unique": True,
            "min_unique": 1,
            "success_min_per_tf_norm": 0.9,
        },
        "block_len_range": (1, 1),
        "multi_k_range": (1, 1),
        "slide_max_shift": 1,
        "swap_len_range": (1, 1),
        "move_probs": {"S": 1.0, "B": 0.0, "M": 0.0, "L": 0.0, "W": 0.0, "I": 0.0},
        "kind": "fixed",
        "beta": 1.0,
        "softmin": {"enabled": False},
    }
    init_cfg = SimpleNamespace(kind="random", length=4, pad_with="background", regulator=None)
    optimizer = GibbsAnnealOptimizer(
        evaluator=evaluator,
        cfg=cfg,
        rng=rng,
        pwms={},
        init_cfg=init_cfg,
    )

    optimizer.optimise()

    assert len(optimizer.all_samples) == cfg["draws"]
    stats = optimizer.stats()
    assert stats.get("unique_successes") == 0
