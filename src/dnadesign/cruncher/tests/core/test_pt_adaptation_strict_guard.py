"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_pt_adaptation_strict_guard.py

Ensures gibbs annealing does not run replica-exchange adaptation logic.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dnadesign.cruncher.core.optimizers.gibbs_anneal import GibbsAnnealOptimizer
from dnadesign.cruncher.core.state import SequenceState


class _DummyEvaluator:
    def __call__(self, state: SequenceState) -> dict[str, float]:
        return {"tf": float(state.seq.sum())}

    def combined_from_scores(self, per_tf: dict[str, float], beta: float | None = None, length: int | None = None):
        return float(next(iter(per_tf.values())))

    def evaluate(self, state: SequenceState, beta: float | None = None, *, length: int | None = None):
        per_tf = self(state)
        return per_tf, float(next(iter(per_tf.values())))


def test_adaptive_swap_config_is_rejected_for_gibbs_anneal() -> None:
    cfg = {
        "draws": 3,
        "tune": 6,
        "chains": 3,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 4,
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
        "mcmc_cooling": {"kind": "fixed", "beta": 1.0},
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
    with pytest.raises(ValueError, match="unsupported for optimizer kind='gibbs_anneal'"):
        GibbsAnnealOptimizer(
            evaluator=_DummyEvaluator(),
            cfg=cfg,
            rng=np.random.default_rng(0),
            pwms={},
            init_cfg=init_cfg,
        )


def test_swap_stats_stay_zero_after_draws() -> None:
    cfg = {
        "draws": 8,
        "tune": 0,
        "chains": 3,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 4,
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
        "mcmc_cooling": {"kind": "linear", "beta_start": 0.2, "beta_end": 1.0},
        "softmin": {"enabled": False},
        "target_worst_tf_prob": 0.0,
        "target_window_pad": 0,
    }
    init_cfg = SimpleNamespace(kind="random", length=4, pad_with="background", regulator=None)
    optimizer = GibbsAnnealOptimizer(
        evaluator=_DummyEvaluator(),
        cfg=cfg,
        rng=np.random.default_rng(0),
        pwms={},
        init_cfg=init_cfg,
    )
    optimizer.optimise()

    stats = optimizer.stats()
    assert stats["swap_attempts"] == 0
    assert stats["swap_accepts"] == 0
    assert stats["swap_accepts_by_pair"] == [0, 0]
    assert stats["swap_attempts_by_pair"] == [0, 0]
