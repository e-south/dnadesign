"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_gibbs_swap_stats_contract.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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


def test_optimizer_stats_exclude_replica_exchange_fields() -> None:
    chains = 3
    draws = 3
    cfg = {
        "draws": draws,
        "tune": 0,
        "chains": chains,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 4,
        "bidirectional": False,
        "record_tune": False,
        "build_trace": False,
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
    assert "swap_attempts_by_pair" not in stats
    assert "swap_accepts_by_pair" not in stats
    assert "swap_events" not in stats
    assert "swap_acceptance_rate" not in stats


def test_optimizer_stats_linear_cooling_uses_explicit_bounds() -> None:
    chains = 2
    draws = 3
    cfg = {
        "draws": draws,
        "tune": 0,
        "chains": chains,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 4,
        "bidirectional": False,
        "record_tune": False,
        "build_trace": False,
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
    assert stats["mcmc_cooling"] == {"kind": "linear", "beta_start": 0.2, "beta_end": 1.0}
    assert optimizer.objective_schedule_summary()["mcmc_cooling"] == {
        "kind": "linear",
        "beta_start": 0.2,
        "beta_end": 1.0,
    }


def test_optimizer_stats_reuses_move_stats_buffer() -> None:
    cfg = {
        "draws": 2,
        "tune": 0,
        "chains": 2,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 4,
        "bidirectional": False,
        "record_tune": False,
        "build_trace": False,
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
    assert stats["move_stats"] is optimizer.move_stats
