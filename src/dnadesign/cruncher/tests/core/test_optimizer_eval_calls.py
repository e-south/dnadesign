"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_optimizer_eval_calls.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dnadesign.cruncher.core.optimizers.gibbs_anneal import GibbsAnnealOptimizer
from dnadesign.cruncher.core.optimizers.policies import TargetingPolicy
from dnadesign.cruncher.core.state import SequenceState


class _CountingEvaluator:
    def __init__(self) -> None:
        self.combined_calls = 0
        self.evaluate_calls = 0
        self.call_count = 0

    def __call__(self, state: SequenceState) -> dict[str, float]:
        self.call_count += 1
        return {"tf": float(state.seq.sum())}

    def combined(self, state: SequenceState, beta: float | None = None) -> float:
        self.combined_calls += 1
        return float(state.seq.sum())

    def evaluate(self, state: SequenceState, beta: float | None = None, *, length: int | None = None):
        self.evaluate_calls += 1
        return {"tf": float(state.seq.sum())}, float(state.seq.sum())


def test_gibbs_block_move_calls_evaluate_once_for_proposal() -> None:
    rng = np.random.default_rng(0)
    evaluator = _CountingEvaluator()
    cfg = {
        "draws": 1,
        "tune": 1,
        "chains": 1,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 4,
        "swap_prob": 0.0,
        "record_tune": False,
        "progress_bar": False,
        "progress_every": 0,
        "early_stop": {},
        "block_len_range": (1, 1),
        "multi_k_range": (1, 1),
        "slide_max_shift": 1,
        "swap_len_range": (1, 1),
        "move_probs": {"S": 0.0, "B": 1.0, "M": 0.0, "L": 0.0, "W": 0.0, "I": 0.0},
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
    seq = np.array([0, 1, 2, 3], dtype=np.int8)
    state = SequenceState(seq)
    current = evaluator.combined(state, beta=None)
    before = evaluator.evaluate_calls
    optimizer._single_chain_move(
        seq,
        current,
        1.0,
        None,
        evaluator,
        rng,
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        state=state,
        scan_cache=None,
    )
    after = evaluator.evaluate_calls
    assert after - before == 1


def test_targeting_policy_uses_precomputed_per_tf() -> None:
    class _TargetEvaluator:
        def __init__(self) -> None:
            self.call_count = 0
            self.best_hit_calls = 0
            self.best_hits_calls = 0

        def __call__(self, state: SequenceState) -> dict[str, float]:
            self.call_count += 1
            return {"tf": 0.1}

        def best_hit(self, state: SequenceState, tf: str):
            self.best_hit_calls += 1
            return 1.0, 0, "+"

        def best_hits(self, state: SequenceState):
            self.best_hits_calls += 1
            return {"tf": (1.0, 0, "+")}

        def pwm_width(self, tf: str) -> int:
            return 1

    evaluator = _TargetEvaluator()
    policy = TargetingPolicy(enabled=True, worst_tf_prob=1.0, window_pad=0)
    rng = np.random.default_rng(0)
    state = SequenceState(np.array([0, 1, 2], dtype=np.int8))
    policy.maybe_target(seq_len=3, state=state, evaluator=evaluator, rng=rng, per_tf={"tf": 0.1})
    assert evaluator.call_count == 0
    assert evaluator.best_hits_calls == 0
    assert evaluator.best_hit_calls == 1


def test_gibbs_block_move_uses_precomputed_scores() -> None:
    rng = np.random.default_rng(0)
    evaluator = _CountingEvaluator()
    cfg = {
        "draws": 1,
        "tune": 1,
        "chains": 1,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 4,
        "swap_prob": 0.0,
        "record_tune": False,
        "progress_bar": False,
        "progress_every": 0,
        "early_stop": {},
        "block_len_range": (1, 1),
        "multi_k_range": (1, 1),
        "slide_max_shift": 1,
        "swap_len_range": (1, 1),
        "move_probs": {"S": 0.0, "B": 1.0, "M": 0.0, "L": 0.0, "W": 0.0, "I": 0.0},
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
    seq = np.array([0, 1, 2, 3], dtype=np.int8)
    state = SequenceState(seq)
    current = evaluator.combined(state, beta=None)
    before = evaluator.call_count
    optimizer._single_chain_move(
        seq,
        current,
        1.0,
        None,
        evaluator,
        rng,
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        state=state,
        scan_cache=None,
        per_tf={"tf": 1.0},
    )
    after = evaluator.call_count
    assert after == before
