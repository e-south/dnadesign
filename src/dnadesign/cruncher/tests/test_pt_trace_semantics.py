"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_pt_trace_semantics.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.cruncher.core.optimizers.pt import PTGibbsOptimizer
from dnadesign.cruncher.core.state import SequenceState


class _SwapEvaluator:
    def __call__(self, state: SequenceState) -> dict[str, float]:
        return {"tf": 2.0 if int(state.seq[0]) == 0 else 0.0}

    def combined(self, state: SequenceState, beta: float | None = None) -> float:
        return 2.0 if int(state.seq[0]) == 0 else 0.0

    def combined_from_scores(
        self, per_tf_scores: dict[str, float], beta: float | None = None, *, length: int | None = None
    ) -> float:
        return float(per_tf_scores["tf"])


def test_pt_records_post_swap(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    evaluator = _SwapEvaluator()
    cfg = {
        "draws": 1,
        "tune": 0,
        "chains": 2,
        "min_dist": 0,
        "top_k": 1,
        "swap_prob": 1.0,
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
        "beta": [1.0, 2.0],
        "softmin": {"enabled": False},
        "adaptive_swap": {"enabled": False},
        "target_worst_tf_prob": 0.0,
        "target_window_pad": 0,
        "init_seeds": [np.array([0], dtype=np.int8), np.array([3], dtype=np.int8)],
    }

    opt = PTGibbsOptimizer(
        evaluator=evaluator,
        cfg=cfg,
        rng=rng,
        pwms={},
        init_cfg=type("Init", (), {"kind": "random", "length": 1, "pad_with": "background", "regulator": None})(),
    )

    def _noop_move(self, seq, current_combined, beta, beta_softmin, evaluator, rng, move_probs, *, per_tf=None):
        if per_tf is None:
            per_tf = evaluator(SequenceState(seq))
        return "S", True, per_tf, current_combined

    monkeypatch.setattr(opt, "_single_chain_move", _noop_move.__get__(opt, PTGibbsOptimizer))
    opt.optimise()

    recorded = {chain: seq for (chain, _), seq in zip(opt.all_meta, opt.all_samples)}
    assert recorded[0][0] == 3  # chain 0 should contain swapped-in state
