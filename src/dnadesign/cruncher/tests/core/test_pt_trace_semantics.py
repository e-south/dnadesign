"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_pt_trace_semantics.py

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
    chains = 2
    draws = 2
    cfg = {
        "draws": draws,
        "tune": 0,
        "chains": chains,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 1,
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
    }
    seed_rng = np.random.default_rng(0)
    base_seed = seed_rng.integers(0, 4, size=1, dtype=np.int8)
    positions = seed_rng.choice(base_seed.size, size=1, replace=False)
    current = int(base_seed[positions[0]])
    pick = int(seed_rng.integers(0, 3))
    if pick >= current:
        pick += 1
    mutated = base_seed.copy()
    mutated[positions[0]] = np.int8(pick)

    opt = PTGibbsOptimizer(
        evaluator=evaluator,
        cfg=cfg,
        rng=rng,
        pwms={},
        init_cfg=type("Init", (), {"kind": "random", "length": 1, "pad_with": "background", "regulator": None})(),
    )

    def _noop_move(
        self,
        seq,
        current_combined,
        beta,
        beta_softmin,
        evaluator,
        rng,
        move_probs,
        *,
        state=None,
        scan_cache=None,
        per_tf=None,
    ):
        if per_tf is None:
            per_tf = evaluator(SequenceState(seq))
        return (
            "S",
            True,
            per_tf,
            current_combined,
            {
                "delta": 0.0,
                "score_old": current_combined,
                "score_new": current_combined,
            },
        )

    monkeypatch.setattr(opt, "_single_chain_move", _noop_move.__get__(opt, PTGibbsOptimizer))
    opt.optimise()

    first_draw = {chain: seq for (chain, draw_idx), seq in zip(opt.all_meta, opt.all_samples) if draw_idx == 0}
    assert int(first_draw[0][0]) == int(mutated[0])  # chain 0 should contain swapped-in state after first swap


def test_pt_can_skip_trace_construction_when_disabled(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    evaluator = _SwapEvaluator()
    chains = 2
    draws = 2
    cfg = {
        "draws": draws,
        "tune": 0,
        "chains": chains,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 1,
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
        "build_trace": False,
    }

    opt = PTGibbsOptimizer(
        evaluator=evaluator,
        cfg=cfg,
        rng=rng,
        pwms={},
        init_cfg=type("Init", (), {"kind": "random", "length": 1, "pad_with": "background", "regulator": None})(),
    )

    def _noop_move(
        self,
        seq,
        current_combined,
        beta,
        beta_softmin,
        evaluator,
        rng,
        move_probs,
        *,
        state=None,
        scan_cache=None,
        per_tf=None,
    ):
        if per_tf is None:
            per_tf = evaluator(SequenceState(seq))
        return (
            "S",
            True,
            per_tf,
            current_combined,
            {
                "delta": 0.0,
                "score_old": current_combined,
                "score_new": current_combined,
            },
        )

    monkeypatch.setattr(opt, "_single_chain_move", _noop_move.__get__(opt, PTGibbsOptimizer))
    opt.optimise()
    assert opt.trace_idata is None
