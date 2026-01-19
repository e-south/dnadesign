"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_insertion_bidirectional.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dnadesign.cruncher.core.optimizers.gibbs import GibbsOptimizer
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.state import SequenceState


class _FixedRNG:
    def random(self) -> float:
        return 0.1

    def integers(self, low, high=None, size=None):
        if size is None:
            return 0
        return np.zeros(size, dtype=int)

    def choice(self, a, p=None, size=None):
        if isinstance(a, (list, np.ndarray)):
            return a[0] if size is None else np.array([a[0]] * size)
        if size is None:
            return 5  # index of "I" in MOVE_KINDS
        return np.zeros(size, dtype=int)


def test_insertion_can_reverse_complement() -> None:
    pwm = PWM(name="tfA", matrix=np.array([[0.9, 0.05, 0.03, 0.02], [0.05, 0.9, 0.03, 0.02]]))
    pwms = {"tfA": pwm}
    evaluator = SimpleNamespace(
        combined=lambda state, beta=None: float(state.seq.sum()),
        evaluate=lambda state, beta=None, length=None: ({"tfA": float(state.seq.sum())}, float(state.seq.sum())),
        pwm_width=lambda tf: pwm.length,
    )
    cfg = {
        "draws": 1,
        "tune": 1,
        "chains": 1,
        "min_dist": 0,
        "top_k": 1,
        "bidirectional": True,
        "record_tune": False,
        "progress_bar": False,
        "progress_every": 0,
        "early_stop": {},
        "block_len_range": (1, 1),
        "multi_k_range": (1, 1),
        "slide_max_shift": 1,
        "swap_len_range": (1, 1),
        "move_probs": {"S": 0.0, "B": 0.0, "M": 0.0, "L": 0.0, "W": 0.0, "I": 1.0},
        "kind": "fixed",
        "beta": 1.0,
        "softmin": {"enabled": False},
        "insertion_consensus_prob": 1.0,
    }
    init_cfg = SimpleNamespace(kind="random", length=2, pad_with="background", regulator=None)
    optimizer = GibbsOptimizer(
        evaluator=evaluator,
        cfg=cfg,
        rng=np.random.default_rng(0),
        init_cfg=init_cfg,
        pwms=pwms,
    )
    seq = np.array([0, 0], dtype=np.int8)
    state = SequenceState(seq)
    current = evaluator.combined(state, beta=None)
    rng = _FixedRNG()
    optimizer._perform_single_move(
        seq,
        current,
        1.0,
        None,
        evaluator,
        cfg,
        rng,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        state=state,
        scan_cache=None,
        per_tf={"tfA": float(seq.sum())},
    )
    assert seq.tolist() == [2, 3]
