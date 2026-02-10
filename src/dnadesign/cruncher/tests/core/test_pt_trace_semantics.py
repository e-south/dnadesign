"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_pt_trace_semantics.py

Validates gibbs annealing trace metadata semantics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.cruncher.core.optimizers.gibbs_anneal import GibbsAnnealOptimizer
from dnadesign.cruncher.core.state import SequenceState


class _FlatEvaluator:
    def __call__(self, state: SequenceState) -> dict[str, float]:
        return {"tf": float(state.seq.sum())}

    def combined_from_scores(
        self, per_tf_scores: dict[str, float], beta: float | None = None, *, length: int | None = None
    ) -> float:
        return float(per_tf_scores["tf"])

    def evaluate(self, state: SequenceState, beta: float | None = None, *, length: int | None = None):
        per_tf = self(state)
        return per_tf, float(per_tf["tf"])


def test_trace_meta_includes_chain_and_sweep_fields() -> None:
    cfg = {
        "draws": 3,
        "tune": 0,
        "chains": 2,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 1,
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
    }

    opt = GibbsAnnealOptimizer(
        evaluator=_FlatEvaluator(),
        cfg=cfg,
        rng=np.random.default_rng(0),
        pwms={},
        init_cfg=type("Init", (), {"kind": "random", "length": 1, "pad_with": "background", "regulator": None})(),
    )
    opt.optimise()

    trace_rows = opt.all_trace_meta
    assert len(trace_rows) == cfg["draws"] * cfg["chains"]
    required = {"chain", "slot_id", "particle_id", "beta", "sweep_idx", "phase"}
    for row in trace_rows:
        assert required.issubset(set(row.keys()))
    assert sorted({int(row["chain"]) for row in trace_rows}) == [0, 1]
    assert sorted({int(row["particle_id"]) for row in trace_rows}) == [0, 1]
    assert opt.swap_events == []


def test_can_skip_trace_construction_when_disabled() -> None:
    cfg = {
        "draws": 2,
        "tune": 0,
        "chains": 2,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 1,
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
        "build_trace": False,
    }

    opt = GibbsAnnealOptimizer(
        evaluator=_FlatEvaluator(),
        cfg=cfg,
        rng=np.random.default_rng(0),
        pwms={},
        init_cfg=type("Init", (), {"kind": "random", "length": 1, "pad_with": "background", "regulator": None})(),
    )
    opt.optimise()

    assert opt.trace_idata is None
