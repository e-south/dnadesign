"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_optimizer_gibbs_behavior_controls.py

Validate Gibbs inertia and adaptation-freeze controls.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dnadesign.cruncher.core.optimizers.gibbs_anneal import GibbsAnnealOptimizer
from dnadesign.cruncher.core.state import SequenceState


class _FlatEvaluator:
    def __call__(self, state: SequenceState) -> dict[str, float]:
        _ = state
        return {"tf": 0.0}

    def evaluate(
        self,
        state: SequenceState,
        beta: float | None = None,
        *,
        length: int | None = None,
    ) -> tuple[dict[str, float], float]:
        _ = (state, beta, length)
        return {"tf": 0.0}, 0.0

    def combined_from_scores(
        self,
        per_tf: dict[str, float],
        *,
        beta: float | None = None,
        length: int | None = None,
    ) -> float:
        _ = (per_tf, beta, length)
        return 0.0


def _base_cfg() -> dict[str, object]:
    return {
        "draws": 3,
        "tune": 0,
        "chains": 1,
        "min_dist": 0,
        "top_k": 0,
        "sequence_length": 4,
        "record_tune": False,
        "build_trace": False,
        "progress_bar": False,
        "progress_every": 0,
        "early_stop": {},
        "mcmc_cooling": {"kind": "fixed", "beta": 1.0},
        "softmin": {"enabled": False},
        "target_worst_tf_prob": 0.0,
        "target_window_pad": 0,
        "block_len_range": (1, 2),
        "multi_k_range": (1, 2),
        "slide_max_shift": 1,
        "swap_len_range": (1, 2),
        "move_probs": {"S": 1.0, "B": 0.0, "M": 0.0, "L": 0.0, "W": 0.0, "I": 0.0},
        "adaptive_weights": {
            "enabled": False,
            "window": 1,
            "k": 0.5,
            "kinds": ["B"],
            "targets": {"B": 0.5},
        },
        "proposal_adapt": {
            "enabled": False,
            "window": 1,
            "step": 0.3,
            "min_scale": 0.5,
            "max_scale": 3.0,
            "target_low": 0.2,
            "target_high": 0.4,
        },
        "gibbs_inertia": {
            "enabled": False,
            "kind": "fixed",
            "p_stay_start": 0.0,
            "p_stay_end": 0.0,
        },
    }


def test_gibbs_inertia_fixed_one_prevents_single_site_flip() -> None:
    cfg = _base_cfg()
    cfg["gibbs_inertia"] = {
        "enabled": True,
        "kind": "fixed",
        "p_stay_start": 1.0,
        "p_stay_end": 1.0,
    }
    rng = np.random.default_rng(7)
    optimizer = GibbsAnnealOptimizer(
        evaluator=_FlatEvaluator(),
        cfg=cfg,
        rng=rng,
        pwms={},
        init_cfg=SimpleNamespace(kind="random", length=4, pad_with="background", regulator=None),
    )
    seq = np.array([0, 0, 0, 0], dtype=np.int8)
    state = SequenceState(seq)
    move_kind, accepted, _per_tf, _score, detail = optimizer._single_chain_move(
        seq,
        0.0,
        20.0,
        None,
        _FlatEvaluator(),
        rng,
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        state=state,
        scan_cache=None,
        per_tf={"tf": 0.0},
        sweep_idx=0,
    )

    assert move_kind == "S"
    assert accepted is True
    assert seq.tolist() == [0, 0, 0, 0]
    assert detail["gibbs_changed"] is False
    assert detail["delta_hamming"] == 0


def test_adaptation_freeze_after_sweep_keeps_move_adaptation_fixed() -> None:
    cfg = _base_cfg()
    cfg["move_probs"] = {"S": 0.0, "B": 1.0, "M": 0.0, "L": 0.0, "W": 0.0, "I": 0.0}
    cfg["adaptive_weights"] = {
        "enabled": True,
        "window": 1,
        "k": 0.8,
        "min_prob": 0.01,
        "max_prob": 0.95,
        "kinds": ["B"],
        "targets": {"B": 0.5},
        "freeze_after_sweep": 0,
    }
    cfg["proposal_adapt"] = {
        "enabled": True,
        "window": 1,
        "step": 0.4,
        "min_scale": 0.5,
        "max_scale": 3.0,
        "target_low": 0.2,
        "target_high": 0.4,
        "freeze_after_sweep": 0,
    }
    rng = np.random.default_rng(11)
    optimizer = GibbsAnnealOptimizer(
        evaluator=_FlatEvaluator(),
        cfg=cfg,
        rng=rng,
        pwms={},
        init_cfg=SimpleNamespace(kind="random", length=4, pad_with="background", regulator=None),
    )

    optimizer.optimise()

    assert optimizer.move_controller.log_weights["B"] == pytest.approx(0.0)
    assert optimizer.proposal_controller.block_scale == pytest.approx(1.0)
    assert optimizer.proposal_controller.multi_scale == pytest.approx(1.0)
