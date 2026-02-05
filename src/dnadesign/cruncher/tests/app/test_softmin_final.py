"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_softmin_final.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dnadesign.cruncher.app.sample_workflow import _resolve_final_softmin_beta
from dnadesign.cruncher.config.schema_v2 import (
    InitConfig,
    SampleComputeConfig,
    SampleConfig,
    SampleObjectiveConfig,
    SoftminConfig,
)
from dnadesign.cruncher.core.optimizers.pt import PTGibbsOptimizer


class _DummyEvaluator:
    def __call__(self, state):  # pragma: no cover - not used in this test
        return {"tf": 0.0}


def _base_cfg() -> dict:
    return {
        "draws": 2,
        "tune": 2,
        "chains": 1,
        "min_dist": 0,
        "top_k": 1,
        "swap_prob": 0.0,
        "record_tune": False,
        "progress_bar": False,
        "progress_every": 0,
        "early_stop": {},
        "block_len_range": (1, 1),
        "multi_k_range": (1, 1),
        "slide_max_shift": 1,
        "swap_len_range": (1, 1),
        "move_probs": {"S": 1.0, "B": 0.0, "M": 0.0, "L": 0.0, "W": 0.0, "I": 0.0},
        "kind": "fixed",
        "beta": 1.0,
        "softmin": {"enabled": False},
    }


def test_pt_final_softmin_beta_respects_schedule() -> None:
    cfg = _base_cfg()
    cfg["softmin"] = {
        "enabled": True,
        "kind": "piecewise",
        "stages": [
            {"sweeps": 0, "beta": 1.0},
            {"sweeps": 3, "beta": 5.0},
        ],
    }

    init_cfg = SimpleNamespace(kind="random", length=4, pad_with="background", regulator=None)
    optimizer = PTGibbsOptimizer(
        evaluator=_DummyEvaluator(),
        cfg=cfg,
        rng=np.random.default_rng(0),
        init_cfg=init_cfg,
        pwms={},
    )
    assert optimizer.final_softmin_beta() == 5.0


def test_resolve_final_softmin_beta_uses_optimizer_value() -> None:
    objective = SampleObjectiveConfig(
        softmin=SoftminConfig(
            enabled=True,
            kind="linear",
            beta=(0.5, 2.0),
        )
    )
    sample_cfg = SampleConfig(
        sequence_length=4,
        compute=SampleComputeConfig(total_sweeps=2, adapt_sweep_frac=0.5),
        init=InitConfig(kind="random"),
        objective=objective,
    )

    class _Opt:
        def final_softmin_beta(self) -> float:
            return 7.5

    assert _resolve_final_softmin_beta(_Opt(), sample_cfg) == 7.5
