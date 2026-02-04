"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_softmin_final.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dnadesign.cruncher.app.sample_workflow import _resolve_final_softmin_beta
from dnadesign.cruncher.config.schema_v2 import (
    InitConfig,
    SampleBudgetConfig,
    SampleConfig,
    SampleObjectiveConfig,
    SoftminConfig,
)
from dnadesign.cruncher.core.optimizers.gibbs import GibbsOptimizer


class _DummyEvaluator:
    def __call__(self, state):  # pragma: no cover - not used in this test
        return {"tf": 0.0}


def _base_cfg() -> dict:
    return {
        "draws": 1,
        "tune": 1,
        "chains": 2,
        "min_dist": 0,
        "top_k": 1,
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
    }


def test_gibbs_final_softmin_beta_respects_schedule_scope() -> None:
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
    per_chain = GibbsOptimizer(
        evaluator=_DummyEvaluator(),
        cfg={**cfg, "schedule_scope": "per_chain", "apply_during": "all"},
        rng=np.random.default_rng(0),
        init_cfg=init_cfg,
        pwms={},
    )
    global_scope = GibbsOptimizer(
        evaluator=_DummyEvaluator(),
        cfg={**cfg, "schedule_scope": "global", "apply_during": "all"},
        rng=np.random.default_rng(0),
        init_cfg=init_cfg,
        pwms={},
    )
    assert global_scope.final_softmin_beta() > per_chain.final_softmin_beta()


def test_resolve_final_softmin_beta_uses_optimizer_value() -> None:
    objective = SampleObjectiveConfig(
        softmin=SoftminConfig(
            enabled=True,
            kind="linear",
            beta=(0.5, 2.0),
        )
    )
    sample_cfg = SampleConfig(
        budget=SampleBudgetConfig(tune=1, draws=1, restarts=1),
        init=InitConfig(kind="random", length=4),
        objective=objective,
    )

    class _Opt:
        def final_softmin_beta(self) -> float:
            return 7.5

    assert _resolve_final_softmin_beta(_Opt(), sample_cfg) == 7.5
