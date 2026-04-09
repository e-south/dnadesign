"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_sample_runtime_capabilities.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.app.sample.run_set_execution import _build_optimizer_cfg
from dnadesign.cruncher.config.schema_v3 import SampleConfig
from dnadesign.cruncher.core.objectives.compiler import compile_objective_plan
from dnadesign.cruncher.core.pwm import PWM


def _pwm(name: str, width: int) -> PWM:
    row = [0.97, 0.01, 0.01, 0.01]
    return PWM(name=name, matrix=[row[:] for _ in range(width)])


def test_multiplicity_plan_disables_incremental_rescore_and_targeting() -> None:
    sample_cfg = SampleConfig.model_validate(
        {
            "seed": 7,
            "sequence_length": 6,
            "budget": {"tune": 0, "draws": 2},
            "objective": {
                "score_scale": "normalized-llr",
                "combine": "min",
                "multiplicity": {
                    "enabled": True,
                    "copies": 2,
                    "distinctness": {"mode": "interval", "min_gap": 0, "strand_rule": "collapse_same_locus"},
                    "aggregation": {"selector": "top_k_distinct", "scalar": "weakest_selected"},
                },
            },
            "moves": {
                "profile": "balanced",
                "overrides": {"target_worst_tf_prob": 0.8, "target_window_pad": 3},
            },
            "optimizer": {"kind": "gibbs_anneal"},
            "elites": {"k": 1, "select": {"diversity": 0.0}, "postprocess": {"trim_uncovered_internal": False}},
        }
    )
    objective_plan = compile_objective_plan(sample_cfg=sample_cfg, tfs=["lexA"], pwms={"lexA": _pwm("lexA", 2)})

    cfg = _build_optimizer_cfg(
        sample_cfg=sample_cfg,
        objective_plan=objective_plan,
        chain_count=1,
        draws=2,
        adapt_sweeps=0,
        progress_bar=False,
        progress_every=0,
    )

    assert cfg["enable_incremental_rescore"] is False
    assert cfg["target_worst_tf_prob"] == 0.0
