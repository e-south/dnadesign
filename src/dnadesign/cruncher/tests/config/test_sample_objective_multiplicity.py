"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_sample_objective_multiplicity.py

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.cruncher.app.sample.preflight import ConfigError, prepare_objective_plan
from dnadesign.cruncher.config.schema_v3 import SampleConfig
from dnadesign.cruncher.core.pwm import PWM


def _sample_config(
    *,
    scale: str = "normalized-llr",
    copies: int = 2,
    diversity: float = 0.0,
    trim_uncovered_internal: bool = False,
    min_gap: int = 0,
    sequence_length: int = 8,
    distinctness_mode: str = "interval",
) -> SampleConfig:
    return SampleConfig.model_validate(
        {
            "seed": 7,
            "sequence_length": sequence_length,
            "budget": {"tune": 0, "draws": 1},
            "objective": {
                "score_scale": scale,
                "combine": "min",
                "multiplicity": {
                    "enabled": True,
                    "copies": copies,
                    "distinctness": {
                        "mode": distinctness_mode,
                        "min_gap": min_gap,
                        "strand_rule": "collapse_same_locus",
                    },
                    "aggregation": {
                        "selector": "top_k_distinct",
                        "scalar": "weakest_selected",
                    },
                },
            },
            "optimizer": {"kind": "gibbs_anneal"},
            "elites": {
                "k": 1,
                "select": {"diversity": diversity},
                "postprocess": {"trim_uncovered_internal": trim_uncovered_internal},
            },
        }
    )


def _pwm(name: str, width: int) -> PWM:
    row = [0.97, 0.01, 0.01, 0.01]
    return PWM(name=name, matrix=[row[:] for _ in range(width)])


def test_prepare_objective_plan_rejects_multiplicity_for_multi_tf_runs() -> None:
    pwms = {"lexA": _pwm("lexA", 3), "cpxR": _pwm("cpxR", 3)}

    with pytest.raises(ConfigError, match="exactly one TF"):
        prepare_objective_plan(
            sample_cfg=_sample_config(),
            tfs=["lexA", "cpxR"],
            pwms=pwms,
        )


def test_prepare_objective_plan_rejects_non_occurrence_safe_score_scale() -> None:
    pwms = {"lexA": _pwm("lexA", 3)}

    with pytest.raises(ConfigError, match="occurrence-safe"):
        prepare_objective_plan(
            sample_cfg=_sample_config(scale="logp"),
            tfs=["lexA"],
            pwms=pwms,
        )


def test_prepare_objective_plan_rejects_infeasible_copy_count_for_sequence_length() -> None:
    pwms = {"lexA": _pwm("lexA", 5)}

    with pytest.raises(ConfigError, match="infeasible"):
        prepare_objective_plan(
            sample_cfg=_sample_config(copies=2, sequence_length=8, min_gap=0),
            tfs=["lexA"],
            pwms=pwms,
        )


def test_prepare_objective_plan_allows_overlap_tolerant_offset_distinctness() -> None:
    pwms = {"lexA": _pwm("lexA", 5)}

    compiled = prepare_objective_plan(
        sample_cfg=_sample_config(copies=4, sequence_length=8, min_gap=0, distinctness_mode="offset"),
        tfs=["lexA"],
        pwms=pwms,
    )

    assert compiled.objectives[0].selector.distinctness.mode == "offset"


def test_prepare_objective_plan_rejects_multiplicity_with_diversity_mmr() -> None:
    pwms = {"lexA": _pwm("lexA", 3)}

    with pytest.raises(ConfigError, match="diversity"):
        prepare_objective_plan(
            sample_cfg=_sample_config(diversity=0.25),
            tfs=["lexA"],
            pwms=pwms,
        )


def test_prepare_objective_plan_rejects_hit_anchored_postprocess_for_multiplicity() -> None:
    pwms = {"lexA": _pwm("lexA", 3)}

    with pytest.raises(ConfigError, match="postprocess"):
        prepare_objective_plan(
            sample_cfg=_sample_config(trim_uncovered_internal=True),
            tfs=["lexA"],
            pwms=pwms,
        )
