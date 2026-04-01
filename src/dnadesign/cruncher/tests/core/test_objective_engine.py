"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_objective_engine.py

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.cruncher.config.schema_v3 import SampleConfig
from dnadesign.cruncher.core.objectives.capabilities import (
    BEST_HIT_CAPABILITIES,
    KDISTINCT_V1_CAPABILITIES,
)
from dnadesign.cruncher.core.objectives.compiler import compile_objective_plan
from dnadesign.cruncher.core.objectives.engine import ObjectiveEngine
from dnadesign.cruncher.core.objectives.models import (
    DistinctnessSpec,
    HitAggregationSpec,
    ObjectiveCapabilities,
    ObjectiveSpec,
    TopKDistinctSelectorSpec,
    WindowHit,
)
from dnadesign.cruncher.core.objectives.selectors import TopKDistinctSelector
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer


def _sample_config(
    *,
    scale: str = "llr",
    copies: int = 2,
    min_gap: int = 0,
    distinctness_mode: str = "interval",
) -> SampleConfig:
    return SampleConfig.model_validate(
        {
            "seed": 7,
            "sequence_length": 8,
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
            "elites": {"k": 1, "select": {"diversity": 0.0}, "postprocess": {"trim_uncovered_internal": False}},
        }
    )


def _motif_pwm(name: str, motif: str) -> PWM:
    matrix = np.full((len(motif), 4), 0.01, dtype=float)
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    for row_idx, base in enumerate(motif):
        matrix[row_idx, base_to_idx[base]] = 0.97
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return PWM(name=name, matrix=matrix)


def test_compile_objective_plan_preserves_best_hit_defaults() -> None:
    sample_cfg = SampleConfig.model_validate(
        {
            "seed": 7,
            "sequence_length": 10,
            "budget": {"tune": 0, "draws": 1},
            "objective": {"score_scale": "normalized-llr", "combine": "min"},
            "optimizer": {"kind": "gibbs_anneal"},
        }
    )
    pwms = {"lexA": _motif_pwm("lexA", "AA"), "cpxR": _motif_pwm("cpxR", "CC")}

    compiled = compile_objective_plan(sample_cfg=sample_cfg, tfs=["lexA", "cpxR"], pwms=pwms)

    assert [spec.objective_id for spec in compiled.objectives] == ["lexA", "cpxR"]
    assert all(spec.capabilities == BEST_HIT_CAPABILITIES for spec in compiled.objectives)
    assert compiled.runtime.supports_incremental_rescore is True
    assert compiled.runtime.supports_targeted_window_hint is True
    assert compiled.runtime.supports_representative_hit_artifact is True


def test_compile_objective_plan_builds_single_tf_multiplicity_objective() -> None:
    pwms = {"lexA": _motif_pwm("lexA", "AA")}

    compiled = compile_objective_plan(
        sample_cfg=_sample_config(scale="normalized-llr", copies=3),
        tfs=["lexA"],
        pwms=pwms,
    )

    assert len(compiled.objectives) == 1
    objective = compiled.objectives[0]
    assert objective.objective_id == "lexA"
    assert objective.tf == "lexA"
    assert objective.capabilities == KDISTINCT_V1_CAPABILITIES
    assert compiled.runtime.supports_incremental_rescore is False
    assert compiled.runtime.supports_targeted_window_hint is False
    assert compiled.runtime.supports_representative_hit_artifact is False


def test_top_k_distinct_selector_uses_exact_dp_instead_of_greedy_masking() -> None:
    selector = TopKDistinctSelector()
    hits = (
        WindowHit(tf="lexA", start=2, end=8, width=6, strand="+", raw_score=10.0, scaled_score=10.0),
        WindowHit(tf="lexA", start=0, end=4, width=4, strand="+", raw_score=8.0, scaled_score=8.0),
        WindowHit(tf="lexA", start=4, end=8, width=4, strand="+", raw_score=8.0, scaled_score=8.0),
    )
    spec = TopKDistinctSelectorSpec(
        copies=2,
        distinctness=DistinctnessSpec(mode="interval", min_gap=0, strand_rule="collapse_same_locus"),
    )

    selected = selector.select(hits, spec)

    assert [(hit.start, hit.end) for hit in selected] == [(0, 4), (4, 8)]
    assert min(hit.scaled_score for hit in selected) == pytest.approx(8.0)


def test_top_k_distinct_selector_respects_min_gap_and_same_locus_collapse() -> None:
    selector = TopKDistinctSelector()
    hits = (
        WindowHit(tf="lexA", start=0, end=4, width=4, strand="+", raw_score=9.0, scaled_score=9.0),
        WindowHit(tf="lexA", start=0, end=4, width=4, strand="-", raw_score=8.5, scaled_score=8.5),
        WindowHit(tf="lexA", start=4, end=8, width=4, strand="+", raw_score=8.0, scaled_score=8.0),
        WindowHit(tf="lexA", start=5, end=9, width=4, strand="+", raw_score=7.5, scaled_score=7.5),
    )
    spec = TopKDistinctSelectorSpec(
        copies=2,
        distinctness=DistinctnessSpec(mode="interval", min_gap=1, strand_rule="collapse_same_locus"),
    )

    selected = selector.select(hits, spec)

    assert [(hit.start, hit.end, hit.strand) for hit in selected] == [(0, 4, "+"), (5, 9, "+")]


def test_top_k_distinct_selector_supports_overlap_tolerant_offset_mode() -> None:
    selector = TopKDistinctSelector()
    hits = (
        WindowHit(tf="lexA", start=0, end=4, width=4, strand="+", raw_score=9.0, scaled_score=9.0),
        WindowHit(tf="lexA", start=0, end=4, width=4, strand="-", raw_score=8.5, scaled_score=8.5),
        WindowHit(tf="lexA", start=1, end=5, width=4, strand="+", raw_score=8.7, scaled_score=8.7),
        WindowHit(tf="lexA", start=2, end=6, width=4, strand="+", raw_score=8.2, scaled_score=8.2),
    )
    spec = TopKDistinctSelectorSpec(
        copies=3,
        distinctness=DistinctnessSpec(mode="offset", min_gap=0, strand_rule="collapse_same_locus"),
    )

    selected = selector.select(hits, spec)

    assert [(hit.start, hit.end, hit.strand) for hit in selected] == [(0, 4, "+"), (1, 5, "+"), (2, 6, "+")]


def test_top_k_distinct_selector_breaks_ties_by_earliest_interval_signature() -> None:
    selector = TopKDistinctSelector()
    hits = (
        WindowHit(tf="lexA", start=1, end=3, width=2, strand="+", raw_score=5.0, scaled_score=5.0),
        WindowHit(tf="lexA", start=3, end=5, width=2, strand="+", raw_score=5.0, scaled_score=5.0),
        WindowHit(tf="lexA", start=0, end=2, width=2, strand="+", raw_score=5.0, scaled_score=5.0),
        WindowHit(tf="lexA", start=4, end=6, width=2, strand="+", raw_score=5.0, scaled_score=5.0),
    )
    spec = TopKDistinctSelectorSpec(
        copies=2,
        distinctness=DistinctnessSpec(mode="interval", min_gap=0, strand_rule="collapse_same_locus"),
    )

    selected = selector.select(hits, spec)

    assert [(hit.start, hit.end, hit.strand) for hit in selected] == [(0, 2, "+"), (3, 5, "+")]


def test_objective_engine_returns_k_distinct_weakest_scalar() -> None:
    tf = "lexA"
    pwms = {tf: _motif_pwm(tf, "AA")}
    scorer = Scorer(pwms, scale="llr", bidirectional=False)
    objective = ObjectiveSpec(
        objective_id=tf,
        tf=tf,
        pwm_source_id=tf,
        score_scale="llr",
        bidirectional=False,
        selector=TopKDistinctSelectorSpec(
            copies=2,
            distinctness=DistinctnessSpec(mode="interval", min_gap=0, strand_rule="collapse_same_locus"),
        ),
        aggregator=HitAggregationSpec(kind="weakest_selected"),
        capabilities=ObjectiveCapabilities(
            supports_incremental_rescore=False,
            supports_targeted_window_hint=False,
            supports_representative_hit_artifact=False,
        ),
        metadata={"objective_kind": "k_distinct_weakest"},
    )
    engine = ObjectiveEngine(scorer=scorer, objectives=(objective,))
    seq = np.asarray([0, 0, 3, 0, 0], dtype=np.int8)  # AATAA

    evaluation = engine.evaluate(seq, seq_length=int(seq.size))
    result = evaluation.results[tf]

    assert evaluation.scalars[tf] == pytest.approx(result.scalar)
    assert [hit.start for hit in result.selected_hits] == [0, 3]
    assert result.scalar == pytest.approx(min(hit.scaled_score for hit in result.selected_hits))
    assert result.diagnostics["requested_copies"] == 2
    assert result.diagnostics["selected_copies"] == 2
    assert [hit.window_seq for hit in result.selected_hits] == ["AA", "AA"]
    assert [hit.core_seq for hit in result.selected_hits] == ["AA", "AA"]
