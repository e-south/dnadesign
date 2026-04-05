"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/objectives/compiler.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dnadesign.cruncher.config.schema_v3 import SampleConfig
from dnadesign.cruncher.core.pwm import PWM

from .capabilities import (
    BEST_HIT_CAPABILITIES,
    KDISTINCT_V1_CAPABILITIES,
    ObjectiveRuntimeCapabilities,
    resolve_runtime_capabilities,
)
from .models import (
    BestHitAggregationSpec,
    BestHitSelectorSpec,
    DistinctnessSpec,
    ObjectiveSpec,
    TopKDistinctSelectorSpec,
    WeakestSelectedAggregationSpec,
)


@dataclass(frozen=True)
class ObjectivePlanCompilation:
    objectives: tuple[ObjectiveSpec, ...]
    runtime: ObjectiveRuntimeCapabilities


def compile_objective_plan(
    *,
    sample_cfg: SampleConfig,
    tfs: Sequence[str],
    pwms: dict[str, PWM],
) -> ObjectivePlanCompilation:
    multiplicity_cfg = sample_cfg.objective.multiplicity
    objectives: list[ObjectiveSpec] = []
    if multiplicity_cfg.enabled:
        if len(tfs) != 1:
            raise ValueError("Multiplicity objective compilation requires exactly one TF.")
        tf = str(tfs[0])
        if tf not in pwms:
            raise ValueError(f"Missing PWM for compiled multiplicity objective TF '{tf}'.")
        objectives.append(
            ObjectiveSpec(
                objective_id=tf,
                tf=tf,
                pwm_source_id=tf,
                score_scale=sample_cfg.objective.score_scale,
                bidirectional=bool(sample_cfg.objective.bidirectional),
                selector=TopKDistinctSelectorSpec(
                    copies=int(multiplicity_cfg.copies),
                    distinctness=DistinctnessSpec(
                        mode=str(multiplicity_cfg.distinctness.mode),
                        min_gap=int(multiplicity_cfg.distinctness.min_gap),
                        strand_rule=str(multiplicity_cfg.distinctness.strand_rule),
                    ),
                ),
                aggregator=WeakestSelectedAggregationSpec(),
                capabilities=KDISTINCT_V1_CAPABILITIES,
                metadata={"objective_kind": "k_distinct_weakest"},
            )
        )
    else:
        for tf in tfs:
            tf_name = str(tf)
            if tf_name not in pwms:
                raise ValueError(f"Missing PWM for compiled objective TF '{tf_name}'.")
            objectives.append(
                ObjectiveSpec(
                    objective_id=tf_name,
                    tf=tf_name,
                    pwm_source_id=tf_name,
                    score_scale=sample_cfg.objective.score_scale,
                    bidirectional=bool(sample_cfg.objective.bidirectional),
                    selector=BestHitSelectorSpec(),
                    aggregator=BestHitAggregationSpec(),
                    capabilities=BEST_HIT_CAPABILITIES,
                    metadata={"objective_kind": "best_hit"},
                )
            )
    runtime = resolve_runtime_capabilities(objectives)
    return ObjectivePlanCompilation(objectives=tuple(objectives), runtime=runtime)
