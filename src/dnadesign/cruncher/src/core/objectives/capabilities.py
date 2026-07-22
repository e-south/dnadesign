"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/core/objectives/capabilities.py

Core runtime primitives for capabilities Cruncher core objectives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import ObjectiveCapabilities, ObjectiveSpec


@dataclass(frozen=True)
class ScoreScaleCapabilities:
    per_window_scalar: bool
    occurrence_aggregation_safe: bool


@dataclass(frozen=True)
class ObjectiveRuntimeCapabilities:
    supports_incremental_rescore: bool
    supports_targeted_window_hint: bool
    supports_representative_hit_artifact: bool


BEST_HIT_CAPABILITIES = ObjectiveCapabilities(
    supports_incremental_rescore=True,
    supports_targeted_window_hint=True,
    supports_representative_hit_artifact=True,
)

KDISTINCT_V1_CAPABILITIES = ObjectiveCapabilities(
    supports_incremental_rescore=False,
    supports_targeted_window_hint=False,
    supports_representative_hit_artifact=False,
)


def resolve_score_scale_capabilities(scale: str) -> ScoreScaleCapabilities:
    normalized = str(scale).strip().lower()
    if normalized in {"llr", "normalized-llr", "z"}:
        return ScoreScaleCapabilities(per_window_scalar=True, occurrence_aggregation_safe=True)
    if normalized in {"logp", "consensus-neglop-sum"}:
        return ScoreScaleCapabilities(per_window_scalar=False, occurrence_aggregation_safe=False)
    raise ValueError(f"Unsupported score scale for capability resolution: {scale!r}")


def resolve_runtime_capabilities(objectives: Iterable[ObjectiveSpec]) -> ObjectiveRuntimeCapabilities:
    objective_list = list(objectives)
    if not objective_list:
        raise ValueError("Objective runtime capability resolution requires at least one objective.")
    return ObjectiveRuntimeCapabilities(
        supports_incremental_rescore=all(
            objective.capabilities.supports_incremental_rescore for objective in objective_list
        ),
        supports_targeted_window_hint=all(
            objective.capabilities.supports_targeted_window_hint for objective in objective_list
        ),
        supports_representative_hit_artifact=all(
            objective.capabilities.supports_representative_hit_artifact for objective in objective_list
        ),
    )
