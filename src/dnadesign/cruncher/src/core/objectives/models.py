"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/core/objectives/models.py

Core runtime primitives for models Cruncher core objectives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class DistinctnessSpec:
    mode: str = "interval"
    min_gap: int = 0
    strand_rule: str = "collapse_same_locus"


@dataclass(frozen=True)
class HitSelectorSpec:
    kind: str


@dataclass(frozen=True)
class BestHitSelectorSpec(HitSelectorSpec):
    kind: str = "best_hit"


@dataclass(frozen=True)
class TopKDistinctSelectorSpec(HitSelectorSpec):
    copies: int = 1
    distinctness: DistinctnessSpec = field(default_factory=DistinctnessSpec)
    kind: str = "top_k_distinct"


@dataclass(frozen=True)
class HitAggregationSpec:
    kind: str


@dataclass(frozen=True)
class BestHitAggregationSpec(HitAggregationSpec):
    kind: str = "best_hit"


@dataclass(frozen=True)
class WeakestSelectedAggregationSpec(HitAggregationSpec):
    kind: str = "weakest_selected"


@dataclass(frozen=True)
class ObjectiveCapabilities:
    supports_incremental_rescore: bool
    supports_targeted_window_hint: bool
    supports_representative_hit_artifact: bool


@dataclass(frozen=True)
class ObjectiveSpec:
    objective_id: str
    tf: str
    pwm_source_id: str
    score_scale: str
    bidirectional: bool
    selector: HitSelectorSpec
    aggregator: HitAggregationSpec
    capabilities: ObjectiveCapabilities
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowHit:
    tf: str
    start: int
    end: int
    width: int
    strand: str
    raw_score: float
    scaled_score: float
    window_seq: str | None = None
    core_seq: str | None = None
    tiebreak_rule: str | None = None


@dataclass(frozen=True)
class SelectedHit:
    tf: str
    start: int
    end: int
    width: int
    strand: str
    raw_score: float
    scaled_score: float
    normalized_score: float
    window_seq: str | None = None
    core_seq: str | None = None
    tiebreak_rule: str | None = None


@dataclass(frozen=True)
class ObjectiveResult:
    objective_id: str
    tf: str
    scalar: float
    selected_hits: tuple[SelectedHit, ...]
    representative_hit: SelectedHit | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
