"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/objectives/aggregators.py

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Protocol, Sequence

from .models import HitAggregationSpec, SelectedHit


class HitAggregator(Protocol):
    def scalar(self, selected_hits: Sequence[SelectedHit], spec: HitAggregationSpec) -> float: ...


class BestHitAggregator:
    def scalar(self, selected_hits: Sequence[SelectedHit], spec: HitAggregationSpec) -> float:
        _ = spec
        if not selected_hits:
            return float("-inf")
        return float(selected_hits[0].scaled_score)


class WeakestSelectedAggregator:
    def scalar(self, selected_hits: Sequence[SelectedHit], spec: HitAggregationSpec) -> float:
        _ = spec
        if not selected_hits:
            return float("-inf")
        return float(min(hit.scaled_score for hit in selected_hits))
