"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/objectives/engine.py

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dnadesign.cruncher.core.scoring import Scorer

from .aggregators import BestHitAggregator, WeakestSelectedAggregator
from .models import ObjectiveResult, ObjectiveSpec, SelectedHit, WindowHit
from .selectors import BestHitSelector, TopKDistinctSelector


@dataclass(frozen=True)
class ObjectiveEngineEvaluation:
    scalars: dict[str, float]
    results: dict[str, ObjectiveResult]


class ObjectiveEngine:
    def __init__(self, *, scorer: Scorer, objectives: Sequence[ObjectiveSpec]) -> None:
        self.scorer = scorer
        self.objectives = tuple(objectives)
        if not self.objectives:
            raise ValueError("ObjectiveEngine requires at least one objective.")
        self._selectors = {
            "best_hit": BestHitSelector(),
            "top_k_distinct": TopKDistinctSelector(),
        }
        self._aggregators = {
            "best_hit": BestHitAggregator(),
            "weakest_selected": WeakestSelectedAggregator(),
        }

    def evaluate(self, seq, *, seq_length: int) -> ObjectiveEngineEvaluation:
        scalars: dict[str, float] = {}
        results: dict[str, ObjectiveResult] = {}
        for objective in self.objectives:
            result = self._evaluate_objective(objective, seq=seq, seq_length=seq_length)
            scalars[objective.objective_id] = float(result.scalar)
            results[objective.objective_id] = result
        return ObjectiveEngineEvaluation(scalars=scalars, results=results)

    def evaluate_objective(self, objective_id: str, seq, *, seq_length: int) -> ObjectiveResult:
        objective = next((item for item in self.objectives if item.objective_id == objective_id), None)
        if objective is None:
            raise KeyError(f"Unknown objective_id: {objective_id!r}")
        return self._evaluate_objective(objective, seq=seq, seq_length=seq_length)

    def _evaluate_objective(self, objective: ObjectiveSpec, *, seq, seq_length: int) -> ObjectiveResult:
        if objective.selector.kind == "best_hit":
            return self._evaluate_best_hit_objective(objective, seq=seq, seq_length=seq_length)

        scan_hits = self.scorer.scan_hits(
            seq,
            objective.tf,
            seq_length,
            scale=objective.score_scale,
            include_sequences=False,
        )
        selector = self._selectors.get(objective.selector.kind)
        if selector is None:
            raise ValueError(f"Unsupported objective selector: {objective.selector.kind!r}")
        selected = selector.select(scan_hits, objective.selector)
        selected_hits = tuple(self._to_selected_hit(seq, objective.tf, seq_length, hit) for hit in selected)
        requested_copies = int(getattr(objective.selector, "copies", len(selected_hits) or 1))
        feasible = len(selected_hits) >= requested_copies
        aggregator = self._aggregators.get(objective.aggregator.kind)
        if aggregator is None:
            raise ValueError(f"Unsupported objective aggregator: {objective.aggregator.kind!r}")
        scalar = aggregator.scalar(selected_hits, objective.aggregator) if feasible else float("-inf")
        normalized_scalar = (
            self._aggregate_normalized(
                selected_hits,
                aggregator_kind=objective.aggregator.kind,
            )
            if feasible
            else float("-inf")
        )
        representative = self._representative_hit(selected_hits, aggregator_kind=objective.aggregator.kind)
        return ObjectiveResult(
            objective_id=objective.objective_id,
            tf=objective.tf,
            scalar=float(scalar),
            selected_hits=selected_hits,
            representative_hit=representative,
            diagnostics={
                "objective_kind": objective.metadata.get("objective_kind"),
                "requested_copies": requested_copies,
                "selected_copies": len(selected_hits),
                "feasible": feasible,
                "normalized_scalar": normalized_scalar,
            },
        )

    def _evaluate_best_hit_objective(self, objective: ObjectiveSpec, *, seq, seq_length: int) -> ObjectiveResult:
        hit = self.scorer.best_hit(seq, objective.tf)
        selected_hit = self._to_selected_hit(
            seq,
            objective.tf,
            seq_length,
            WindowHit(
                tf=objective.tf,
                start=int(hit["best_start"]),
                end=int(hit["best_start"]) + int(hit["width"]),
                width=int(hit["width"]),
                strand=str(hit["strand"]),
                raw_score=float(hit["best_score_raw"]),
                scaled_score=self.scorer.scale_raw_score(
                    objective.tf,
                    float(hit["best_score_raw"]),
                    seq_length,
                    scale=objective.score_scale,
                ),
                window_seq=str(hit.get("best_window_seq") or ""),
                core_seq=str(hit.get("best_core_seq") or ""),
                tiebreak_rule=str(hit.get("best_hit_tiebreak") or "best_hit"),
            ),
        )
        return ObjectiveResult(
            objective_id=objective.objective_id,
            tf=objective.tf,
            scalar=float(selected_hit.scaled_score),
            selected_hits=(selected_hit,),
            representative_hit=selected_hit,
            diagnostics={
                "objective_kind": objective.metadata.get("objective_kind"),
                "requested_copies": 1,
                "selected_copies": 1,
                "feasible": True,
                "normalized_scalar": float(selected_hit.normalized_score),
            },
        )

    def _to_selected_hit(self, seq, tf: str, seq_length: int, hit: WindowHit) -> SelectedHit:
        normalized_score = self.scorer.scale_raw_score(tf, hit.raw_score, seq_length, scale="normalized-llr")
        window_seq = hit.window_seq
        core_seq = hit.core_seq
        if window_seq is None or core_seq is None:
            window_seq, core_seq = self.scorer.hit_sequences(
                seq,
                start=int(hit.start),
                width=int(hit.width),
                strand=str(hit.strand),
            )
        return SelectedHit(
            tf=hit.tf,
            start=int(hit.start),
            end=int(hit.end),
            width=int(hit.width),
            strand=str(hit.strand),
            raw_score=float(hit.raw_score),
            scaled_score=float(hit.scaled_score),
            normalized_score=float(normalized_score),
            window_seq=window_seq,
            core_seq=core_seq,
            tiebreak_rule=hit.tiebreak_rule,
        )

    @staticmethod
    def _aggregate_normalized(selected_hits: Sequence[SelectedHit], *, aggregator_kind: str) -> float:
        if not selected_hits:
            return float("-inf")
        if aggregator_kind == "best_hit":
            return float(selected_hits[0].normalized_score)
        if aggregator_kind == "weakest_selected":
            return float(min(hit.normalized_score for hit in selected_hits))
        raise ValueError(f"Unsupported normalized aggregator: {aggregator_kind!r}")

    @staticmethod
    def _representative_hit(
        selected_hits: Sequence[SelectedHit],
        *,
        aggregator_kind: str,
    ) -> SelectedHit | None:
        if not selected_hits:
            return None
        if aggregator_kind == "weakest_selected":
            return min(
                selected_hits,
                key=lambda hit: (hit.scaled_score, hit.start, hit.end, hit.strand),
            )
        return selected_hits[0]
