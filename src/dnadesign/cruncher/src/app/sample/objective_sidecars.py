"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/objective_sidecars.py

Persist occurrence-aware objective sidecars for sample runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from dnadesign.cruncher.app.sample.artifacts import (
    _objective_scores_parquet_schema,
    _occurrences_parquet_schema,
    _write_parquet_rows,
)
from dnadesign.cruncher.artifacts.layout import elites_objective_scores_path, elites_occurrences_path
from dnadesign.cruncher.core.objectives.compiler import ObjectivePlanCompilation


def write_elite_objective_sidecars(
    *,
    out_dir: Path,
    entries: list[dict[str, object]],
    evaluator: object,
    objective_plan: ObjectivePlanCompilation,
) -> tuple[Path, Path]:
    objective_scores_file = elites_objective_scores_path(out_dir)
    occurrences_file = elites_occurrences_path(out_dir)
    objective_schema = _objective_scores_parquet_schema(id_field="elite_id")
    occurrence_schema = _occurrences_parquet_schema(id_field="elite_id")
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}

    def _objective_rows() -> Iterable[dict[str, object]]:
        objective_by_id = {objective.objective_id: objective for objective in objective_plan.objectives}
        engine = getattr(evaluator, "objective_engine", None)
        if engine is None:
            raise ValueError("Evaluator missing objective_engine for objective sidecars.")
        for entry in entries:
            elite_id = str(entry["id"])
            sequence = str(entry["sequence"])
            seq_arr = np.asarray([base_map[base] for base in sequence], dtype=np.int8)
            evaluation = engine.evaluate(seq_arr, seq_length=int(seq_arr.size))
            for objective_id, result in evaluation.results.items():
                objective = objective_by_id[objective_id]
                requested_copies = int(getattr(objective.selector, "copies", 1))
                yield {
                    "elite_id": elite_id,
                    "objective_id": objective_id,
                    "tf": result.tf,
                    "pwm_source_id": objective.pwm_source_id,
                    "objective_kind": objective.metadata.get("objective_kind"),
                    "score_scale": objective.score_scale,
                    "scalar_score": float(result.scalar),
                    "normalized_scalar": float(result.diagnostics.get("normalized_scalar", float("-inf"))),
                    "requested_copies": requested_copies,
                    "selected_copies": int(result.diagnostics.get("selected_copies", len(result.selected_hits))),
                    "selection_kind": objective.selector.kind,
                    "aggregation_kind": objective.aggregator.kind,
                    "sequence_length": int(seq_arr.size),
                }

    def _occurrence_rows() -> Iterable[dict[str, object]]:
        objective_by_id = {objective.objective_id: objective for objective in objective_plan.objectives}
        engine = getattr(evaluator, "objective_engine", None)
        if engine is None:
            raise ValueError("Evaluator missing objective_engine for objective sidecars.")
        for entry in entries:
            elite_id = str(entry["id"])
            sequence = str(entry["sequence"])
            seq_arr = np.asarray([base_map[base] for base in sequence], dtype=np.int8)
            evaluation = engine.evaluate(seq_arr, seq_length=int(seq_arr.size))
            for objective_id, result in evaluation.results.items():
                objective = objective_by_id[objective_id]
                distinctness = getattr(objective.selector, "distinctness", None)
                for occurrence_rank, hit in enumerate(result.selected_hits, start=1):
                    yield {
                        "elite_id": elite_id,
                        "objective_id": objective_id,
                        "tf": result.tf,
                        "occurrence_rank": occurrence_rank,
                        "start": int(hit.start),
                        "end": int(hit.end),
                        "strand": str(hit.strand),
                        "raw_score": float(hit.raw_score),
                        "scaled_score": float(hit.scaled_score),
                        "normalized_score": float(hit.normalized_score),
                        "selected": True,
                        "distinctness_mode": getattr(distinctness, "mode", "best_hit"),
                        "min_gap": int(getattr(distinctness, "min_gap", 0) or 0),
                        "locus_group": f"{int(hit.start)}:{int(hit.end)}",
                    }

    _write_parquet_rows(objective_scores_file, _objective_rows(), chunk_size=2000, schema=objective_schema)
    _write_parquet_rows(occurrences_file, _occurrence_rows(), chunk_size=2000, schema=occurrence_schema)
    return objective_scores_file, occurrences_file
