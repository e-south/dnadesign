"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/sample/random_baseline.py

Random-baseline artifact writers for sample runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dnadesign.cruncher.app.sample.artifacts import (
    _baseline_hits_parquet_schema,
    _objective_scores_parquet_schema,
    _occurrences_parquet_schema,
)
from dnadesign.cruncher.app.sample.diagnostics import _norm_map_for_elites
from dnadesign.cruncher.artifacts.entries import artifact_entry
from dnadesign.cruncher.artifacts.layout import (
    random_baseline_hits_path,
    random_baseline_objective_scores_path,
    random_baseline_occurrences_path,
    random_baseline_path,
)
from dnadesign.cruncher.config.schema_v3 import SampleConfig
from dnadesign.cruncher.core.evaluator import SequenceEvaluator
from dnadesign.cruncher.core.objectives.capabilities import BEST_HIT_CAPABILITIES, ObjectiveRuntimeCapabilities
from dnadesign.cruncher.core.objectives.compiler import ObjectivePlanCompilation
from dnadesign.cruncher.core.objectives.models import (
    BestHitAggregationSpec,
    BestHitSelectorSpec,
    ObjectiveSpec,
)
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.sequence import canon_int
from dnadesign.cruncher.core.state import SequenceState

logger = logging.getLogger(__name__)


def _best_hit_objective_plan(
    *,
    sample_cfg: SampleConfig,
    tfs: list[str],
) -> ObjectivePlanCompilation:
    objectives = tuple(
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
        for tf_name in sorted(tfs)
    )
    return ObjectivePlanCompilation(
        objectives=objectives,
        runtime=ObjectiveRuntimeCapabilities(
            supports_incremental_rescore=True,
            supports_targeted_window_hint=True,
            supports_representative_hit_artifact=True,
        ),
    )


def _objective_result_rows(
    *,
    run_item_id: str | int,
    id_field: str,
    objective_plan: ObjectivePlanCompilation,
    evaluation: object,
    sequence_length: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    objective_rows: list[dict[str, object]] = []
    occurrence_rows: list[dict[str, object]] = []
    objective_by_id = {objective.objective_id: objective for objective in objective_plan.objectives}
    for objective_id, result in evaluation.results.items():
        objective = objective_by_id[objective_id]
        distinctness = getattr(objective.selector, "distinctness", None)
        requested_copies = int(getattr(objective.selector, "copies", 1))
        objective_rows.append(
            {
                id_field: run_item_id,
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
                "sequence_length": int(sequence_length),
            }
        )
        for occurrence_rank, hit in enumerate(result.selected_hits, start=1):
            occurrence_rows.append(
                {
                    id_field: run_item_id,
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
            )
    return objective_rows, occurrence_rows


def write_random_baseline_artifacts(
    *,
    out_dir: Path,
    sample_cfg: SampleConfig,
    objective_plan: ObjectivePlanCompilation | None = None,
    set_index: int,
    tfs: list[str],
    scorer: Scorer,
    evaluator: SequenceEvaluator | None = None,
    pwms: dict[str, object],
    pwm_ref_by_tf: dict[str, str | None],
    pwm_hash_by_tf: dict[str, str | None],
    core_def_by_tf: dict[str, str],
    stage: str,
    artifacts: list[dict[str, object]],
) -> None:
    if not sample_cfg.output.save_random_baseline:
        return
    baseline_seed = int(sample_cfg.seed + set_index - 1)
    baseline_n = int(sample_cfg.output.random_baseline_n)
    baseline_path = random_baseline_path(out_dir)
    baseline_hits_file = random_baseline_hits_path(out_dir)
    baseline_objective_scores_file = random_baseline_objective_scores_path(out_dir)
    baseline_occurrences_file = random_baseline_occurrences_path(out_dir)
    tf_order = sorted(tfs)
    baseline_canonical = bool(sample_cfg.objective.bidirectional)
    rng = np.random.default_rng(baseline_seed)
    resolved_objective_plan = objective_plan or _best_hit_objective_plan(sample_cfg=sample_cfg, tfs=tfs)
    _write_random_baseline_tables(
        baseline_path=baseline_path,
        baseline_hits_path=baseline_hits_file,
        baseline_objective_scores_file=baseline_objective_scores_file,
        baseline_occurrences_file=baseline_occurrences_file,
        baseline_seed=baseline_seed,
        baseline_n=baseline_n,
        baseline_canonical=baseline_canonical,
        sequence_length=sample_cfg.sequence_length,
        score_scale=sample_cfg.objective.score_scale,
        tf_order=tf_order,
        scorer=scorer,
        evaluator=evaluator,
        objective_plan=resolved_objective_plan,
        pwms=pwms,
        pwm_ref_by_tf=pwm_ref_by_tf,
        pwm_hash_by_tf=pwm_hash_by_tf,
        core_def_by_tf=core_def_by_tf,
        rng=rng,
    )
    artifacts.append(
        artifact_entry(
            baseline_path,
            out_dir,
            kind="table",
            label="Random baseline (Parquet)",
            stage=stage,
        )
    )
    if resolved_objective_plan.runtime.supports_representative_hit_artifact:
        artifacts.append(
            artifact_entry(
                baseline_hits_file,
                out_dir,
                kind="table",
                label="Random baseline hits (Parquet)",
                stage=stage,
            )
        )
    else:
        artifacts.append(
            artifact_entry(
                baseline_objective_scores_file,
                out_dir,
                kind="table",
                label="Random baseline objective scores (Parquet)",
                stage=stage,
            )
        )
        artifacts.append(
            artifact_entry(
                baseline_occurrences_file,
                out_dir,
                kind="table",
                label="Random baseline selected occurrences (Parquet)",
                stage=stage,
            )
        )
    logger.debug("Saved random baseline -> %s", baseline_path.relative_to(out_dir.parent))


def _write_random_baseline_tables(
    *,
    baseline_path: Path,
    baseline_hits_path: Path,
    baseline_objective_scores_file: Path,
    baseline_occurrences_file: Path,
    baseline_seed: int,
    baseline_n: int,
    baseline_canonical: bool,
    sequence_length: int,
    score_scale: str,
    tf_order: list[str],
    scorer: Scorer,
    evaluator: SequenceEvaluator | None,
    objective_plan: ObjectivePlanCompilation,
    pwms: dict[str, object],
    pwm_ref_by_tf: dict[str, str | None],
    pwm_hash_by_tf: dict[str, str | None],
    core_def_by_tf: dict[str, str],
    rng: np.random.Generator,
    chunk_size: int = 1024,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_hits_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_objective_scores_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_occurrences_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_tmp_path = baseline_path.with_suffix(baseline_path.suffix + ".tmp")
    baseline_hits_tmp_path = baseline_hits_path.with_suffix(baseline_hits_path.suffix + ".tmp")
    baseline_objective_scores_tmp_path = baseline_objective_scores_file.with_suffix(
        baseline_objective_scores_file.suffix + ".tmp"
    )
    baseline_occurrences_tmp_path = baseline_occurrences_file.with_suffix(baseline_occurrences_file.suffix + ".tmp")
    baseline_writer: pq.ParquetWriter | None = None
    baseline_hits_writer: pq.ParquetWriter | None = None
    baseline_objective_scores_writer: pq.ParquetWriter | None = None
    baseline_occurrences_writer: pq.ParquetWriter | None = None
    baseline_rows: list[dict[str, object]] = []
    baseline_hit_rows: list[dict[str, object]] = []
    baseline_objective_score_rows: list[dict[str, object]] = []
    baseline_occurrence_rows: list[dict[str, object]] = []
    hit_schema = _baseline_hits_parquet_schema()
    objective_score_schema = _objective_scores_parquet_schema(id_field="baseline_id")
    occurrence_schema = _occurrences_parquet_schema(id_field="baseline_id")

    def _flush() -> None:
        nonlocal baseline_writer
        nonlocal baseline_hits_writer
        nonlocal baseline_objective_scores_writer
        nonlocal baseline_occurrences_writer
        if baseline_rows:
            table = pa.Table.from_pylist(baseline_rows)
            if baseline_writer is None:
                baseline_writer = pq.ParquetWriter(str(baseline_tmp_path), table.schema)
            baseline_writer.write_table(table)
            baseline_rows.clear()
        if baseline_hit_rows:
            table = pa.Table.from_pylist(baseline_hit_rows, schema=hit_schema)
            if baseline_hits_writer is None:
                baseline_hits_writer = pq.ParquetWriter(str(baseline_hits_tmp_path), table.schema)
            baseline_hits_writer.write_table(table)
            baseline_hit_rows.clear()
        if baseline_objective_score_rows:
            table = pa.Table.from_pylist(baseline_objective_score_rows, schema=objective_score_schema)
            if baseline_objective_scores_writer is None:
                baseline_objective_scores_writer = pq.ParquetWriter(
                    str(baseline_objective_scores_tmp_path),
                    table.schema,
                )
            baseline_objective_scores_writer.write_table(table)
            baseline_objective_score_rows.clear()
        if baseline_occurrence_rows:
            table = pa.Table.from_pylist(baseline_occurrence_rows, schema=occurrence_schema)
            if baseline_occurrences_writer is None:
                baseline_occurrences_writer = pq.ParquetWriter(str(baseline_occurrences_tmp_path), table.schema)
            baseline_occurrences_writer.write_table(table)
            baseline_occurrence_rows.clear()

    try:
        for baseline_id in range(baseline_n):
            seq_state = SequenceState.random(sequence_length, rng)
            seq_arr = seq_state.seq
            seq_str = seq_state.to_string()
            engine = getattr(evaluator, "objective_engine", None) if evaluator is not None else None
            objective_eval = engine.evaluate(seq_arr, seq_length=int(seq_arr.size)) if engine is not None else None
            if objective_eval is not None:
                per_tf = objective_eval.scalars
            else:
                per_tf, hit_map = scorer.compute_all_per_pwm_and_hits(seq_arr, sequence_length)
            row: dict[str, object] = {
                "baseline_id": baseline_id,
                "sequence": seq_str,
                "baseline_seed": baseline_seed,
                "baseline_n": baseline_n,
                "seed": baseline_seed,
                "n_samples": baseline_n,
                "sequence_length": sequence_length,
                "length": sequence_length,
                "score_scale": score_scale,
                "bidirectional": bool(baseline_canonical),
                "bg_model": "uniform",
                "bg_a": 0.25,
                "bg_c": 0.25,
                "bg_g": 0.25,
                "bg_t": 0.25,
            }
            if baseline_canonical:
                row["canonical_sequence"] = SequenceState(canon_int(seq_arr)).to_string()
            for tf_name in tf_order:
                row[f"score_{tf_name}"] = float(per_tf[tf_name])
            if objective_plan.runtime.supports_representative_hit_artifact:
                if objective_eval is not None:
                    for tf_name in tf_order:
                        result = objective_eval.results[tf_name]
                        hit = result.representative_hit
                        if hit is None:
                            raise ValueError(f"Missing representative hit for TF '{tf_name}'.")
                        pwm = pwms.get(tf_name)
                        if pwm is None:
                            raise ValueError(f"Missing PWM for TF '{tf_name}'.")
                        baseline_hit_rows.append(
                            {
                                "baseline_id": baseline_id,
                                "tf": tf_name,
                                "best_start": int(hit.start),
                                "best_core_offset": int(hit.start),
                                "best_strand": str(hit.strand),
                                "best_window_seq": hit.window_seq,
                                "best_core_seq": hit.core_seq,
                                "best_score_raw": float(hit.raw_score),
                                "best_score_scaled": float(hit.scaled_score),
                                "best_score_norm": float(hit.normalized_score),
                                "tiebreak_rule": hit.tiebreak_rule,
                                "pwm_ref": pwm_ref_by_tf.get(tf_name),
                                "pwm_hash": pwm_hash_by_tf.get(tf_name),
                                "pwm_width": int(pwm.length),
                                "core_width": int(hit.width),
                                "core_def_hash": core_def_by_tf.get(tf_name),
                            }
                        )
                else:
                    norm_map = _norm_map_for_elites(
                        seq_arr,
                        per_tf,
                        scorer=scorer,
                        score_scale=score_scale,
                    )
                    for tf_name in tf_order:
                        hit = hit_map[tf_name]
                        pwm = pwms.get(tf_name)
                        if pwm is None:
                            raise ValueError(f"Missing PWM for TF '{tf_name}'.")
                        width = hit.get("width")
                        core_width = int(width) if isinstance(width, int) else int(pwm.length)
                        baseline_hit_rows.append(
                            {
                                "baseline_id": baseline_id,
                                "tf": tf_name,
                                "best_start": hit.get("best_start"),
                                "best_core_offset": hit.get("best_start"),
                                "best_strand": hit.get("strand"),
                                "best_window_seq": hit.get("best_window_seq"),
                                "best_core_seq": hit.get("best_core_seq"),
                                "best_score_raw": hit.get("best_score_raw"),
                                "best_score_scaled": float(per_tf[tf_name]),
                                "best_score_norm": float(norm_map.get(tf_name, 0.0)),
                                "tiebreak_rule": hit.get("best_hit_tiebreak"),
                                "pwm_ref": pwm_ref_by_tf.get(tf_name),
                                "pwm_hash": pwm_hash_by_tf.get(tf_name),
                                "pwm_width": int(pwm.length),
                                "core_width": core_width,
                                "core_def_hash": core_def_by_tf.get(tf_name),
                            }
                        )
            else:
                if objective_eval is None:
                    raise ValueError("Occurrence-aware baseline artifacts require an evaluator with objective_engine.")
                score_rows, occurrence_rows = _objective_result_rows(
                    run_item_id=baseline_id,
                    id_field="baseline_id",
                    objective_plan=objective_plan,
                    evaluation=objective_eval,
                    sequence_length=int(seq_arr.size),
                )
                baseline_objective_score_rows.extend(score_rows)
                baseline_occurrence_rows.extend(occurrence_rows)
            baseline_rows.append(row)
            if (
                len(baseline_rows) >= chunk_size
                or len(baseline_hit_rows) >= chunk_size * max(1, len(tf_order))
                or len(baseline_objective_score_rows) >= chunk_size * max(1, len(tf_order))
                or len(baseline_occurrence_rows) >= chunk_size * max(1, len(tf_order))
            ):
                _flush()
        _flush()
    except Exception:
        baseline_tmp_path.unlink(missing_ok=True)
        baseline_hits_tmp_path.unlink(missing_ok=True)
        baseline_objective_scores_tmp_path.unlink(missing_ok=True)
        baseline_occurrences_tmp_path.unlink(missing_ok=True)
        raise
    finally:
        if baseline_writer is not None:
            baseline_writer.close()
        if baseline_hits_writer is not None:
            baseline_hits_writer.close()
        if baseline_objective_scores_writer is not None:
            baseline_objective_scores_writer.close()
        if baseline_occurrences_writer is not None:
            baseline_occurrences_writer.close()

    baseline_tmp_path.replace(baseline_path)
    if objective_plan.runtime.supports_representative_hit_artifact:
        baseline_hits_tmp_path.replace(baseline_hits_path)
    else:
        if baseline_objective_scores_writer is not None:
            baseline_objective_scores_tmp_path.replace(baseline_objective_scores_file)
        if baseline_occurrences_writer is not None:
            baseline_occurrences_tmp_path.replace(baseline_occurrences_file)
