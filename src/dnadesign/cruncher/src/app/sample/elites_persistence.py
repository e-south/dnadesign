"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/elites_persistence.py

Persistence and metadata helpers for sampled elite outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from dnadesign.cruncher.app.sample.artifacts import (
    _elite_hits_parquet_schema,
    _elite_parquet_schema,
    _write_parquet_rows,
)
from dnadesign.cruncher.artifacts.entries import artifact_entry
from dnadesign.cruncher.artifacts.layout import (
    config_used_path,
    elites_hits_path,
    elites_mmr_meta_path,
    elites_path,
)
from dnadesign.cruncher.config.schema_v3 import SampleConfig
from dnadesign.cruncher.core.labels import format_regulator_slug


def _elite_rows_from(entries: list[dict[str, object]]) -> Iterable[dict[str, object]]:
    for entry in entries:
        row = dict(entry)
        per_tf = row.pop("per_tf", None)
        if per_tf is not None:
            row["per_tf_json"] = json.dumps(per_tf, sort_keys=True)
            for tf_name, details in per_tf.items():
                row[f"score_{tf_name}"] = details.get("best_score_scaled")
                row[f"norm_{tf_name}"] = details.get("best_score_norm")
        yield row


def _elite_hits_rows(
    *,
    entries: list[dict[str, object]],
    pwms: dict[str, object],
    pwm_ref_by_tf: dict[str, str | None],
    pwm_hash_by_tf: dict[str, str | None],
    core_def_by_tf: dict[str, str],
) -> Iterable[dict[str, object]]:
    for entry in entries:
        elite_id = entry.get("id")
        rank = entry.get("rank")
        chain_id = entry.get("chain")
        draw_idx = entry.get("draw_idx")
        per_tf = entry.get("per_tf")
        if not isinstance(per_tf, dict):
            raise ValueError("Elite entry missing per_tf metadata.")
        for tf_name, details in per_tf.items():
            if not isinstance(details, dict):
                raise ValueError(f"Elite per_tf details missing for '{tf_name}'.")
            start = details.get("best_start")
            if start is None:
                start = details.get("offset")
            if not isinstance(start, int):
                raise ValueError(f"Elite hit missing best_start for '{tf_name}'.")
            strand = details.get("strand")
            if not isinstance(strand, str):
                raise ValueError(f"Elite hit missing strand for '{tf_name}'.")
            width = details.get("width")
            pwm = pwms.get(tf_name)
            if pwm is None:
                raise ValueError(f"Missing PWM for TF '{tf_name}'.")
            pwm_width = int(pwm.length)
            core_width = int(width) if isinstance(width, int) else pwm_width
            yield {
                "elite_id": elite_id,
                "tf": tf_name,
                "rank": rank,
                "chain": chain_id,
                "draw_idx": draw_idx,
                "best_start": int(start),
                "best_core_offset": int(start),
                "best_strand": strand,
                "best_window_seq": details.get("best_window_seq"),
                "best_core_seq": details.get("best_core_seq"),
                "best_score_raw": details.get("best_score_raw"),
                "best_score_scaled": details.get("best_score_scaled"),
                "best_score_norm": details.get("best_score_norm"),
                "tiebreak_rule": details.get("best_hit_tiebreak"),
                "pwm_ref": pwm_ref_by_tf.get(tf_name),
                "pwm_hash": pwm_hash_by_tf.get(tf_name),
                "pwm_width": pwm_width,
                "core_width": core_width,
                "core_def_hash": core_def_by_tf.get(tf_name),
            }


def _write_elite_tables(
    *,
    out_dir: Path,
    tfs: list[str],
    elites: list[dict[str, object]],
    pwms: dict[str, object],
    pwm_ref_by_tf: dict[str, str | None],
    pwm_hash_by_tf: dict[str, str | None],
    core_def_by_tf: dict[str, str],
    want_canonical: bool,
    write_representative_hits: bool,
) -> tuple[Path, Path | None]:
    parquet_path = elites_path(out_dir)
    elite_schema = _elite_parquet_schema(tfs, include_canonical=want_canonical)
    _write_parquet_rows(parquet_path, _elite_rows_from(elites), chunk_size=2000, schema=elite_schema)

    hits_path: Path | None = None
    if write_representative_hits:
        hits_path = elites_hits_path(out_dir)
        hits_schema = _elite_hits_parquet_schema()
        _write_parquet_rows(
            hits_path,
            _elite_hits_rows(
                entries=elites,
                pwms=pwms,
                pwm_ref_by_tf=pwm_ref_by_tf,
                pwm_hash_by_tf=pwm_hash_by_tf,
                core_def_by_tf=core_def_by_tf,
            ),
            chunk_size=2000,
            schema=hits_schema,
        )
    return parquet_path, hits_path


def _selection_fields(
    *,
    mmr_summary: dict[str, object] | None,
    diversity_value: float,
) -> tuple[str, str, str, float, float]:
    selection_policy = "mmr"
    relevance_label = "min_tf_score"
    pool_strategy = "stratified"
    score_weight = 1.0 - diversity_value
    diversity_weight = diversity_value
    if isinstance(mmr_summary, dict):
        selection_policy = str(mmr_summary.get("selection_policy") or selection_policy)
        relevance_label = str(mmr_summary.get("relevance") or relevance_label)
        pool_strategy = str(mmr_summary.get("pool_strategy") or pool_strategy)
        score_weight_meta = mmr_summary.get("score_weight")
        diversity_weight_meta = mmr_summary.get("diversity_weight")
        if score_weight_meta is not None:
            score_weight = float(score_weight_meta)
        if diversity_weight_meta is not None:
            diversity_weight = float(diversity_weight_meta)
    return selection_policy, relevance_label, pool_strategy, score_weight, diversity_weight


def _write_mmr_meta(*, out_dir: Path, mmr_meta_rows: list[dict[str, object]] | None) -> Path | None:
    if not mmr_meta_rows:
        return None
    mmr_meta_path = elites_mmr_meta_path(out_dir)
    _write_parquet_rows(mmr_meta_path, mmr_meta_rows, chunk_size=2000)
    return mmr_meta_path


def _append_optimizer_stats(*, meta: dict[str, object], optimizer: object) -> None:
    if not hasattr(optimizer, "stats"):
        return
    optimizer_stats = optimizer.stats()
    if not isinstance(optimizer_stats, dict) or not optimizer_stats:
        return
    beta_base = optimizer_stats.get("beta_ladder_base")
    beta_final = optimizer_stats.get("beta_ladder_final")
    if isinstance(beta_base, (list, tuple)):
        meta["beta_ladder_base"] = [float(item) for item in beta_base]
    if isinstance(beta_final, (list, tuple)):
        meta["beta_ladder_final"] = [float(item) for item in beta_final]
    final_beta = optimizer_stats.get("final_mcmc_beta")
    meta["final_mcmc_beta"] = float(final_beta) if final_beta is not None else None
    cooling_payload = optimizer_stats.get("mcmc_cooling")
    if isinstance(cooling_payload, dict):
        meta["mcmc_cooling"] = cooling_payload


def _build_elites_metadata(
    *,
    sample_cfg: SampleConfig,
    tfs: list[str],
    out_dir: Path,
    elites: list[dict[str, object]],
    raw_elites: list[object],
    kept_after_mmr: int,
    total_draws_seen: int,
    combine_resolved: str,
    beta_softmin_final: float | None,
    pool_size: int,
    diversity_value: float,
    dsdna_mode: bool,
    mmr_summary: dict[str, object] | None,
    optimizer: object,
    postprocess_stats: dict[str, int] | None = None,
) -> dict[str, object]:
    tf_label = format_regulator_slug(tfs)
    selection_policy, relevance_label, pool_strategy, score_weight, diversity_weight = _selection_fields(
        mmr_summary=mmr_summary,
        diversity_value=diversity_value,
    )
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_elites": len(elites),
        "selection_policy": selection_policy,
        "selection_relevance": relevance_label,
        "selection_score_weight": score_weight,
        "selection_diversity_weight": diversity_weight,
        "selection_diversity": diversity_value,
        "dsdna_canonicalize": dsdna_mode,
        "total_draws_seen": total_draws_seen,
        "candidate_count": len(raw_elites),
        "kept_after_mmr": kept_after_mmr,
        "objective_combine": combine_resolved,
        "softmin_final_beta_used": beta_softmin_final,
        "pool_size": pool_size,
        "pool_strategy": pool_strategy,
        "indexing_note": (
            "chain is a 0-based independent chain index; chain_1based is 1-based; "
            "draw_idx/sweep_idx are absolute sweeps; "
            "draw_in_phase is phase-relative"
        ),
        "tf_label": tf_label,
        "sequence_length": sample_cfg.sequence_length,
        "config_file": str(config_used_path(out_dir).resolve()),
    }
    _append_optimizer_stats(meta=meta, optimizer=optimizer)
    if mmr_summary is not None:
        meta["mmr_summary"] = mmr_summary
    if isinstance(postprocess_stats, dict):
        meta["postprocess"] = {
            "polish_edits": int(postprocess_stats.get("polish_edits", 0)),
            "trim_left": int(postprocess_stats.get("trim_left", 0)),
            "trim_right": int(postprocess_stats.get("trim_right", 0)),
            "trim_internal_bp": int(postprocess_stats.get("trim_internal_bp", 0)),
            "trim_internal_segments": int(postprocess_stats.get("trim_internal_segments", 0)),
            "dedup_dropped": int(postprocess_stats.get("dedup_dropped", 0)),
        }
    return meta


def _append_elite_artifacts(
    *,
    out_dir: Path,
    stage: str,
    artifacts: list[dict[str, object]],
    parquet_path: Path,
    json_path: Path,
    yaml_path: Path,
    hits_path: Path | None,
    objective_scores_path: Path | None,
    occurrences_path: Path | None,
    mmr_meta_path: Path | None,
) -> None:
    artifacts.extend(
        [
            artifact_entry(
                parquet_path,
                out_dir,
                kind="table",
                label="Elite sequences (Parquet)",
                stage=stage,
            ),
            artifact_entry(
                json_path,
                out_dir,
                kind="json",
                label="Elite sequences (JSON)",
                stage=stage,
            ),
            artifact_entry(
                yaml_path,
                out_dir,
                kind="metadata",
                label="Elite metadata (YAML)",
                stage=stage,
            ),
        ]
    )
    if hits_path is not None:
        artifacts.append(
            artifact_entry(
                hits_path,
                out_dir,
                kind="table",
                label="Elite best-hit metadata (Parquet)",
                stage=stage,
            )
        )
    if objective_scores_path is not None:
        artifacts.append(
            artifact_entry(
                objective_scores_path,
                out_dir,
                kind="table",
                label="Elite objective scores (Parquet)",
                stage=stage,
            )
        )
    if occurrences_path is not None:
        artifacts.append(
            artifact_entry(
                occurrences_path,
                out_dir,
                kind="table",
                label="Elite selected occurrences (Parquet)",
                stage=stage,
            )
        )
    if mmr_meta_path is None:
        return
    artifacts.append(
        artifact_entry(
            mmr_meta_path,
            out_dir,
            kind="table",
            label="Elite MMR selection metadata",
            stage=stage,
        )
    )
