"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/run_set.py

Run a sample set and persist all sampling artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dnadesign.cruncher.app.parse_signature import compute_parse_signature
from dnadesign.cruncher.app.progress import progress_adapter
from dnadesign.cruncher.app.run_service import update_run_index_from_status
from dnadesign.cruncher.app.sample.artifacts import (
    _baseline_hits_parquet_schema,
    _elite_hits_parquet_schema,
    _elite_parquet_schema,
    _save_config,
    _write_parquet_rows,
)
from dnadesign.cruncher.app.sample.diagnostics import _norm_map_for_elites
from dnadesign.cruncher.app.sample.elites_mmr import (
    build_elite_entries,
    build_elite_pool,
    select_elites_mmr,
)
from dnadesign.cruncher.app.sample.resources import _load_pwms_for_set
from dnadesign.cruncher.app.sample.run_layout import prepare_run_layout, write_run_manifest_and_update
from dnadesign.cruncher.app.telemetry import RunTelemetry
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_yaml
from dnadesign.cruncher.artifacts.entries import artifact_entry
from dnadesign.cruncher.artifacts.layout import (
    config_used_path,
    elites_hits_path,
    elites_json_path,
    elites_mmr_meta_path,
    elites_path,
    elites_yaml_path,
    random_baseline_hits_path,
    random_baseline_path,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.config.moves import resolve_move_config
from dnadesign.cruncher.config.schema_v3 import CruncherConfig, SampleConfig
from dnadesign.cruncher.core.evaluator import SequenceEvaluator
from dnadesign.cruncher.core.labels import format_regulator_slug
from dnadesign.cruncher.core.optimizers.cooling import make_beta_scheduler
from dnadesign.cruncher.core.optimizers.kinds import resolve_optimizer_kind
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.sequence import canon_int
from dnadesign.cruncher.core.state import SequenceState
from dnadesign.cruncher.store.lockfile import read_lockfile
from dnadesign.cruncher.utils.hashing import sha256_bytes
from dnadesign.cruncher.utils.paths import resolve_lock_path

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised when run configuration is internally contradictory."""


class RunError(RuntimeError):
    """Raised when run-time state violates strict sampling contracts."""


def _assert_sequence_buffers_aligned(optimizer: object) -> tuple[list[object], list[object], list[object]]:
    required_buffers = ("all_meta", "all_samples", "all_scores")
    resolved: dict[str, list[object]] = {}
    for name in required_buffers:
        value = getattr(optimizer, name, None)
        if not isinstance(value, list):
            raise RunError(f"Optimizer missing required sequence buffer '{name}' while saving sequences.")
        resolved[name] = value
    lengths = {name: len(values) for name, values in resolved.items()}
    if len(set(lengths.values())) != 1:
        raise RunError(
            "Optimizer sequence buffers have inconsistent lengths: "
            + ", ".join(f"{name}={lengths[name]}" for name in required_buffers)
        )
    return resolved["all_meta"], resolved["all_samples"], resolved["all_scores"]


def _assert_trace_meta_aligned(optimizer: object, *, expected_rows: int) -> list[dict[str, object]]:
    value = getattr(optimizer, "all_trace_meta", None)
    if not isinstance(value, list):
        raise RunError("Optimizer missing required sequence buffer 'all_trace_meta' while saving sequences.")
    if len(value) != expected_rows:
        raise RunError(
            f"Optimizer sequence trace metadata length mismatch: all_trace_meta={len(value)} expected={expected_rows}"
        )
    rows: list[dict[str, object]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise RunError(f"Optimizer trace metadata row {idx} must be a dictionary.")
        rows.append(item)
    return rows


def _validate_objective_preflight(sample_cfg: SampleConfig, *, n_tfs: int) -> None:
    scale = sample_cfg.objective.score_scale
    combine_cfg = sample_cfg.objective.combine
    softmin_enabled = bool(sample_cfg.objective.softmin.enabled)

    if n_tfs >= 2 and scale == "llr":
        raise ConfigError(
            "cruncher.sample.objective.score_scale='llr' is not allowed for multi-TF runs. "
            "Multi-TF runs require comparable scales. Use normalized-llr, logp, z, or consensus-neglop-sum."
        )

    if combine_cfg == "sum" and softmin_enabled:
        raise ConfigError(
            "cruncher.sample.objective.softmin.enabled=true is incompatible with "
            "cruncher.sample.objective.combine='sum'. Disable softmin or switch combine to "
            "min/softmin-compatible mode."
        )

    if scale == "consensus-neglop-sum" and softmin_enabled:
        raise ConfigError(
            "cruncher.sample.objective.softmin.enabled=true is incompatible with "
            "cruncher.sample.objective.score_scale='consensus-neglop-sum'. Disable softmin or switch combine to "
            "min/softmin-compatible mode."
        )


def _softmin_schedule_payload(sample_cfg: SampleConfig) -> dict[str, object]:
    softmin_cfg = sample_cfg.objective.softmin
    if softmin_cfg.schedule == "fixed":
        return {"kind": "fixed", "beta": float(softmin_cfg.beta_end)}
    if softmin_cfg.schedule == "linear":
        return {"kind": "linear", "beta": (float(softmin_cfg.beta_start), float(softmin_cfg.beta_end))}
    raise ConfigError(
        "Unsupported cruncher.sample.objective.softmin.schedule value "
        f"'{softmin_cfg.schedule}'. Expected 'fixed' or 'linear'."
    )


def _mcmc_cooling_payload(sample_cfg: SampleConfig) -> dict[str, object]:
    cooling_cfg = sample_cfg.optimizer.cooling
    if cooling_cfg.kind == "fixed":
        if cooling_cfg.beta is None:
            raise ConfigError("sample.optimizer.cooling.beta is required when kind='fixed'.")
        return {"kind": "fixed", "beta": float(cooling_cfg.beta)}
    if cooling_cfg.kind == "linear":
        if cooling_cfg.beta_start is None or cooling_cfg.beta_end is None:
            raise ConfigError("sample.optimizer.cooling.beta_start and beta_end are required when kind='linear'.")
        return {
            "kind": "linear",
            "beta_start": float(cooling_cfg.beta_start),
            "beta_end": float(cooling_cfg.beta_end),
        }
    if cooling_cfg.kind == "piecewise":
        stages = [{"sweeps": int(stage.sweeps), "beta": float(stage.beta)} for stage in cooling_cfg.stages]
        return {"kind": "piecewise", "stages": stages}
    raise ConfigError(f"Unsupported sample.optimizer.cooling.kind: {cooling_cfg.kind!r}.")


def _final_executed_sweep_index(optimizer: object) -> int:
    all_meta = getattr(optimizer, "all_meta", None)
    if not isinstance(all_meta, list) or not all_meta:
        raise RunError(
            "Unable to resolve final executed sweep for softmin schedule: optimizer.all_meta is missing or empty."
        )
    sweep_indices: list[int] = []
    for item in all_meta:
        if not isinstance(item, tuple) or len(item) < 2 or not isinstance(item[1], (int, np.integer)):
            raise RunError(
                "Unable to resolve final executed sweep for softmin schedule: invalid optimizer.all_meta row."
            )
        sweep_indices.append(int(item[1]))
    final_idx = max(sweep_indices)
    if final_idx < 0:
        raise RunError("Unable to resolve final executed sweep for softmin schedule: final sweep index is negative.")
    return final_idx


def _resolve_final_softmin_beta(optimizer: object, sample_cfg: SampleConfig) -> float | None:
    softmin_cfg = sample_cfg.objective.softmin
    if not softmin_cfg.enabled:
        return None

    total = int(sample_cfg.budget.tune + sample_cfg.budget.draws)
    if total < 1:
        raise ConfigError(
            "cruncher.sample.budget is invalid for softmin scheduling; tune + draws must be >= 1 when "
            "cruncher.sample.objective.softmin.enabled=true."
        )

    final_sweep_idx = _final_executed_sweep_index(optimizer)
    if final_sweep_idx >= total:
        raise RunError(
            "Unable to resolve final softmin beta: final executed sweep index exceeds configured "
            "cruncher.sample.budget.tune + cruncher.sample.budget.draws."
        )

    schedule = _softmin_schedule_payload(sample_cfg)
    try:
        beta_of = make_beta_scheduler(schedule, total)
        return float(beta_of(final_sweep_idx))
    except Exception as exc:
        raise ConfigError(
            "Failed to evaluate cruncher.sample.objective.softmin schedule for the final executed sweep."
        ) from exc


def _assert_init_length_fits_pwms(sample_cfg: SampleConfig, pwms: dict[str, PWM]) -> None:
    if not pwms:
        raise ValueError("No PWMs available for sampling.")
    max_w = max(pwm.length for pwm in pwms.values())
    if sample_cfg.sequence_length < max_w:
        names = ", ".join(f"{tf}:{pwms[tf].length}" for tf in sorted(pwms))
        raise ValueError(
            f"sequence_length={sample_cfg.sequence_length} is shorter than the widest PWM (max={max_w}). "
            f"Per-TF lengths: {names}. Increase sample.sequence_length or trim PWMs via "
            "cruncher.sample.motif_width.maxw (with strategy=max_info)."
        )


def _resolve_elite_pool_size(
    *,
    pool_size_cfg: str | int,
    elite_k: int,
    candidate_count: int,
) -> int:
    if candidate_count < 0:
        raise ValueError("candidate_count must be >= 0")
    if pool_size_cfg == "all":
        return int(candidate_count)
    if pool_size_cfg == "auto":
        auto_target = max(4000, 500 * max(int(elite_k), 1))
        auto_target = min(auto_target, 20_000)
        return min(int(candidate_count), int(auto_target))
    value = int(pool_size_cfg)
    if value < 1:
        raise ValueError("pool_size must be >= 1")
    return min(value, int(candidate_count))


def _core_def_hash(pwm: PWM, pwm_hash: str | None) -> str:
    payload = {
        "pwm_hash": pwm_hash,
        "pwm_width": pwm.length,
        "source_length": pwm.source_length,
        "window_start": pwm.window_start,
        "window_strategy": pwm.window_strategy,
        "window_score": pwm.window_score,
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def _run_sample_for_set(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    set_index: int,
    set_count: int,
    include_set_index: bool,
    tfs: list[str],
    lockmap: dict[str, object],
    sample_cfg: SampleConfig,
    stage: str = "sample",
    run_kind: str | None = None,
    force_overwrite: bool = False,
    progress_bar: bool = True,
    progress_every: int = 0,
) -> Path:
    """
    Run MCMC sampler, save config/meta plus artifacts (trace.nc, sequences.parquet, elites.*).
    Each chain gets its own independent seed (random/consensus/consensus_mix).
    """
    optimizer_kind = resolve_optimizer_kind(sample_cfg.optimizer.kind, context="sample.optimizer.kind")
    chain_count = int(sample_cfg.optimizer.chains)
    layout = prepare_run_layout(
        cfg=cfg,
        config_path=config_path,
        sample_cfg=sample_cfg,
        tfs=tfs,
        set_index=set_index,
        set_count=set_count,
        include_set_index=include_set_index,
        stage=stage,
        run_kind=run_kind,
        chain_count=chain_count,
        optimizer_kind=optimizer_kind,
        force_overwrite=force_overwrite,
    )
    out_dir = layout.run_dir
    run_group = layout.run_group
    stage_label = layout.stage_label
    run_logger = logger.info
    run_logger("=== RUN %s: %s ===", stage_label, out_dir)
    logger.debug("Full sample config: %s", sample_cfg.model_dump_json())

    status_writer = layout.status_writer
    telemetry = RunTelemetry(status_writer)
    progress = progress_adapter(progress_bar)

    def _finish_failed(exc: Exception) -> None:
        status_writer.finish(status="failed", status_message="failed", error=str(exc))
        update_run_index_from_status(
            config_path,
            out_dir,
            status_writer.payload,
            catalog_root=cfg.catalog.catalog_root,
        )
        raise exc

    # 1) LOAD all required PWMs
    pwms = _load_pwms_for_set(cfg=cfg, config_path=config_path, tfs=tfs, lockmap=lockmap)

    _assert_init_length_fits_pwms(sample_cfg, pwms)
    pwm_ref_by_tf: dict[str, str | None] = {}
    pwm_hash_by_tf: dict[str, str | None] = {}
    core_def_by_tf: dict[str, str] = {}
    for tf_name in tfs:
        locked = lockmap.get(tf_name)
        pwm_ref_by_tf[tf_name] = f"{locked.source}:{locked.motif_id}" if locked else None
        pwm_hash_by_tf[tf_name] = locked.sha256 if locked else None
        pwm = pwms.get(tf_name)
        if pwm is None:
            raise ValueError(f"Missing PWM for TF '{tf_name}'.")
        core_def_by_tf[tf_name] = _core_def_hash(pwm, pwm_hash_by_tf[tf_name])
    lockfile = read_lockfile(resolve_lock_path(config_path))
    parse_signature, parse_inputs = compute_parse_signature(
        cfg=cfg,
        lockfile=lockfile,
        tfs=tfs,
    )
    adapt_sweeps = int(sample_cfg.budget.tune)
    draws = int(sample_cfg.budget.draws)
    # 2) INSTANTIATE SCORER and SequenceEvaluator
    scale = sample_cfg.objective.score_scale
    combine_cfg = sample_cfg.objective.combine
    logger.debug("Using score_scale = %r", scale)
    _validate_objective_preflight(sample_cfg, n_tfs=len(tfs))

    def _sum_combine(values):
        return float(sum(values))

    combiner = None
    combine_resolved = "min"
    if combine_cfg == "sum":
        combiner = _sum_combine
        combine_resolved = "sum"
    elif combine_cfg == "min":
        if scale == "consensus-neglop-sum":
            combiner = min
        combine_resolved = "min"
    else:
        raise ConfigError(
            f"Unsupported cruncher.sample.objective.combine value '{combine_cfg}'. Expected 'min' or 'sum'."
        )
    logger.debug("Building Scorer and SequenceEvaluator with scale=%r", scale)
    scorer = Scorer(
        pwms,
        bidirectional=sample_cfg.objective.bidirectional,
        scale=scale,
        background=(0.25, 0.25, 0.25, 0.25),
        pseudocounts=sample_cfg.objective.scoring.pwm_pseudocounts,
        log_odds_clip=sample_cfg.objective.scoring.log_odds_clip,
    )
    evaluator = SequenceEvaluator(
        pwms=pwms,
        scale=scale,
        combiner=combiner,  # defaults to min(...) for llr/logp/normalized-llr/z
        scorer=scorer,
        bidirectional=sample_cfg.objective.bidirectional,
        background=(0.25, 0.25, 0.25, 0.25),
        pseudocounts=sample_cfg.objective.scoring.pwm_pseudocounts,
        log_odds_clip=sample_cfg.objective.scoring.log_odds_clip,
    )

    logger.debug("Scorer and SequenceEvaluator instantiated")
    logger.debug("  Scorer.scale = %r", scorer.scale)
    logger.debug("  SequenceEvaluator.scale = %r", evaluator._scale)
    logger.debug("  SequenceEvaluator.combiner = %r", evaluator._combiner)

    dsdna_mode_for_opt = bool(sample_cfg.objective.bidirectional)

    # 3) FLATTEN optimizer config
    moves = resolve_move_config(sample_cfg.moves)
    softmin_cfg = sample_cfg.objective.softmin
    softmin_sched = _softmin_schedule_payload(sample_cfg)
    mcmc_cooling = _mcmc_cooling_payload(sample_cfg)
    if not sample_cfg.output.save_sequences:
        raise ConfigError("sample.output.save_sequences must be true because analyze requires sequences.parquet")
    opt_cfg: dict[str, object] = {
        "draws": draws,
        "tune": adapt_sweeps,
        "chains": chain_count,
        "min_dist": 0,
        "top_k": 0,
        "sequence_length": sample_cfg.sequence_length,
        "bidirectional": sample_cfg.objective.bidirectional,
        "dsdna_canonicalize": dsdna_mode_for_opt,
        "score_scale": sample_cfg.objective.score_scale,
        "record_tune": sample_cfg.output.include_tune_in_sequences,
        "build_trace": bool(sample_cfg.output.save_trace),
        "progress_bar": bool(progress_bar),
        "progress_every": int(progress_every),
        "mcmc_cooling": mcmc_cooling,
        "early_stop": sample_cfg.optimizer.early_stop.model_dump(mode="json"),
        **moves.model_dump(),
        "softmin": {"enabled": softmin_cfg.enabled, **softmin_sched},
    }

    logger.debug("Optimizer config flattened: %s", opt_cfg)

    # 4) INSTANTIATE OPTIMIZER
    from dnadesign.cruncher.core.optimizers.registry import get_optimizer

    optimizer_factory = get_optimizer(optimizer_kind)
    run_logger("Instantiating optimizer: %s", optimizer_kind)
    rng = np.random.default_rng(sample_cfg.seed + set_index - 1)
    optimizer = optimizer_factory(
        evaluator=evaluator,
        cfg=opt_cfg,
        rng=rng,
        pwms=pwms,
        init_cfg=None,
        telemetry=telemetry,
        progress=progress,
    )

    # 5) RUN the MCMC sampling
    status_writer.update(status_message="sampling")
    run_logger("Starting MCMC sampling …")
    try:
        optimizer.optimise()
    except KeyboardInterrupt as exc:
        status_writer.finish(status="aborted", status_message="aborted", error=str(exc) or "KeyboardInterrupt")
        update_run_index_from_status(
            config_path,
            out_dir,
            status_writer.payload,
            catalog_root=cfg.catalog.catalog_root,
        )
        raise
    except Exception as exc:  # pragma: no cover - ensures run status is updated on failure
        status_writer.finish(status="failed", status_message="failed", error=str(exc))
        update_run_index_from_status(
            config_path,
            out_dir,
            status_writer.payload,
            catalog_root=cfg.catalog.catalog_root,
        )
        raise
    if status_writer is not None and hasattr(optimizer, "best_score"):
        best_meta = getattr(optimizer, "best_meta", None)
        status_writer.update(
            best_score=getattr(optimizer, "best_score", None),
            best_chain=(best_meta[0] + 1) if best_meta else None,
            best_draw=(best_meta[1]) if best_meta else None,
        )
    run_logger("MCMC sampling complete.")
    status_writer.update(status_message="sampling complete")
    status_writer.update(status_message="saving_artifacts")

    # 6) SAVE enriched config
    _save_config(cfg, out_dir, config_path, tfs=tfs, set_index=set_index, sample_cfg=sample_cfg, log_fn=run_logger)

    artifacts: list[dict[str, object]] = [
        artifact_entry(
            config_used_path(out_dir),
            out_dir,
            kind="config",
            label="Resolved config (config_used.yaml)",
            stage=stage,
        )
    ]

    # 7) SAVE trace.nc
    if sample_cfg.output.save_trace and hasattr(optimizer, "trace_idata") and optimizer.trace_idata is not None:
        from dnadesign.cruncher.artifacts.traces import save_trace

        save_trace(optimizer.trace_idata, trace_path(out_dir))
        artifacts.append(
            artifact_entry(
                trace_path(out_dir),
                out_dir,
                kind="trace",
                label="Trace (NetCDF)",
                stage=stage,
            )
        )
        logger.debug("Saved MCMC trace -> %s", trace_path(out_dir).relative_to(out_dir.parent))
    elif not sample_cfg.output.save_trace:
        logger.debug("Skipping trace.nc (sample.output.save_trace=false)")

    # Resolve final soft-min beta for elite ranking (from optimizer schedule)
    beta_softmin_final: float | None = _resolve_final_softmin_beta(optimizer, sample_cfg)

    # 8) SAVE sequences.parquet (chain, draw, phase, sequence_string, per-TF scaled scores)
    if sample_cfg.output.save_sequences:
        seq_parquet = sequences_path(out_dir)
        tf_order = sorted(tfs)
        want_canonical_all = bool(sample_cfg.objective.bidirectional)
        record_tune = bool(sample_cfg.output.include_tune_in_sequences)
        all_meta, all_samples, all_scores = _assert_sequence_buffers_aligned(optimizer)
        all_trace_meta = _assert_trace_meta_aligned(optimizer, expected_rows=len(all_meta))

        def _sequence_rows() -> Iterable[dict[str, object]]:
            for meta_row, trace_meta, seq_arr, per_tf_map in zip(all_meta, all_trace_meta, all_samples, all_scores):
                if not isinstance(meta_row, tuple) or len(meta_row) < 2:
                    raise RunError("Optimizer all_meta row must be a tuple with at least (chain, sweep_idx).")
                chain_id = int(meta_row[0])
                draw_i = int(meta_row[1])
                chain_raw = trace_meta.get("chain")
                beta_raw = trace_meta.get("beta")
                sweep_idx_raw = trace_meta.get("sweep_idx")
                phase_raw = trace_meta.get("phase")
                if beta_raw is None or sweep_idx_raw is None:
                    raise RunError("Optimizer trace metadata row is missing one of: beta, sweep_idx.")
                if chain_raw is not None and int(chain_raw) != chain_id:
                    raise RunError(
                        f"Optimizer trace metadata chain mismatch: chain={int(chain_raw)} all_meta.chain={chain_id}"
                    )
                beta = float(beta_raw)
                sweep_idx = int(sweep_idx_raw)
                phase = str(phase_raw or "")
                if phase not in {"tune", "draw"}:
                    raise RunError(f"Optimizer trace metadata row has invalid phase '{phase}'.")
                if sweep_idx != draw_i:
                    raise RunError(
                        f"Optimizer trace metadata sweep mismatch: sweep_idx={sweep_idx} all_meta.draw={draw_i}"
                    )
                if not record_tune and phase == "tune":
                    continue
                if phase == "tune":
                    draw_i_to_write = sweep_idx
                    draw_in_phase = sweep_idx
                else:
                    draw_i_to_write = sweep_idx
                    draw_in_phase = sweep_idx - adapt_sweeps if record_tune else sweep_idx
                seq_str = SequenceState(seq_arr).to_string()
                norm_map = _norm_map_for_elites(
                    seq_arr,
                    per_tf_map,
                    scorer=scorer,
                    score_scale=sample_cfg.objective.score_scale,
                )
                row: dict[str, object] = {
                    "chain": int(chain_id),
                    "chain_1based": int(chain_id) + 1,
                    "beta": float(beta),
                    "sweep_idx": int(sweep_idx),
                    "draw": int(draw_i_to_write),
                    "draw_in_phase": int(draw_in_phase),
                    "phase": phase,
                    "sequence": seq_str,
                }
                if want_canonical_all:
                    row["canonical_sequence"] = SequenceState(canon_int(seq_arr)).to_string()
                row["combined_score_final"] = float(
                    evaluator.combined_from_scores(per_tf_map, beta=beta_softmin_final, length=seq_arr.size)
                )
                min_norm = float(min(norm_map.values())) if norm_map else 0.0
                row["min_norm"] = min_norm
                for tf in tf_order:
                    row[f"score_{tf}"] = float(per_tf_map[tf])
                yield row

        _write_parquet_rows(seq_parquet, _sequence_rows())
        artifacts.append(
            artifact_entry(
                seq_parquet,
                out_dir,
                kind="table",
                label="Sequences with per-TF scores (Parquet)",
                stage=stage,
            )
        )
        logger.debug(
            "Saved all sequences with per-TF scores -> %s",
            seq_parquet.relative_to(out_dir.parent),
        )

    # 8b) SAVE random baseline cloud + hits (artifact-only analysis)
    baseline_seed = int(sample_cfg.seed + set_index - 1)
    baseline_n = 2048
    baseline_path = random_baseline_path(out_dir)
    baseline_hits_path = random_baseline_hits_path(out_dir)
    tf_order = sorted(tfs)
    baseline_canonical = bool(sample_cfg.objective.bidirectional)
    rng = np.random.default_rng(baseline_seed)
    baseline_rows: list[dict[str, object]] = []
    baseline_hit_rows: list[dict[str, object]] = []
    for baseline_id in range(baseline_n):
        seq_state = SequenceState.random(sample_cfg.sequence_length, rng)
        seq_arr = seq_state.seq
        seq_str = seq_state.to_string()
        per_tf = scorer.compute_all_per_pwm(seq_arr, sample_cfg.sequence_length)
        norm_map = _norm_map_for_elites(
            seq_arr,
            per_tf,
            scorer=scorer,
            score_scale=sample_cfg.objective.score_scale,
        )
        row: dict[str, object] = {
            "baseline_id": baseline_id,
            "sequence": seq_str,
            "baseline_seed": baseline_seed,
            "baseline_n": baseline_n,
            "seed": baseline_seed,
            "n_samples": baseline_n,
            "sequence_length": sample_cfg.sequence_length,
            "length": sample_cfg.sequence_length,
            "score_scale": sample_cfg.objective.score_scale,
            "bidirectional": bool(sample_cfg.objective.bidirectional),
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
            hit = scorer.best_hit(seq_arr, tf_name)
            pwm = pwms.get(tf_name)
            if pwm is None:
                raise ValueError(f"Missing PWM for TF '{tf_name}'.")
            pwm_width = int(pwm.length)
            width = hit.get("width")
            core_width = int(width) if isinstance(width, int) else pwm_width
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
                    "pwm_width": pwm_width,
                    "core_width": core_width,
                    "core_def_hash": core_def_by_tf.get(tf_name),
                }
            )
        baseline_rows.append(row)

    _write_parquet_rows(baseline_path, baseline_rows)
    _write_parquet_rows(baseline_hits_path, baseline_hit_rows, schema=_baseline_hits_parquet_schema())
    artifacts.append(
        artifact_entry(
            baseline_path,
            out_dir,
            kind="table",
            label="Random baseline (Parquet)",
            stage=stage,
        )
    )
    artifacts.append(
        artifact_entry(
            baseline_hits_path,
            out_dir,
            kind="table",
            label="Random baseline hits (Parquet)",
            stage=stage,
        )
    )
    logger.debug("Saved random baseline -> %s", baseline_path.relative_to(out_dir.parent))

    # 9) BUILD elites list — score-driven candidates, then MMR diversity selection
    pool_result = build_elite_pool(
        optimizer=optimizer,
        evaluator=evaluator,
        scorer=scorer,
        sample_cfg=sample_cfg,
        beta_softmin_final=beta_softmin_final,
    )
    raw_elites = pool_result.raw_elites
    norm_sums = pool_result.norm_sums
    min_norms = pool_result.min_norms
    total_draws_seen = pool_result.total_draws_seen

    # -------- percentile diagnostics ----------------------------------------
    if norm_sums:
        p50, p90 = np.percentile(norm_sums, [50, 90])
        n_tf = scorer.pwm_count
        logger.debug("Normalised-sum percentiles  |  median %.2f   90%% %.2f", p50, p90)
        logger.debug(
            "Typical draw: med %.2f (≈ %.0f%%/TF); top-10%% %.2f (≈ %.0f%%/TF)",
            p50,
            100 * p50 / n_tf if n_tf else 0.0,
            p90,
            100 * p90 / n_tf if n_tf else 0.0,
        )
    if min_norms:
        p50_min, p90_min = np.percentile(min_norms, [50, 90])
        logger.debug("Normalised-min percentiles |  median %.2f   90%% %.2f", p50_min, p90_min)

    elite_k = int(sample_cfg.elites.k or 0)
    if elite_k > 0 and not raw_elites:
        _finish_failed(
            RunError(
                "No elite candidates were generated from sampled draws. "
                "Increase cruncher.sample.sequence_length or cruncher.sample.budget.draws."
            )
        )

    dsdna_mode = bool(sample_cfg.objective.bidirectional)
    pool_size_cfg = sample_cfg.elites.select.pool_size
    pool_size = _resolve_elite_pool_size(
        pool_size_cfg=pool_size_cfg,
        elite_k=elite_k,
        candidate_count=len(raw_elites),
    )
    diversity_value = float(sample_cfg.elites.select.diversity)
    mmr_meta_rows: list[dict[str, object]] | None = None
    mmr_summary: dict[str, object] | None = None
    mmr_meta_path: Path | None = None

    # -------- select elites --------------------------------------------------
    selection_result = select_elites_mmr(
        raw_elites=raw_elites,
        elite_k=elite_k,
        pool_size=pool_size,
        scorer=scorer,
        pwms=pwms,
        dsdna_mode=dsdna_mode,
        diversity=diversity_value,
        sample_sequence_length=int(sample_cfg.sequence_length),
        cooling_config=sample_cfg.optimizer.cooling.model_dump(mode="json"),
    )
    kept_elites = selection_result.kept_elites
    kept_after_mmr = selection_result.kept_after_mmr
    mmr_meta_rows = selection_result.mmr_meta_rows
    mmr_summary = selection_result.mmr_summary
    if isinstance(mmr_summary, dict):
        summary_pool = mmr_summary.get("pool_size")
        if isinstance(summary_pool, (int, float)):
            pool_size = int(summary_pool)

    if elite_k > 0 and len(kept_elites) < elite_k:
        _finish_failed(
            RunError(
                f"Elite selection returned {len(kept_elites)} candidates, "
                f"fewer than cruncher.sample.elites.k={elite_k}. "
                "Increase cruncher.sample.sequence_length or cruncher.sample.budget.draws."
            )
        )

    # serialise elites
    want_cons = False
    want_canonical = bool(dsdna_mode)
    elites = build_elite_entries(
        kept_elites,
        scorer=scorer,
        sample_cfg=sample_cfg,
        want_consensus=want_cons,
        want_canonical=want_canonical,
        meta_source=out_dir.name,
    )
    dedupe_key = "canonical_sequence" if want_canonical else "sequence"
    seen_keys: set[str] = set()
    duplicate_keys: set[str] = set()
    for entry in elites:
        key_val = entry.get(dedupe_key)
        if not isinstance(key_val, str) or not key_val:
            _finish_failed(RunError(f"Elite entry missing required '{dedupe_key}' for uniqueness validation."))
        if key_val in seen_keys:
            duplicate_keys.add(key_val)
        seen_keys.add(key_val)
    if duplicate_keys:
        _finish_failed(
            RunError(
                "Elite selection produced duplicate sequences after MMR selection "
                f"(key={dedupe_key}, duplicates={len(duplicate_keys)})."
            )
        )

    run_logger("Final elite count: %d", len(elites))

    tf_label = format_regulator_slug(tfs)

    # 1)  parquet payload ----------------------------------------------------
    parquet_path = elites_path(out_dir)

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

    def _elite_hits_rows(entries: list[dict[str, object]]) -> Iterable[dict[str, object]]:
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

    elite_schema = _elite_parquet_schema(tfs, include_canonical=want_canonical)
    _write_parquet_rows(parquet_path, _elite_rows_from(elites), chunk_size=2000, schema=elite_schema)
    logger.debug("Saved elites Parquet -> %s", parquet_path.relative_to(out_dir.parent))

    hits_path = elites_hits_path(out_dir)
    hits_schema = _elite_hits_parquet_schema()
    _write_parquet_rows(hits_path, _elite_hits_rows(elites), chunk_size=2000, schema=hits_schema)
    logger.debug("Saved elites hits Parquet -> %s", hits_path.relative_to(out_dir.parent))

    json_path = elites_json_path(out_dir)
    atomic_write_json(json_path, elites)
    logger.debug("Saved elites JSON -> %s", json_path.relative_to(out_dir.parent))

    if mmr_meta_rows:
        mmr_meta_path = elites_mmr_meta_path(out_dir)
        _write_parquet_rows(mmr_meta_path, mmr_meta_rows, chunk_size=2000)
        logger.debug("Saved elites MMR meta -> %s", mmr_meta_path.relative_to(out_dir.parent))

    # 2)  .yaml run-metadata -----------------------------------------------
    optimizer_stats = optimizer.stats() if hasattr(optimizer, "stats") else {}
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
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_elites": len(elites),
        "selection_policy": selection_policy,
        "selection_relevance": relevance_label,
        "selection_score_weight": score_weight,
        "selection_diversity_weight": diversity_weight,
        "selection_diversity": diversity_value,
        "dsdna_canonicalize": want_canonical,
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
        #  you can inline the full cfg if you prefer – this keeps it concise
        "config_file": str(config_used_path(out_dir).resolve()),
        "regulator_set": {"index": set_index, "tfs": tfs},
    }
    if isinstance(optimizer_stats, dict) and optimizer_stats:

        def _float_list(value: object) -> list[float] | None:
            if not isinstance(value, (list, tuple)):
                return None
            return [float(item) for item in value]

        meta["beta_ladder_base"] = _float_list(optimizer_stats.get("beta_ladder_base"))
        meta["beta_ladder_final"] = _float_list(optimizer_stats.get("beta_ladder_final"))
        final_beta = optimizer_stats.get("final_mcmc_beta")
        meta["final_mcmc_beta"] = float(final_beta) if final_beta is not None else None
        cooling_payload = optimizer_stats.get("mcmc_cooling")
        if isinstance(cooling_payload, dict):
            meta["mcmc_cooling"] = cooling_payload
    if mmr_summary is not None:
        meta["mmr_summary"] = mmr_summary
    yaml_path = elites_yaml_path(out_dir)
    atomic_write_yaml(yaml_path, meta, sort_keys=False)
    logger.debug("Saved metadata -> %s", yaml_path.relative_to(out_dir.parent))

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
            artifact_entry(
                hits_path,
                out_dir,
                kind="table",
                label="Elite best-hit metadata (Parquet)",
                stage=stage,
            ),
        ]
    )
    if mmr_meta_path is not None:
        artifacts.append(
            artifact_entry(
                mmr_meta_path,
                out_dir,
                kind="table",
                label="Elite MMR selection metadata",
                stage=stage,
            )
        )

    # 10) RUN MANIFEST (for reporting + provenance)
    manifest_path_out = write_run_manifest_and_update(
        cfg=cfg,
        config_path=config_path,
        sample_cfg=sample_cfg,
        tfs=tfs,
        set_index=set_index,
        set_count=set_count,
        run_group=run_group,
        run_kind=run_kind,
        stage=stage,
        run_dir=out_dir,
        lockmap=lockmap,
        artifacts=artifacts,
        optimizer=optimizer,
        optimizer_kind=optimizer_kind,
        combine_resolved=combine_resolved,
        beta_softmin_final=beta_softmin_final,
        parse_signature=parse_signature,
        parse_inputs=parse_inputs,
        status_writer=status_writer,
    )
    logger.debug("Wrote run manifest -> %s", manifest_path_out.relative_to(out_dir.parent))
    return out_dir
