"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/run_set_execution.py

Execution-stage implementation for running a sample set and persisting artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dnadesign.cruncher.app.parse_signature import compute_parse_signature
from dnadesign.cruncher.app.progress import progress_adapter
from dnadesign.cruncher.app.run_service import update_run_index_from_status
from dnadesign.cruncher.app.sample.artifacts import (
    _save_config,
    _write_parquet_rows,
)
from dnadesign.cruncher.app.sample.diagnostics import _norm_map_for_elites
from dnadesign.cruncher.app.sample.elites_stage import select_and_persist_elites
from dnadesign.cruncher.app.sample.preflight import (
    ConfigError,
    RunError,
    _assert_init_length_fits_pwms,
    _assert_sequence_buffers_aligned,
    _assert_trace_meta_aligned,
    _core_def_hash,
    _mcmc_cooling_payload,
    _resolve_final_softmin_beta,
    _softmin_schedule_payload,
    _validate_objective_preflight,
    prepare_objective_plan,
)
from dnadesign.cruncher.app.sample.random_baseline import (
    write_random_baseline_artifacts as _write_random_baseline_artifacts,
)
from dnadesign.cruncher.app.sample.resources import _load_pwms_for_set
from dnadesign.cruncher.app.sample.run_layout import prepare_run_layout, write_run_manifest_and_update
from dnadesign.cruncher.app.telemetry import RunTelemetry
from dnadesign.cruncher.artifacts.entries import artifact_entry
from dnadesign.cruncher.artifacts.layout import (
    config_used_path,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.config.moves import resolve_move_config
from dnadesign.cruncher.config.schema_v3 import CruncherConfig, SampleConfig
from dnadesign.cruncher.core.evaluator import SequenceEvaluator
from dnadesign.cruncher.core.objectives.compiler import ObjectivePlanCompilation
from dnadesign.cruncher.core.optimizers.kinds import resolve_optimizer_kind
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.sequence import canon_int
from dnadesign.cruncher.core.state import SequenceState
from dnadesign.cruncher.store.lockfile import read_lockfile
from dnadesign.cruncher.utils.paths import resolve_lock_path, resolve_workspace_root

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RunExecutionContext:
    out_dir: Path
    run_group: str
    stage_label: str
    status_writer: object
    telemetry: RunTelemetry
    progress: object
    run_logger: Callable[..., None]


@dataclass(frozen=True)
class _SamplingPreparation:
    pwms: dict[str, object]
    pwm_ref_by_tf: dict[str, str | None]
    pwm_hash_by_tf: dict[str, str | None]
    core_def_by_tf: dict[str, str]
    objective_plan: ObjectivePlanCompilation
    parse_signature: str
    parse_inputs: dict[str, object]
    scorer: Scorer
    evaluator: SequenceEvaluator
    combine_resolved: str
    adapt_sweeps: int
    optimizer: object


def _sum_combine(values: Iterable[float]) -> float:
    return float(sum(values))


def _build_pwm_provenance(
    *,
    tfs: list[str],
    pwms: dict[str, object],
    lockmap: dict[str, object],
) -> tuple[dict[str, str | None], dict[str, str | None], dict[str, str]]:
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
    return pwm_ref_by_tf, pwm_hash_by_tf, core_def_by_tf


def _resolve_combiner(
    *,
    combine_cfg: str,
    scale: str,
) -> tuple[Callable[[Iterable[float]], float] | None, str]:
    if combine_cfg == "sum":
        return _sum_combine, "sum"
    if combine_cfg == "min":
        if scale == "consensus-neglop-sum":
            return min, "min"
        return None, "min"
    raise ConfigError(f"Unsupported cruncher.sample.objective.combine value '{combine_cfg}'. Expected 'min' or 'sum'.")


def _build_scorer_and_evaluator(
    *,
    pwms: dict[str, object],
    sample_cfg: SampleConfig,
    tfs: list[str],
    objective_plan: ObjectivePlanCompilation,
) -> tuple[Scorer, SequenceEvaluator, str]:
    scale = sample_cfg.objective.score_scale
    combine_cfg = sample_cfg.objective.combine
    logger.debug("Using score_scale = %r", scale)
    _validate_objective_preflight(sample_cfg, n_tfs=len(tfs))
    combiner, combine_resolved = _resolve_combiner(combine_cfg=combine_cfg, scale=scale)
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
        objective_specs=objective_plan.objectives,
    )
    logger.debug("Scorer and SequenceEvaluator instantiated")
    logger.debug("  Scorer.scale = %r", scorer.scale)
    logger.debug("  SequenceEvaluator.scale = %r", evaluator._scale)
    logger.debug("  SequenceEvaluator.combiner = %r", evaluator._combiner)
    return scorer, evaluator, combine_resolved


def _build_optimizer_cfg(
    *,
    sample_cfg: SampleConfig,
    objective_plan: ObjectivePlanCompilation,
    chain_count: int,
    draws: int,
    adapt_sweeps: int,
    progress_bar: bool,
    progress_every: int,
) -> dict[str, object]:
    moves = resolve_move_config(sample_cfg.moves)
    softmin_cfg = sample_cfg.objective.softmin
    softmin_sched = _softmin_schedule_payload(sample_cfg)
    mcmc_cooling = _mcmc_cooling_payload(sample_cfg)
    return {
        "draws": draws,
        "tune": adapt_sweeps,
        "chains": chain_count,
        "min_dist": 0,
        "top_k": 0,
        "sequence_length": sample_cfg.sequence_length,
        "bidirectional": sample_cfg.objective.bidirectional,
        "dsdna_canonicalize": bool(sample_cfg.objective.bidirectional),
        "score_scale": sample_cfg.objective.score_scale,
        "record_tune": sample_cfg.output.include_tune_in_sequences,
        "build_trace": bool(sample_cfg.output.save_trace),
        "progress_bar": bool(progress_bar),
        "progress_every": int(progress_every),
        "mcmc_cooling": mcmc_cooling,
        "enable_incremental_rescore": bool(objective_plan.runtime.supports_incremental_rescore),
        "early_stop": sample_cfg.optimizer.early_stop.model_dump(mode="json"),
        **{
            **moves.model_dump(),
            "target_worst_tf_prob": (
                float(moves.target_worst_tf_prob) if objective_plan.runtime.supports_targeted_window_hint else 0.0
            ),
        },
        "softmin": {"enabled": softmin_cfg.enabled, **softmin_sched},
    }


def _instantiate_optimizer(
    *,
    optimizer_kind: str,
    evaluator: SequenceEvaluator,
    opt_cfg: dict[str, object],
    sample_cfg: SampleConfig,
    set_index: int,
    pwms: dict[str, object],
    telemetry: RunTelemetry,
    progress: object,
    run_logger: Callable[..., None],
) -> object:
    from dnadesign.cruncher.core.optimizers.registry import get_optimizer

    optimizer_factory = get_optimizer(optimizer_kind)
    run_logger("Instantiating optimizer: %s", optimizer_kind)
    rng = np.random.default_rng(sample_cfg.seed + set_index - 1)
    return optimizer_factory(
        evaluator=evaluator,
        cfg=opt_cfg,
        rng=rng,
        pwms=pwms,
        init_cfg=None,
        telemetry=telemetry,
        progress=progress,
    )


def _update_run_index_if_enabled(
    *,
    register_run_in_index: bool,
    config_path: Path,
    out_dir: Path,
    status_payload: dict[str, object],
    catalog_root: Path,
) -> None:
    if not register_run_in_index:
        return
    update_run_index_from_status(
        config_path,
        out_dir,
        status_payload,
        catalog_root=catalog_root,
    )


def _run_optimizer(
    *,
    optimizer: object,
    status_writer: object,
    run_logger: Callable[..., None],
    register_run_in_index: bool,
    config_path: Path,
    out_dir: Path,
    catalog_root: Path,
) -> None:
    status_writer.update(status_message="sampling")
    run_logger("Starting MCMC sampling …")
    try:
        optimizer.optimise()
    except KeyboardInterrupt as exc:
        status_writer.finish(status="aborted", status_message="aborted", error=str(exc) or "KeyboardInterrupt")
        _update_run_index_if_enabled(
            register_run_in_index=register_run_in_index,
            config_path=config_path,
            out_dir=out_dir,
            status_payload=status_writer.payload,
            catalog_root=catalog_root,
        )
        raise
    except Exception as exc:  # pragma: no cover - ensures run status is updated on failure
        status_writer.finish(status="failed", status_message="failed", error=str(exc))
        _update_run_index_if_enabled(
            register_run_in_index=register_run_in_index,
            config_path=config_path,
            out_dir=out_dir,
            status_payload=status_writer.payload,
            catalog_root=catalog_root,
        )
        raise

    if hasattr(optimizer, "best_score"):
        best_meta = getattr(optimizer, "best_meta", None)
        status_writer.update(
            best_score=getattr(optimizer, "best_score", None),
            best_chain=(best_meta[0] + 1) if best_meta else None,
            best_draw=(best_meta[1]) if best_meta else None,
        )
    run_logger("MCMC sampling complete.")
    status_writer.update(status_message="sampling complete")
    status_writer.update(status_message="saving_artifacts")


def _init_artifacts(*, out_dir: Path, stage: str) -> list[dict[str, object]]:
    return [
        artifact_entry(
            config_used_path(out_dir),
            out_dir,
            kind="config",
            label="Resolved config (config_used.yaml)",
            stage=stage,
        )
    ]


def _save_trace_artifact(
    *,
    sample_cfg: SampleConfig,
    optimizer: object,
    out_dir: Path,
    stage: str,
    artifacts: list[dict[str, object]],
) -> None:
    if not sample_cfg.output.save_trace:
        logger.debug("Skipping trace.nc (sample.output.save_trace=false)")
        return
    trace_idata = getattr(optimizer, "trace_idata", None)
    if trace_idata is None:
        return
    from dnadesign.cruncher.artifacts.traces import save_trace

    save_trace(trace_idata, trace_path(out_dir))
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


def _build_sequence_row(
    *,
    meta_row: object,
    trace_meta: dict[str, object],
    seq_arr: np.ndarray,
    per_tf_map: dict[str, float],
    record_tune: bool,
    adapt_sweeps: int,
    tf_order: list[str],
    want_canonical_all: bool,
    beta_softmin_final: float | None,
    scorer: Scorer,
    evaluator: SequenceEvaluator,
    sample_cfg: SampleConfig,
) -> dict[str, object] | None:
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
        raise RunError(f"Optimizer trace metadata chain mismatch: chain={int(chain_raw)} all_meta.chain={chain_id}")
    beta = float(beta_raw)
    sweep_idx = int(sweep_idx_raw)
    phase = str(phase_raw or "")
    if phase not in {"tune", "draw"}:
        raise RunError(f"Optimizer trace metadata row has invalid phase '{phase}'.")
    if sweep_idx != draw_i:
        raise RunError(f"Optimizer trace metadata sweep mismatch: sweep_idx={sweep_idx} all_meta.draw={draw_i}")
    if not record_tune and phase == "tune":
        return None
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
        evaluator.combined_from_scores(
            per_tf_map,
            beta=beta_softmin_final,
            length=seq_arr.size,
        )
    )
    row["min_norm"] = float(min(norm_map.values())) if norm_map else 0.0
    for tf_name in tf_order:
        row[f"score_{tf_name}"] = float(per_tf_map[tf_name])
    return row


def _iter_sequence_rows(
    *,
    all_meta: list[object],
    all_trace_meta: list[dict[str, object]],
    all_samples: list[object],
    all_scores: list[object],
    record_tune: bool,
    adapt_sweeps: int,
    tf_order: list[str],
    want_canonical_all: bool,
    beta_softmin_final: float | None,
    scorer: Scorer,
    evaluator: SequenceEvaluator,
    sample_cfg: SampleConfig,
) -> Iterable[dict[str, object]]:
    for meta_row, trace_meta, seq_arr, per_tf_map in zip(all_meta, all_trace_meta, all_samples, all_scores):
        if not isinstance(trace_meta, dict):
            raise RunError("Optimizer trace metadata rows must be dictionaries.")
        if not isinstance(seq_arr, np.ndarray):
            raise RunError("Optimizer all_samples row must be a numpy sequence array.")
        if not isinstance(per_tf_map, dict):
            raise RunError("Optimizer all_scores row must be a per-TF score mapping.")
        row = _build_sequence_row(
            meta_row=meta_row,
            trace_meta=trace_meta,
            seq_arr=seq_arr,
            per_tf_map=per_tf_map,
            record_tune=record_tune,
            adapt_sweeps=adapt_sweeps,
            tf_order=tf_order,
            want_canonical_all=want_canonical_all,
            beta_softmin_final=beta_softmin_final,
            scorer=scorer,
            evaluator=evaluator,
            sample_cfg=sample_cfg,
        )
        if row is not None:
            yield row


def _write_sequences_artifact(
    *,
    optimizer: object,
    out_dir: Path,
    tfs: list[str],
    sample_cfg: SampleConfig,
    adapt_sweeps: int,
    beta_softmin_final: float | None,
    scorer: Scorer,
    evaluator: SequenceEvaluator,
    stage: str,
    artifacts: list[dict[str, object]],
) -> None:
    if not sample_cfg.output.save_sequences:
        return
    seq_parquet = sequences_path(out_dir)
    tf_order = sorted(tfs)
    want_canonical_all = bool(sample_cfg.objective.bidirectional)
    record_tune = bool(sample_cfg.output.include_tune_in_sequences)
    all_meta, all_samples, all_scores = _assert_sequence_buffers_aligned(optimizer)
    all_trace_meta = _assert_trace_meta_aligned(optimizer, expected_rows=len(all_meta))
    _write_parquet_rows(
        seq_parquet,
        _iter_sequence_rows(
            all_meta=all_meta,
            all_trace_meta=all_trace_meta,
            all_samples=all_samples,
            all_scores=all_scores,
            record_tune=record_tune,
            adapt_sweeps=adapt_sweeps,
            tf_order=tf_order,
            want_canonical_all=want_canonical_all,
            beta_softmin_final=beta_softmin_final,
            scorer=scorer,
            evaluator=evaluator,
            sample_cfg=sample_cfg,
        ),
    )
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


def _finish_failed_and_raise(
    *,
    exc: Exception,
    status_writer: object,
    register_run_in_index: bool,
    config_path: Path,
    out_dir: Path,
    catalog_root: Path,
) -> None:
    status_writer.finish(status="failed", status_message="failed", error=str(exc))
    _update_run_index_if_enabled(
        register_run_in_index=register_run_in_index,
        config_path=config_path,
        out_dir=out_dir,
        status_payload=status_writer.payload,
        catalog_root=catalog_root,
    )
    raise exc


def _prepare_run_execution_context(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    sample_cfg: SampleConfig,
    tfs: list[str],
    set_index: int,
    set_count: int,
    include_set_index: bool,
    stage: str,
    run_kind: str | None,
    chain_count: int,
    optimizer_kind: str,
    force_overwrite: bool,
    register_run_in_index: bool,
    progress_bar: bool,
) -> _RunExecutionContext:
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
        register_run_in_index=register_run_in_index,
    )
    run_logger = logger.info
    run_logger("=== RUN %s: %s ===", layout.stage_label, layout.run_dir)
    logger.debug("Full sample config: %s", sample_cfg.model_dump_json())
    status_writer = layout.status_writer
    return _RunExecutionContext(
        out_dir=layout.run_dir,
        run_group=layout.run_group,
        stage_label=layout.stage_label,
        status_writer=status_writer,
        telemetry=RunTelemetry(status_writer),
        progress=progress_adapter(progress_bar),
        run_logger=run_logger,
    )


def _prepare_sampling(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    sample_cfg: SampleConfig,
    tfs: list[str],
    lockmap: dict[str, object],
    set_index: int,
    chain_count: int,
    optimizer_kind: str,
    progress_bar: bool,
    progress_every: int,
    load_pwms_for_set: Callable[..., dict[str, object]],
    telemetry: RunTelemetry,
    progress: object,
    run_logger: Callable[..., None],
) -> _SamplingPreparation:
    pwms = load_pwms_for_set(cfg=cfg, config_path=config_path, tfs=tfs, lockmap=lockmap)
    _assert_init_length_fits_pwms(sample_cfg, pwms)
    objective_plan = prepare_objective_plan(sample_cfg=sample_cfg, tfs=tfs, pwms=pwms)
    pwm_ref_by_tf, pwm_hash_by_tf, core_def_by_tf = _build_pwm_provenance(
        tfs=tfs,
        pwms=pwms,
        lockmap=lockmap,
    )
    lockfile = read_lockfile(resolve_lock_path(config_path))
    parse_signature, parse_inputs = compute_parse_signature(
        cfg=cfg,
        lockfile=lockfile,
        tfs=tfs,
    )
    adapt_sweeps = int(sample_cfg.budget.tune)
    draws = int(sample_cfg.budget.draws)
    scorer, evaluator, combine_resolved = _build_scorer_and_evaluator(
        pwms=pwms,
        sample_cfg=sample_cfg,
        tfs=tfs,
        objective_plan=objective_plan,
    )
    opt_cfg = _build_optimizer_cfg(
        sample_cfg=sample_cfg,
        objective_plan=objective_plan,
        chain_count=chain_count,
        draws=draws,
        adapt_sweeps=adapt_sweeps,
        progress_bar=progress_bar,
        progress_every=progress_every,
    )
    logger.debug("Optimizer config flattened: %s", opt_cfg)
    optimizer = _instantiate_optimizer(
        optimizer_kind=optimizer_kind,
        evaluator=evaluator,
        opt_cfg=opt_cfg,
        sample_cfg=sample_cfg,
        set_index=set_index,
        pwms=pwms,
        telemetry=telemetry,
        progress=progress,
        run_logger=run_logger,
    )
    return _SamplingPreparation(
        pwms=pwms,
        pwm_ref_by_tf=pwm_ref_by_tf,
        pwm_hash_by_tf=pwm_hash_by_tf,
        core_def_by_tf=core_def_by_tf,
        objective_plan=objective_plan,
        parse_signature=parse_signature,
        parse_inputs=parse_inputs,
        scorer=scorer,
        evaluator=evaluator,
        combine_resolved=combine_resolved,
        adapt_sweeps=adapt_sweeps,
        optimizer=optimizer,
    )


def _persist_run_outputs(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    sample_cfg: SampleConfig,
    tfs: list[str],
    set_index: int,
    set_count: int,
    run_group: str,
    run_kind: str | None,
    stage: str,
    out_dir: Path,
    lockmap: dict[str, object],
    optimizer_kind: str,
    preparation: _SamplingPreparation,
    status_writer: object,
    run_logger: Callable[..., None],
    register_run_in_index: bool,
    save_config: Callable[..., None],
) -> None:
    save_config(cfg, out_dir, config_path, tfs=tfs, set_index=set_index, sample_cfg=sample_cfg, log_fn=run_logger)
    artifacts = _init_artifacts(out_dir=out_dir, stage=stage)
    _save_trace_artifact(
        sample_cfg=sample_cfg,
        optimizer=preparation.optimizer,
        out_dir=out_dir,
        stage=stage,
        artifacts=artifacts,
    )
    beta_softmin_final: float | None = _resolve_final_softmin_beta(preparation.optimizer, sample_cfg)
    _write_sequences_artifact(
        optimizer=preparation.optimizer,
        out_dir=out_dir,
        tfs=tfs,
        sample_cfg=sample_cfg,
        adapt_sweeps=preparation.adapt_sweeps,
        beta_softmin_final=beta_softmin_final,
        scorer=preparation.scorer,
        evaluator=preparation.evaluator,
        stage=stage,
        artifacts=artifacts,
    )
    _write_random_baseline_artifacts(
        out_dir=out_dir,
        sample_cfg=sample_cfg,
        objective_plan=preparation.objective_plan,
        set_index=set_index,
        tfs=tfs,
        scorer=preparation.scorer,
        evaluator=preparation.evaluator,
        pwms=preparation.pwms,
        pwm_ref_by_tf=preparation.pwm_ref_by_tf,
        pwm_hash_by_tf=preparation.pwm_hash_by_tf,
        core_def_by_tf=preparation.core_def_by_tf,
        stage=stage,
        artifacts=artifacts,
    )

    def _finish_failed(exc: Exception) -> None:
        _finish_failed_and_raise(
            exc=exc,
            status_writer=status_writer,
            register_run_in_index=register_run_in_index,
            config_path=config_path,
            out_dir=out_dir,
            catalog_root=cfg.catalog.catalog_root,
        )

    select_and_persist_elites(
        optimizer=preparation.optimizer,
        evaluator=preparation.evaluator,
        objective_plan=preparation.objective_plan,
        scorer=preparation.scorer,
        sample_cfg=sample_cfg,
        pwms=preparation.pwms,
        tfs=tfs,
        out_dir=out_dir,
        workspace_slug=resolve_workspace_root(config_path).name,
        pwm_ref_by_tf=preparation.pwm_ref_by_tf,
        pwm_hash_by_tf=preparation.pwm_hash_by_tf,
        core_def_by_tf=preparation.core_def_by_tf,
        beta_softmin_final=beta_softmin_final,
        combine_resolved=preparation.combine_resolved,
        stage=stage,
        status_writer=status_writer,
        run_logger=run_logger,
        artifacts=artifacts,
        finish_failed=_finish_failed,
    )
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
        optimizer=preparation.optimizer,
        optimizer_kind=optimizer_kind,
        combine_resolved=preparation.combine_resolved,
        beta_softmin_final=beta_softmin_final,
        objective_plan=preparation.objective_plan,
        parse_signature=preparation.parse_signature,
        parse_inputs=preparation.parse_inputs,
        status_writer=status_writer,
        register_run_in_index=register_run_in_index,
    )
    logger.debug("Wrote run manifest -> %s", manifest_path_out.relative_to(out_dir.parent))


def run_sample_for_set(
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
    register_run_in_index: bool = True,
    load_pwms_for_set: Callable[..., dict[str, object]] = _load_pwms_for_set,
    save_config: Callable[..., None] = _save_config,
) -> Path:
    """
    Run MCMC sampler, save config/meta plus artifacts (trace.nc, sequences.parquet, elites.*).
    Each chain gets its own independent seed (random/consensus/consensus_mix).
    """
    optimizer_kind = resolve_optimizer_kind(sample_cfg.optimizer.kind, context="sample.optimizer.kind")
    chain_count = int(sample_cfg.optimizer.chains)
    runtime = _prepare_run_execution_context(
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
        register_run_in_index=register_run_in_index,
        progress_bar=progress_bar,
    )
    preparation = _prepare_sampling(
        cfg=cfg,
        config_path=config_path,
        sample_cfg=sample_cfg,
        tfs=tfs,
        lockmap=lockmap,
        set_index=set_index,
        chain_count=chain_count,
        optimizer_kind=optimizer_kind,
        progress_bar=progress_bar,
        progress_every=progress_every,
        load_pwms_for_set=load_pwms_for_set,
        telemetry=runtime.telemetry,
        progress=runtime.progress,
        run_logger=runtime.run_logger,
    )
    _run_optimizer(
        optimizer=preparation.optimizer,
        status_writer=runtime.status_writer,
        run_logger=runtime.run_logger,
        register_run_in_index=register_run_in_index,
        config_path=config_path,
        out_dir=runtime.out_dir,
        catalog_root=cfg.catalog.catalog_root,
    )
    _persist_run_outputs(
        cfg=cfg,
        config_path=config_path,
        sample_cfg=sample_cfg,
        tfs=tfs,
        set_index=set_index,
        set_count=set_count,
        run_group=runtime.run_group,
        run_kind=run_kind,
        stage=stage,
        out_dir=runtime.out_dir,
        lockmap=lockmap,
        optimizer_kind=optimizer_kind,
        preparation=preparation,
        status_writer=runtime.status_writer,
        run_logger=runtime.run_logger,
        register_run_in_index=register_run_in_index,
        save_config=save_config,
    )
    return runtime.out_dir
