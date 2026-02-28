"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/orchestrator.py

DenseGen pipeline orchestration (CLI-agnostic).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from ...adapters.optimizer import OptimizerAdapter
from ...adapters.outputs import resolve_bio_alphabet
from ...adapters.outputs.usr_writer import USRWriter
from ...config import (
    DenseGenConfig,
    LoadedConfig,
    ResolvedPlanItem,
    resolve_outputs_scoped_path,
    resolve_run_root,
)
from ...utils.logging_utils import install_native_stderr_filters
from ..artifacts.pool import POOL_MODE_TFBS, PoolData
from ..motif_labels import motif_display_name
from ..run_paths import (
    candidates_root,
    ensure_run_meta_dir,
    has_existing_run_outputs,
    run_outputs_root,
    run_tables_root,
)
from ..seeding import derive_seed_map
from .attempts import _load_existing_library_index
from .deps import PipelineDeps, default_deps
from .library_artifacts import prepare_library_source
from .outputs import _assert_sink_alignment, _write_effective_config
from .plan_context import PlanExecutionState, PlanRunContext
from .plan_execution import run_plan_schedule
from .plan_pools import PLAN_POOL_INPUT_TYPE, PlanPoolSpec
from .progress_runtime import _build_shared_dashboard
from .resume_state import load_resume_state
from .run_finalization import finalize_run_outputs
from .run_setup import build_display_map_by_input, init_plan_stats, init_state_counts, validate_resume_outputs
from .run_state_manager import init_run_state, reconcile_run_state_with_outputs, write_run_state
from .stage_a_pools import prepare_stage_a_pools
from .stage_b_plan_setup import _init_plan_settings, _load_plan_pool
from .stage_b_runtime_runner import _run_stage_b_sampling
from .versioning import _resolve_dense_arrays_version

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    total_generated: int
    per_plan: dict[tuple[str, str], int]
    generated_this_run: int = 0


def _pending_usr_overlay_backlog_parts(dataset_dir: Path) -> list[Path]:
    backlog_root = dataset_dir / "_artifacts" / "pending_overlay"
    if not backlog_root.exists():
        return []
    return sorted(backlog_root.glob("part-*.parquet"))


def _replay_usr_overlay_backlog_for_resume(
    *,
    cfg: DenseGenConfig,
    cfg_path: Path,
    run_root: Path,
) -> int:
    output_cfg = cfg.output
    usr_cfg = output_cfg.usr
    if "usr" not in output_cfg.targets or usr_cfg is None:
        return 0
    dataset_name = str(usr_cfg.dataset).strip()
    if not dataset_name:
        raise RuntimeError("output.usr.dataset must be a non-empty string.")

    usr_root = resolve_outputs_scoped_path(cfg_path, run_root, usr_cfg.root, label="output.usr.root")
    dataset_dir = (usr_root / dataset_name).resolve()
    pending_before = _pending_usr_overlay_backlog_parts(dataset_dir)
    if not pending_before:
        return 0

    records_path = dataset_dir / "records.parquet"
    if not records_path.exists():
        raise RuntimeError(
            f"USR overlay backlog exists but records.parquet is missing at {records_path}. "
            "Restore the dataset or reset the workspace before resuming."
        )

    log.warning(
        "Found %d pending USR overlay backlog part(s); replaying before resume-state scan.",
        len(pending_before),
    )
    writer = USRWriter(
        dataset=dataset_name,
        root=usr_root,
        namespace="densegen",
        chunk_size=max(1, int(getattr(usr_cfg, "chunk_size", 1) or 1)),
        run_id=str(cfg.run.id),
    )
    writer.flush()
    pending_after = _pending_usr_overlay_backlog_parts(dataset_dir)
    if pending_after:
        raise RuntimeError(
            "USR overlay backlog replay failed; "
            f"{len(pending_after)} part(s) still pending under {dataset_dir / '_artifacts' / 'pending_overlay'}."
        )
    return len(pending_before)


def _candidate_logging_enabled(cfg: DenseGenConfig) -> bool:
    for inp in cfg.inputs:
        sampling = getattr(inp, "sampling", None)
        if sampling is None:
            continue
        if getattr(sampling, "keep_all_candidates_debug", False):
            return True
    return False


def _plan_pool_input_meta(spec: PlanPoolSpec) -> dict:
    meta = {
        "input_type": PLAN_POOL_INPUT_TYPE,
        "input_name": spec.pool_name,
        "input_source_names": list(spec.include_inputs),
        "input_mode": "plan_pool",
    }
    if spec.pool.pool_mode == POOL_MODE_TFBS:
        if spec.pool.df is not None and "tf" in spec.pool.df.columns:
            meta["input_pwm_ids"] = sorted(set(spec.pool.df["tf"].tolist()))
        else:
            meta["input_pwm_ids"] = []
    else:
        meta["input_pwm_ids"] = []
    return meta


def resolve_plan(loaded: LoadedConfig) -> List[ResolvedPlanItem]:
    return loaded.root.densegen.generation.resolve_plan()


def select_solver(
    preferred: str | None,
    optimizer: OptimizerAdapter,
    *,
    strategy: str,
    test_length: int = 10,
) -> str | None:
    """Probe the requested solver once and fail fast if unavailable."""
    if strategy == "approximate":
        return preferred
    if not preferred:
        raise ValueError("solver.backend is required unless strategy=approximate")
    try:
        optimizer.probe_solver(preferred, test_length=test_length)
        return preferred
    except Exception as exc:
        raise RuntimeError(
            f"Requested solver '{preferred}' failed during probe: {exc}\n"
            "Please install/configure this solver or choose another in solver.backend."
        ) from exc


def _process_plan_for_source(
    source_cfg,
    plan_item: ResolvedPlanItem,
    context: PlanRunContext,
    execution_state: PlanExecutionState,
    plan_started_at: float | None = None,
    *,
    one_subsample_only: bool = False,
    already_generated: int = 0,
) -> tuple[int, dict]:
    deps = context.deps
    np_rng = context.np_rng
    cfg_path = context.cfg_path
    run_id = context.run_id
    run_root = context.run_root

    inputs_manifest = execution_state.inputs_manifest
    source_cache = execution_state.source_cache
    pool_override = execution_state.pool_override
    input_meta_override = execution_state.input_meta_override
    attempt_counters = execution_state.attempt_counters
    display_map_by_input = execution_state.display_map_by_input

    source_label = source_cfg.name
    plan_name = plan_item.name
    attempt_counters = attempt_counters or {}
    display_map = display_map_by_input.get(source_label, {}) if display_map_by_input else {}

    def _display_tf_label(label: str) -> str:
        if not label:
            return label
        if label in {"neutral_bg", "neutral"}:
            return "background"
        if label in display_map:
            return display_map[label]
        return motif_display_name(label, None)

    def _next_attempt_index() -> int:
        key = (source_label, plan_name)
        current = int(attempt_counters.get(key, 0)) + 1
        attempt_counters[key] = current
        return current

    settings = _init_plan_settings(
        source_cfg=source_cfg,
        plan_item=plan_item,
        context=context,
        execution_state=execution_state,
        one_subsample_only=one_subsample_only,
        plan_started_at=plan_started_at,
    )
    settings.progress.reporter.display_tf_label = _display_tf_label
    progress_style = settings.progress.progress_style

    run_root_path = Path(run_root)
    outputs_root = run_outputs_root(run_root_path)
    tables_root = run_tables_root(run_root_path)
    existing_library_builds = _load_existing_library_index(tables_root)

    input_state = _load_plan_pool(
        source_cfg=source_cfg,
        cfg_path=cfg_path,
        deps=deps,
        np_rng=np_rng,
        outputs_root=outputs_root,
        run_id=str(run_id),
        pool_override=pool_override,
        source_cache=source_cache,
        input_meta_override=input_meta_override,
        inputs_manifest=inputs_manifest,
        source_label=source_label,
        plan_name=plan_name,
        progress_style=progress_style,
        display_tf_label=_display_tf_label,
    )

    return _run_stage_b_sampling(
        settings=settings,
        input_state=input_state,
        plan_item=plan_item,
        context=context,
        execution_state=execution_state,
        existing_library_builds=existing_library_builds,
        already_generated=already_generated,
        one_subsample_only=one_subsample_only,
        display_tf_label=_display_tf_label,
        next_attempt_index=_next_attempt_index,
    )


def run_pipeline(
    loaded: LoadedConfig,
    *,
    resume: bool,
    build_stage_a: bool = False,
    show_tfbs: bool = False,
    show_solutions: bool = False,
    allow_config_mismatch: bool = False,
    deps: PipelineDeps | None = None,
) -> RunSummary:
    deps = deps or default_deps()
    cfg = loaded.root.densegen
    install_native_stderr_filters(suppress_solver_messages=bool(cfg.logging.suppress_solver_stderr))
    run_root = resolve_run_root(loaded.path, cfg.run.root)
    run_root_str = str(run_root)
    config_sha = hashlib.sha256(loaded.path.read_bytes()).hexdigest()
    try:
        run_cfg_path = str(loaded.path.relative_to(run_root))
    except ValueError:
        run_cfg_path = str(loaded.path)

    outputs_root = run_outputs_root(run_root)
    tables_root = run_tables_root(run_root)
    existing_outputs = has_existing_run_outputs(run_root)
    validate_resume_outputs(
        resume=resume,
        existing_outputs=existing_outputs,
        outputs_root=outputs_root,
        run_root=run_root,
    )

    # Seed
    seed = int(cfg.runtime.random_seed)
    seeds = derive_seed_map(seed, ["stage_a", "stage_b", "solver"])
    rng = random.Random(seeds["stage_b"])
    np_rng_stage_a = np.random.default_rng(seeds["stage_a"])
    np_rng_stage_b = np.random.default_rng(seeds["stage_b"])

    # Plan & solver
    pl = cfg.generation.resolve_plan()
    chosen_solver = select_solver(
        cfg.solver.backend,
        deps.optimizer,
        strategy=str(cfg.solver.strategy),
    )
    solver_attempt_timeout_seconds = (
        float(cfg.solver.solver_attempt_timeout_seconds)
        if cfg.solver.solver_attempt_timeout_seconds is not None
        else None
    )
    solver_threads = int(cfg.solver.threads) if cfg.solver.threads is not None else None
    dense_arrays_version, dense_arrays_version_source = _resolve_dense_arrays_version(loaded.path)

    # Build sinks
    sinks = list(deps.sink_factory(cfg, loaded.path))
    _assert_sink_alignment(sinks)
    output_bio_type, output_alphabet = resolve_bio_alphabet(cfg)

    total = 0
    per_plan: dict[tuple[str, str], int] = {}
    plan_stats: dict[tuple[str, str], dict[str, int]] = {}
    plan_order: list[tuple[str, str]] = []
    plan_leaderboards: dict[tuple[str, str], dict] = {}
    inputs_manifest_entries: dict[str, dict] = {}
    source_cache: dict[str, PoolData] = {}
    library_build_rows: list[dict] = []
    library_member_rows: list[dict] = []
    solution_rows: list[dict] = []
    composition_rows: list[dict] = []
    outputs_root.mkdir(parents=True, exist_ok=True)
    candidates_dir = candidates_root(outputs_root, cfg.run.id)
    candidate_logging = _candidate_logging_enabled(cfg)
    events_path = outputs_root / "meta" / "events.jsonl"
    try:
        _write_effective_config(
            cfg=cfg, cfg_path=loaded.path, run_root=run_root, seeds=seeds, outputs_root=outputs_root
        )
    except Exception as exc:
        raise RuntimeError("Failed to write effective_config.json.") from exc
    stage_a_state = prepare_stage_a_pools(
        cfg=cfg,
        cfg_path=loaded.path,
        run_root=run_root,
        outputs_root=outputs_root,
        rng=np_rng_stage_a,
        build_stage_a=build_stage_a,
        candidate_logging=candidate_logging,
        candidates_dir=candidates_dir,
        plan_items=pl,
        events_path=events_path,
        run_id=str(cfg.run.id),
        deps=deps,
    )
    pool_data = stage_a_state.pool_data
    plan_pools = stage_a_state.plan_pools
    plan_pool_sources = stage_a_state.plan_pool_sources
    source_cache.update(stage_a_state.source_cache)

    if resume and pool_data is None:
        raise RuntimeError(
            "resume=True requires existing Stage-A pools. "
            "Run `uv run dense stage-a build-pool` first or rerun without resume."
        )
    sampling_cfg = cfg.generation.sampling
    library_state = prepare_library_source(
        sampling_cfg=sampling_cfg,
        cfg_path=loaded.path,
        run_root=run_root,
        plan_items=pl,
        plan_pools=plan_pools,
        tables_root=tables_root,
    )
    library_source = library_state.source
    library_artifact = library_state.artifact
    library_records = library_state.records
    library_cursor = library_state.cursor
    ensure_run_meta_dir(run_root)
    state_ctx = init_run_state(
        run_root=run_root,
        run_id=str(cfg.run.id),
        schema_version=str(cfg.schema_version),
        config_sha256=config_sha,
        allow_config_mismatch=allow_config_mismatch,
    )

    if resume:
        _replay_usr_overlay_backlog_for_resume(
            cfg=cfg,
            cfg_path=loaded.path,
            run_root=run_root,
        )

    resume_state = load_resume_state(
        resume=resume,
        loaded=loaded,
        tables_root=tables_root,
        config_sha=config_sha,
        allowed_config_sha256=state_ctx.accepted_config_sha256,
    )
    existing_counts = resume_state.existing_counts
    existing_usage_by_plan = resume_state.existing_usage_by_plan
    site_failure_counts = resume_state.site_failure_counts
    attempt_counters = resume_state.attempt_counters
    library_build_counts = resume_state.library_build_counts
    if existing_counts:
        total = sum(existing_counts.values())
        per_plan = dict(existing_counts)
        log.info(
            "Resuming from existing outputs: %d sequences across %d plan(s).",
            total,
            len(existing_counts),
        )
    reconciliation = reconcile_run_state_with_outputs(
        path=state_ctx.path,
        run_id=str(cfg.run.id),
        schema_version=str(cfg.schema_version),
        config_sha256=config_sha,
        accepted_config_sha256=state_ctx.accepted_config_sha256,
        run_root=str(run_root),
        created_at=state_ctx.created_at,
        existing_counts=existing_counts,
    )
    if reconciliation.updated:
        direction = "ahead" if reconciliation.state_total > reconciliation.durable_total else "behind"
        log.warning(
            "Reconciled run_state.json (%s durable outputs): state_total=%d durable_total=%d.",
            direction,
            reconciliation.state_total,
            reconciliation.durable_total,
        )
    existing_total = sum(existing_counts.values())

    plan_stats, plan_order = init_plan_stats(
        plan_items=pl,
        plan_pools=plan_pools,
        existing_counts=existing_counts,
        existing_library_build_counts=library_build_counts,
    )

    def _accumulate_stats(key: tuple[str, str], stats: dict) -> None:
        if key not in plan_stats:
            plan_stats[key] = {
                "generated": 0,
                "duplicates_skipped": 0,
                "failed_solutions": 0,
                "total_resamples": 0,
                "libraries_built": 0,
                "stall_events": 0,
                "failed_min_count_per_tf": 0,
                "failed_required_regulators": 0,
                "failed_min_count_by_regulator": 0,
                "failed_min_required_regulators": 0,
                "duplicate_solutions": 0,
            }
            plan_order.append(key)
        dest = plan_stats[key]
        for field in dest:
            dest[field] += int(stats.get(field, 0))

    # Round-robin scheduler
    round_robin = bool(cfg.runtime.round_robin)
    if round_robin and str(cfg.generation.sampling.pool_strategy) == "iterative_subsample":
        log.warning(
            "round_robin=true with pool_strategy=iterative_subsample will rebuild libraries more frequently; "
            "expect higher runtime for multi-plan runs."
        )
    display_map_by_input = build_display_map_by_input(
        plan_items=pl,
        plan_pools=plan_pools,
        inputs=cfg.inputs,
        cfg_path=loaded.path,
    )
    checkpoint_every = int(cfg.runtime.checkpoint_every)
    state_counts = init_state_counts(
        plan_items=pl,
        plan_pools=plan_pools,
        existing_counts=existing_counts,
    )
    total_quota = sum(int(item.quota) for item in pl)
    shared_dashboard = _build_shared_dashboard(cfg.logging)

    def _write_state() -> None:
        write_run_state(
            path=state_ctx.path,
            run_id=str(cfg.run.id),
            schema_version=str(cfg.schema_version),
            config_sha256=config_sha,
            accepted_config_sha256=state_ctx.accepted_config_sha256,
            run_root=str(run_root),
            counts=state_counts,
            created_at=state_ctx.created_at,
        )

    _write_state()

    plan_context = PlanRunContext(
        global_cfg=cfg,
        sinks=sinks,
        chosen_solver=chosen_solver,
        deps=deps,
        rng=rng,
        np_rng=np_rng_stage_b,
        cfg_path=loaded.path,
        run_id=str(cfg.run.id),
        run_root=run_root_str,
        run_config_path=run_cfg_path,
        run_config_sha256=config_sha,
        random_seed=seed,
        dense_arrays_version=dense_arrays_version,
        dense_arrays_version_source=dense_arrays_version_source,
        show_tfbs=show_tfbs,
        show_solutions=show_solutions,
        output_bio_type=output_bio_type,
        output_alphabet=output_alphabet,
    )
    execution_state = PlanExecutionState(
        inputs_manifest=inputs_manifest_entries,
        state_counts=state_counts,
        total_quota=total_quota,
        checkpoint_every=checkpoint_every,
        write_state=_write_state,
        shared_dashboard=shared_dashboard,
        site_failure_counts=site_failure_counts,
        source_cache=source_cache,
        attempt_counters=attempt_counters,
        library_records=library_records,
        library_cursor=library_cursor,
        library_source=library_source,
        library_build_rows=library_build_rows,
        library_member_rows=library_member_rows,
        solution_rows=solution_rows,
        composition_rows=composition_rows,
        events_path=events_path,
        display_map_by_input=display_map_by_input,
        consecutive_failures_by_plan={},
        no_progress_seconds_by_plan={},
    )
    try:
        plan_execution = run_plan_schedule(
            plan_items=pl,
            plan_pools=plan_pools,
            plan_pool_sources=plan_pool_sources,
            existing_counts=existing_counts,
            round_robin=round_robin,
            process_plan=_process_plan_for_source,
            plan_context=plan_context,
            execution_state=execution_state,
            accumulate_stats=_accumulate_stats,
            plan_pool_input_meta=_plan_pool_input_meta,
            existing_usage_by_plan=existing_usage_by_plan,
        )
        per_plan = plan_execution.per_plan
        total = plan_execution.total
        plan_leaderboards = plan_execution.plan_leaderboards

        for sink in sinks:
            sink.finalize()

        finalize_run_outputs(
            cfg=cfg,
            run_root=run_root,
            run_root_str=run_root_str,
            cfg_path=loaded.path,
            config_sha=config_sha,
            seed=seed,
            seeds=seeds,
            chosen_solver=chosen_solver,
            solver_attempt_timeout_seconds=solver_attempt_timeout_seconds,
            solver_threads=solver_threads,
            dense_arrays_version=dense_arrays_version,
            dense_arrays_version_source=dense_arrays_version_source,
            plan_stats=plan_stats,
            plan_order=plan_order,
            plan_leaderboards=plan_leaderboards,
            plan_pools=plan_pools,
            plan_items=pl,
            inputs_manifest_entries=inputs_manifest_entries,
            library_source=library_source,
            library_artifact=library_artifact,
            library_build_rows=library_build_rows,
            library_member_rows=library_member_rows,
            composition_rows=composition_rows,
        )

        _write_state()

        return RunSummary(
            total_generated=total,
            per_plan=per_plan,
            generated_this_run=max(0, int(total) - int(existing_total)),
        )
    except Exception as exc:
        for sink in sinks:
            try:
                sink.on_run_failure(exc)
            except Exception:
                log.debug("Failed to emit sink failure signal.", exc_info=True)
        raise
    finally:
        if shared_dashboard is not None:
            shared_dashboard.close()
