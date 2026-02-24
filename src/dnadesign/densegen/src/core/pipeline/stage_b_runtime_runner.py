"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/stage_b_runtime_runner.py

Stage-B sampling execution loop and library resampling runtime flow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

from ...config import ResolvedPlanItem
from ..run_paths import run_tables_root
from ..sequence_constraints import compile_sequence_constraints
from .attempts import (
    _append_attempt,
    _flush_attempts,
    _flush_solutions,
)
from .outputs import _emit_event
from .plan_context import PlanExecutionState, PlanRunContext
from .sampling_diagnostics import SamplingDiagnostics
from .sequence_validation import _apply_pad_offsets
from .stage_b import _fixed_elements_dump
from .stage_b_library_builder import LibraryBuilder
from .stage_b_runtime_callbacks import (
    StageBLibraryRuntimeCallbacks,
    StageBLibraryRuntimeContext,
    StageBLibraryRuntimeState,
)
from .stage_b_runtime_types import (
    PlanInputState,
    PlanRunSettings,
    _close_plan_dashboard,
)
from .stage_b_sampler import SamplingCounters, StageBSampler
from .stage_b_solution_types import (
    StageBProgressContext,
    StageBRejectionContext,
    StageBSolutionOutputContext,
)

log = logging.getLogger(__name__)


def _run_stage_b_sampling(
    *,
    settings: PlanRunSettings,
    input_state: PlanInputState,
    plan_item: ResolvedPlanItem,
    context: PlanRunContext,
    execution_state: PlanExecutionState,
    existing_library_builds: int,
    already_generated: int,
    one_subsample_only: bool,
    display_tf_label,
    next_attempt_index,
) -> tuple[int, dict]:
    source_label = settings.source_label
    plan_name = settings.plan_name
    quota = settings.quota
    seq_len = settings.seq_len
    sampling_cfg = settings.sampling_cfg
    pool_strategy = settings.pool_strategy
    runtime = settings.runtime
    pad = settings.pad
    solver = settings.solver
    progress = settings.progress
    policy = settings.policy
    policy_pad = settings.policy_pad
    policy_sampling = settings.policy_sampling
    policy_solver = settings.policy_solver
    plan_start = settings.plan_start

    sinks = context.sinks
    chosen_solver = context.chosen_solver
    deps = context.deps
    rng = context.rng
    np_rng = context.np_rng
    run_id = context.run_id
    run_root = context.run_root
    run_config_path = context.run_config_path
    run_config_sha256 = context.run_config_sha256
    random_seed = context.random_seed
    dense_arrays_version = context.dense_arrays_version
    dense_arrays_version_source = context.dense_arrays_version_source
    output_bio_type = context.output_bio_type
    output_alphabet = context.output_alphabet

    existing_usage_counts = execution_state.existing_usage_counts
    state_counts = execution_state.state_counts
    total_quota = execution_state.total_quota
    write_state = execution_state.write_state
    site_failure_counts = execution_state.site_failure_counts
    library_records = execution_state.library_records
    library_cursor = execution_state.library_cursor
    library_source = execution_state.library_source
    library_build_rows = execution_state.library_build_rows
    library_member_rows = execution_state.library_member_rows
    solution_rows = execution_state.solution_rows
    composition_rows = execution_state.composition_rows
    events_path = execution_state.events_path

    pool = input_state.pool
    input_meta = input_state.input_meta
    input_row_count = input_state.input_row_count
    input_tf_count = input_state.input_tf_count
    input_tfbs_count = input_state.input_tfbs_count
    input_tf_tfbs_pair_count = input_state.input_tf_tfbs_pair_count

    progress_style = progress.progress_style
    show_tfbs = progress.show_tfbs
    show_solutions = progress.show_solutions
    progress_reporter = progress.reporter
    dashboard = progress.dashboard
    shared_dashboard = execution_state.shared_dashboard

    max_per_subsample = runtime.max_per_subsample
    min_count_per_tf = runtime.min_count_per_tf
    max_dupes = runtime.max_dupes
    stall_seconds = runtime.stall_seconds
    max_failed_solutions = runtime.max_failed_solutions
    leaderboard_every = runtime.leaderboard_every
    checkpoint_every = runtime.checkpoint_every

    pad_enabled = pad.enabled
    pad_mode = pad.mode
    pad_end = pad.end
    pad_gc_mode = pad.gc_mode
    pad_gc_min = pad.gc_min
    pad_gc_max = pad.gc_max
    pad_gc_target = pad.gc_target
    pad_gc_tolerance = pad.gc_tolerance
    pad_gc_min_length = pad.gc_min_length
    pad_max_tries = pad.max_tries
    solver_strategy = solver.strategy
    solver_strands = solver.strands
    solver_time_limit_seconds = solver.time_limit_seconds
    solver_threads = solver.threads
    extra_library_label = settings.extra_library_label

    backend = str(chosen_solver) if chosen_solver is not None else "-"
    strategy = str(solver_strategy)
    strands = str(solver_strands)
    time_limit = "-" if solver_time_limit_seconds is None else str(solver_time_limit_seconds)
    threads = "-" if solver_threads is None else str(solver_threads)
    progress_reporter.solver_settings = (
        f"backend={backend} strategy={strategy} strands={strands} "
        f"time_limit={time_limit}s threads={threads} seq_len={seq_len}"
    )
    if progress_style == "stream":
        log.info(
            "Stage-B solver settings: backend=%s strategy=%s strands=%s "
            "time_limit_seconds=%s threads=%s sequence_length=%s",
            backend,
            strategy,
            strands,
            time_limit,
            threads,
            seq_len,
        )

    fixed_elements = plan_item.fixed_elements
    constraints = plan_item.regulator_constraints
    plan_min_count_by_regulator = dict(constraints.min_count_by_regulator or {})
    fixed_elements_dump = _fixed_elements_dump(fixed_elements)
    compiled_sequence_constraints = None
    sequence_constraint_patterns: list[str] = []
    sequence_constraints_cfg = getattr(context.global_cfg.generation, "sequence_constraints", None)
    if sequence_constraints_cfg is not None and list(getattr(sequence_constraints_cfg, "forbid_kmers", []) or []):
        try:
            compiled_sequence_constraints = compile_sequence_constraints(
                sequence_constraints=sequence_constraints_cfg,
                motif_sets=dict(getattr(context.global_cfg, "motif_sets", {}) or {}),
                fixed_elements_dump=fixed_elements_dump,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[{source_label}/{plan_name}] Failed to compile generation.sequence_constraints: {exc}"
            ) from exc
        sequence_constraint_patterns = list(compiled_sequence_constraints.generation_forbidden_patterns)

    libraries_built = int(existing_library_builds)
    libraries_built_start = int(existing_library_builds)
    library_source_label = str(library_source or getattr(sampling_cfg, "library_source", "build")).lower()
    if library_source_label not in {"build", "artifact"}:
        raise ValueError(f"Unsupported Stage-B sampling.library_source: {library_source_label}")
    if library_source_label == "artifact" and library_cursor is not None:
        prior_used = int(library_cursor.get((source_label, plan_name), 0))
        libraries_built = prior_used
        libraries_built_start = prior_used
    library_sampling_strategy = str(sampling_cfg.library_sampling_strategy)
    iterative_max_libraries = int(sampling_cfg.iterative_max_libraries)
    iterative_min_new_solutions = int(sampling_cfg.iterative_min_new_solutions)

    usage_counts: dict[tuple[str, str], int] = dict(existing_usage_counts or {})
    tf_usage_counts: dict[str, int] = {}
    for (tf, _tfbs), count in usage_counts.items():
        tf_usage_counts[tf] = tf_usage_counts.get(tf, 0) + int(count)
    track_failures = site_failure_counts is not None
    failure_counts = site_failure_counts if site_failure_counts is not None else {}
    attempts_buffer: list[dict] = []
    run_root_path = Path(run_root)
    tables_root = run_tables_root(run_root_path)

    builder = LibraryBuilder(
        source_label=source_label,
        plan_item=plan_item,
        pool=pool,
        sampling_cfg=sampling_cfg,
        seq_len=seq_len,
        min_count_per_tf=min_count_per_tf,
        usage_counts=usage_counts,
        failure_counts=site_failure_counts,
        rng=rng,
        np_rng=np_rng,
        library_source_label=library_source_label,
        library_records=library_records,
        library_cursor=library_cursor,
        events_path=events_path,
        library_build_rows=library_build_rows,
        library_member_rows=library_member_rows,
    )

    diagnostics = SamplingDiagnostics(
        usage_counts=usage_counts,
        tf_usage_counts=tf_usage_counts,
        failure_counts=failure_counts,
        source_label=source_label,
        plan_name=plan_name,
        display_tf_label=display_tf_label,
        progress_style=progress_style,
        show_tfbs=show_tfbs,
        track_failures=track_failures,
        logger=log,
        library_tfs=[],
        library_tfbs=[],
        library_site_ids=[],
    )

    runtime_state = StageBLibraryRuntimeState(
        libraries_built=libraries_built,
        libraries_built_start=libraries_built_start,
    )
    counters = SamplingCounters()

    solution_output_context = StageBSolutionOutputContext(
        source_label=source_label,
        plan_name=plan_name,
        fixed_elements=fixed_elements,
        fixed_elements_dump=fixed_elements_dump,
        chosen_solver=chosen_solver,
        solver_strategy=solver_strategy,
        solver_time_limit_seconds=solver_time_limit_seconds,
        solver_threads=solver_threads,
        solver_strands=solver_strands,
        seq_len=seq_len,
        schema_version=str(context.global_cfg.schema_version),
        run_id=run_id,
        run_root=run_root,
        run_config_path=run_config_path,
        run_config_sha256=run_config_sha256,
        random_seed=random_seed,
        policy_pad=policy_pad,
        policy_sampling=policy_sampling,
        policy_solver=policy_solver,
        input_meta=input_meta,
        min_count_per_tf=min_count_per_tf,
        plan_min_count_by_regulator=plan_min_count_by_regulator,
        input_row_count=input_row_count,
        input_tf_count=input_tf_count,
        input_tfbs_count=input_tfbs_count,
        input_tf_tfbs_pair_count=input_tf_tfbs_pair_count,
        output_bio_type=output_bio_type,
        output_alphabet=output_alphabet,
        tables_root=tables_root,
        next_attempt_index=next_attempt_index,
        dense_arrays_version=dense_arrays_version,
        dense_arrays_version_source=dense_arrays_version_source,
        sinks=sinks,
        attempts_buffer=attempts_buffer,
        solution_rows=solution_rows,
        composition_rows=composition_rows,
        events_path=events_path,
    )

    rejection_context = StageBRejectionContext(
        source_label=source_label,
        plan_name=plan_name,
        tables_root=tables_root,
        run_id=run_id,
        dense_arrays_version=dense_arrays_version,
        dense_arrays_version_source=dense_arrays_version_source,
        attempts_buffer=attempts_buffer,
        next_attempt_index=next_attempt_index,
    )

    progress_context = StageBProgressContext(
        source_label=source_label,
        plan_name=plan_name,
        checkpoint_every=checkpoint_every,
        tables_root=tables_root,
        attempts_buffer=attempts_buffer,
        solution_rows=solution_rows,
        state_counts=state_counts,
        write_state=write_state,
        total_quota=total_quota,
        progress_reporter=progress_reporter,
        counters=counters,
        failure_counts=failure_counts,
        leaderboard_every=leaderboard_every,
        show_solutions=show_solutions,
        usage_counts=usage_counts,
        tf_usage_counts=tf_usage_counts,
        diagnostics=diagnostics,
        logger=log,
    )

    callback_context = StageBLibraryRuntimeContext(
        source_label=source_label,
        plan_name=plan_name,
        seq_len=seq_len,
        pool_strategy=pool_strategy,
        min_count_per_tf=min_count_per_tf,
        max_dupes=max_dupes,
        stall_seconds=stall_seconds,
        max_failed_solutions=max_failed_solutions,
        progress_style=progress_style,
        show_solutions=show_solutions,
        pad_enabled=pad_enabled,
        pad_end=pad_end,
        pad_mode=pad_mode,
        pad_gc_mode=pad_gc_mode,
        pad_gc_min=pad_gc_min,
        pad_gc_max=pad_gc_max,
        pad_gc_target=pad_gc_target,
        pad_gc_tolerance=pad_gc_tolerance,
        pad_gc_min_length=pad_gc_min_length,
        pad_max_tries=pad_max_tries,
        solver_strategy=solver_strategy,
        solver_strands=solver_strands,
        solver_time_limit_seconds=solver_time_limit_seconds,
        solver_threads=solver_threads,
        extra_library_label=extra_library_label,
        fixed_elements=fixed_elements,
        fixed_elements_dump=fixed_elements_dump,
        plan_min_count_by_regulator=plan_min_count_by_regulator,
        input_meta=input_meta,
        input_tfbs_count=input_tfbs_count,
        input_tf_tfbs_pair_count=input_tf_tfbs_pair_count,
        sequence_constraint_patterns=sequence_constraint_patterns,
        compiled_sequence_constraints=compiled_sequence_constraints,
        library_sampling_strategy=library_sampling_strategy,
        policy=policy,
        deps=deps,
        rng=rng,
        diagnostics=diagnostics,
        display_tf_label=display_tf_label,
        rejection_context=rejection_context,
        solution_output_context=solution_output_context,
        progress_context=progress_context,
        events_path=events_path,
        tables_root=tables_root,
        run_id=run_id,
        dense_arrays_version=dense_arrays_version,
        dense_arrays_version_source=dense_arrays_version_source,
        next_attempt_index=next_attempt_index,
        emit_event=_emit_event,
        append_attempt=_append_attempt,
        apply_pad_offsets=_apply_pad_offsets,
        logger=log,
    )
    callbacks = StageBLibraryRuntimeCallbacks(
        builder=builder,
        library_source_label=library_source_label,
        context=callback_context,
        state=runtime_state,
    )

    sampler = StageBSampler(
        source_label=source_label,
        plan_name=plan_name,
        quota=quota,
        policy=policy,
        max_per_subsample=max_per_subsample,
        pool_strategy=pool_strategy,
        iterative_min_new_solutions=iterative_min_new_solutions,
        iterative_max_libraries=iterative_max_libraries,
        counters=counters,
    )
    sampling_result = sampler.run(
        build_next_library=callbacks.build_next_library,
        run_library=callbacks.run_library,
        on_no_solution=callbacks.on_no_solution,
        on_resample=callbacks.on_resample,
        already_generated=already_generated,
        one_subsample_only=one_subsample_only,
        plan_start=plan_start,
    )
    produced_total_this_call = sampling_result.generated
    global_generated = int(already_generated) + int(produced_total_this_call)
    runtime_state = callbacks.state
    for sink in sinks:
        sink.flush()

        if one_subsample_only:
            _flush_attempts(tables_root, attempts_buffer)
            if solution_rows is not None:
                _flush_solutions(tables_root, solution_rows)
            if state_counts is not None:
                state_counts[(source_label, plan_name)] = int(global_generated)
                if write_state is not None:
                    write_state()
            snapshot = diagnostics.snapshot()
            if global_generated >= quota and (usage_counts or tf_usage_counts or failure_counts):
                diagnostics.log_snapshot()
            _close_plan_dashboard(dashboard=dashboard, shared_dashboard=shared_dashboard)
            return produced_total_this_call, {
                "generated": produced_total_this_call,
                "duplicates_skipped": runtime_state.duplicate_records,
                "failed_solutions": runtime_state.failed_solutions,
                "total_resamples": counters.total_resamples,
                "libraries_built": max(0, runtime_state.libraries_built - runtime_state.libraries_built_start),
                "stall_events": runtime_state.stall_events,
                "failed_min_count_per_tf": runtime_state.failed_min_count_per_tf,
                "failed_required_regulators": runtime_state.failed_required_regulators,
                "failed_min_count_by_regulator": runtime_state.failed_min_count_by_regulator,
                "failed_min_required_regulators": runtime_state.failed_min_required_regulators,
                "duplicate_solutions": runtime_state.duplicate_solutions,
                "leaderboard_latest": snapshot,
            }

    _flush_attempts(tables_root, attempts_buffer)
    if solution_rows is not None:
        _flush_solutions(tables_root, solution_rows)
    log.info("Completed %s/%s: %d/%d", source_label, plan_name, global_generated, quota)
    if state_counts is not None:
        state_counts[(source_label, plan_name)] = int(global_generated)
        if write_state is not None:
            write_state()
    snapshot = diagnostics.snapshot()
    if usage_counts or tf_usage_counts or failure_counts:
        diagnostics.log_snapshot()
    _close_plan_dashboard(dashboard=dashboard, shared_dashboard=shared_dashboard)
    return produced_total_this_call, {
        "generated": produced_total_this_call,
        "duplicates_skipped": runtime_state.duplicate_records,
        "failed_solutions": runtime_state.failed_solutions,
        "total_resamples": counters.total_resamples,
        "libraries_built": max(0, runtime_state.libraries_built - runtime_state.libraries_built_start),
        "stall_events": runtime_state.stall_events,
        "failed_min_count_per_tf": runtime_state.failed_min_count_per_tf,
        "failed_required_regulators": runtime_state.failed_required_regulators,
        "failed_min_count_by_regulator": runtime_state.failed_min_count_by_regulator,
        "failed_min_required_regulators": runtime_state.failed_min_required_regulators,
        "duplicate_solutions": runtime_state.duplicate_solutions,
        "leaderboard_latest": snapshot,
    }
