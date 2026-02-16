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
import time
from pathlib import Path
from typing import List

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
from .progress import _summarize_tf_counts
from .sampling_diagnostics import SamplingDiagnostics
from .sequence_validation import (
    _apply_pad_offsets,
)
from .stage_b import (
    _compute_sampling_fraction,
    _compute_sampling_fraction_pairs,
    _fixed_elements_dump,
    _merge_min_counts,
    _min_count_by_regulator,
)
from .stage_b_library_builder import LibraryBuilder, LibraryContext
from .stage_b_runtime_checks import (
    _evaluate_sequence_constraints,
    _evaluate_solution_requirements,
    _extract_solver_metrics,
    _maybe_pad_sequence,
)
from .stage_b_runtime_types import (
    PlanInputState,
    PlanRunSettings,
    _close_plan_dashboard,
)
from .stage_b_sampler import LibraryRunResult, SamplingCounters, StageBSampler
from .stage_b_solution_persistence import persist_candidate_solution
from .stage_b_solution_rejections import (
    handle_duplicate_sequence,
    mark_stall_detected,
    reject_sequence_validation_failure,
    reject_solution_requirement_failure,
)
from .stage_b_solution_types import (
    StageBProgressContext,
    StageBRejectionContext,
    StageBSolutionOutputContext,
)
from .usage_tracking import _compute_used_tf_info

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
    if progress_style != "screen":
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

    libraries_built = existing_library_builds
    libraries_built_start = existing_library_builds
    libraries_used = 0
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

    def _build_next_library() -> LibraryContext:
        nonlocal libraries_built, libraries_used
        context = builder.build_next(library_index_start=libraries_built)
        libraries_used += 1
        if library_source_label == "artifact":
            libraries_built = libraries_used
        else:
            libraries_built = int(context.sampling_info.get("library_index", libraries_built))
        return context

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

    solver_min_counts: dict[str, int] | None = None

    def _make_generator(
        _library_for_opt: List[str],
        _regulator_labels: List[str],
        *,
        required_regulators_local: list[str],
        min_required_regulators_local: int | None,
    ):
        nonlocal solver_min_counts
        regulator_by_index = list(_regulator_labels) if _regulator_labels else None
        base_min_counts = _min_count_by_regulator(regulator_by_index, min_count_per_tf)
        solver_min_counts = _merge_min_counts(base_min_counts, plan_min_count_by_regulator)
        fe_dict = fixed_elements.model_dump() if hasattr(fixed_elements, "model_dump") else fixed_elements
        solver_required_regs = required_regulators_local or None
        run = deps.optimizer.build(
            library=_library_for_opt,
            sequence_length=seq_len,
            solver=chosen_solver,
            strategy=solver_strategy,
            fixed_elements=fe_dict,
            strands=solver_strands,
            regulator_by_index=regulator_by_index,
            required_regulators=solver_required_regs,
            min_count_by_regulator=solver_min_counts,
            min_required_regulators=min_required_regulators_local,
            solver_time_limit_seconds=solver_time_limit_seconds,
            solver_threads=solver_threads,
            extra_label=extra_library_label,
        )
        return run

    failed_solutions = 0
    duplicate_records = 0
    stall_events = 0
    failed_min_count_per_tf = 0
    failed_required_regulators = 0
    failed_min_count_by_regulator = 0
    failed_min_required_regulators = 0
    duplicate_solutions = 0
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

    def _run_library(
        library_context: LibraryContext,
        max_per_subsample: int,
        global_generated: int,
        quota: int,
    ) -> LibraryRunResult:
        nonlocal duplicate_records
        nonlocal duplicate_solutions
        nonlocal failed_min_count_by_regulator
        nonlocal failed_min_count_per_tf
        nonlocal failed_required_regulators
        nonlocal failed_solutions
        nonlocal stall_events
        library_for_opt = list(library_context.library_for_opt)
        tfbs_parts = list(library_context.tfbs_parts)
        regulator_labels = list(library_context.regulator_labels)
        sampling_info = dict(library_context.sampling_info)
        sampling_library_index = int(library_context.sampling_library_index)
        sampling_library_hash = str(library_context.sampling_library_hash)
        library_tfbs = list(library_context.library_tfbs)
        library_tfs = list(library_context.library_tfs)
        library_site_ids = list(library_context.library_site_ids)
        library_sources = list(library_context.library_sources)
        required_regulators = list(library_context.required_regulators)
        min_required_regulators = None
        tf_list_from_library = sorted(set(regulator_labels)) if regulator_labels else []
        site_id_by_index = sampling_info.get("site_id_by_index")
        source_by_index = sampling_info.get("source_by_index")
        tfbs_id_by_index = sampling_info.get("tfbs_id_by_index")
        motif_id_by_index = sampling_info.get("motif_id_by_index")
        stage_a_best_hit_score_by_index = sampling_info.get("stage_a_best_hit_score_by_index")
        stage_a_rank_within_regulator_by_index = sampling_info.get("stage_a_rank_within_regulator_by_index")
        stage_a_tier_by_index = sampling_info.get("stage_a_tier_by_index")
        stage_a_fimo_start_by_index = sampling_info.get("stage_a_fimo_start_by_index")
        stage_a_fimo_stop_by_index = sampling_info.get("stage_a_fimo_stop_by_index")
        stage_a_fimo_strand_by_index = sampling_info.get("stage_a_fimo_strand_by_index")
        stage_a_selection_rank_by_index = sampling_info.get("stage_a_selection_rank_by_index")
        stage_a_selection_score_norm_by_index = sampling_info.get("stage_a_selection_score_norm_by_index")
        stage_a_tfbs_core_by_index = sampling_info.get("stage_a_tfbs_core_by_index")

        diagnostics.update_library(
            library_tfs=library_tfs,
            library_tfbs=library_tfbs,
            library_site_ids=library_site_ids,
        )

        sampling_fraction = _compute_sampling_fraction(
            library_for_opt,
            input_tfbs_count=input_tfbs_count,
            pool_strategy=pool_strategy,
        )
        input_meta["sampling_fraction"] = sampling_fraction
        sampling_fraction_pairs = _compute_sampling_fraction_pairs(
            library_for_opt,
            regulator_labels,
            input_pair_count=input_tf_tfbs_pair_count,
            pool_strategy=pool_strategy,
        )
        input_meta["sampling_fraction_pairs"] = sampling_fraction_pairs
        tf_summary = _summarize_tf_counts([display_tf_label(tf) for tf in regulator_labels] if regulator_labels else [])
        library_index = sampling_info.get("library_index")
        strategy_label = sampling_info.get("library_sampling_strategy", library_sampling_strategy)
        pool_label = sampling_info.get("pool_strategy")
        achieved_len = sampling_info.get("achieved_length")
        header = f"Stage-B library for {source_label}/{plan_name}"
        if library_index is not None:
            header = f"{header} (build {library_index})"
        if progress_style != "screen":
            if tf_summary:
                log.info(
                    "%s: %d motifs | TF counts: %s | library_bp=%s pool=%s stage_b_sampling=%s",
                    header,
                    len(library_for_opt),
                    tf_summary,
                    achieved_len,
                    pool_label,
                    strategy_label,
                )
            else:
                log.info(
                    "%s: %d motifs | library_bp=%s pool=%s stage_b_sampling=%s",
                    header,
                    len(library_for_opt),
                    achieved_len,
                    pool_label,
                    strategy_label,
                )

        run = _make_generator(
            library_for_opt,
            regulator_labels,
            required_regulators_local=required_regulators,
            min_required_regulators_local=min_required_regulators,
        )
        opt = run.optimizer
        generator = run.generator
        forbid_each = run.forbid_each

        local_generated = 0
        produced_this_library = 0
        stall_triggered = False
        last_progress = time.monotonic()

        if local_generated < max_per_subsample and global_generated < quota:
            fingerprints = set()
            consecutive_dup = 0
            subsample_started = time.monotonic()
            last_log_warn = subsample_started
            last_progress = subsample_started
            produced_this_library = 0
            stall_triggered = False

            for sol in generator:
                now = time.monotonic()
                if policy.should_trigger_stall(
                    now=now,
                    last_progress=last_progress,
                ):
                    stall_events, stall_triggered = mark_stall_detected(
                        events_path=events_path,
                        source_label=source_label,
                        plan_name=plan_name,
                        stall_seconds=stall_seconds,
                        last_progress=last_progress,
                        now=now,
                        sampling_library_index=int(sampling_library_index),
                        sampling_library_hash=str(sampling_library_hash),
                        stall_events=stall_events,
                        stall_triggered=stall_triggered,
                        emit_event=_emit_event,
                        logger=log,
                    )
                    break
                if policy.should_warn_stall(
                    now=now,
                    last_warn=last_log_warn,
                    last_progress=last_progress,
                ):
                    log.info(
                        "[%s/%s] Still working... %.1fs on current library.",
                        source_label,
                        plan_name,
                        now - subsample_started,
                    )
                    last_log_warn = now
                last_progress = now

                if forbid_each:
                    opt.forbid(sol)
                seq = sol.sequence
                should_continue, should_break, duplicate_solutions, consecutive_dup = handle_duplicate_sequence(
                    sequence=seq,
                    fingerprints=fingerprints,
                    duplicate_solutions=duplicate_solutions,
                    consecutive_dup=consecutive_dup,
                    max_dupes=max_dupes,
                    source_label=source_label,
                    plan_name=plan_name,
                    logger=log,
                )
                if should_break:
                    break
                if should_continue:
                    continue

                used_tfbs, used_tfbs_detail, used_tf_counts, used_tf_list = _compute_used_tf_info(
                    sol,
                    library_for_opt,
                    regulator_labels,
                    fixed_elements,
                    site_id_by_index,
                    source_by_index,
                    tfbs_id_by_index,
                    motif_id_by_index,
                    stage_a_best_hit_score_by_index,
                    stage_a_rank_within_regulator_by_index,
                    stage_a_tier_by_index,
                    stage_a_fimo_start_by_index,
                    stage_a_fimo_stop_by_index,
                    stage_a_fimo_strand_by_index,
                    stage_a_selection_rank_by_index,
                    stage_a_selection_score_norm_by_index,
                    stage_a_tfbs_core_by_index,
                )
                solver_status, solver_objective, solver_solve_time_s = _extract_solver_metrics(sol)

                covers_all, covers_required, rejection_reason, rejection_detail = _evaluate_solution_requirements(
                    min_count_per_tf=min_count_per_tf,
                    tf_list_from_library=tf_list_from_library,
                    required_regulators=required_regulators,
                    plan_min_count_by_regulator=plan_min_count_by_regulator,
                    used_tf_counts=used_tf_counts,
                )
                if rejection_reason is not None:
                    (
                        failed_solutions,
                        failed_min_count_per_tf,
                        failed_required_regulators,
                        failed_min_count_by_regulator,
                    ) = reject_solution_requirement_failure(
                        rejection_context=rejection_context,
                        rejection_reason=rejection_reason,
                        rejection_detail=rejection_detail,
                        sequence=seq,
                        used_tf_counts=used_tf_counts,
                        used_tf_list=used_tf_list,
                        sampling_library_index=int(sampling_library_index),
                        sampling_library_hash=str(sampling_library_hash),
                        solver_status=solver_status,
                        solver_objective=solver_objective,
                        solver_solve_time_s=solver_solve_time_s,
                        library_tfbs=library_tfbs,
                        library_tfs=library_tfs,
                        library_site_ids=library_site_ids,
                        library_sources=library_sources,
                        failed_solutions=failed_solutions,
                        max_failed_solutions=max_failed_solutions,
                        source_label=source_label,
                        plan_name=plan_name,
                        failed_min_count_per_tf=failed_min_count_per_tf,
                        failed_required_regulators=failed_required_regulators,
                        failed_min_count_by_regulator=failed_min_count_by_regulator,
                        diagnostics=diagnostics,
                    )
                    continue

                final_seq, pad_meta = _maybe_pad_sequence(
                    sequence=seq,
                    seq_len=seq_len,
                    source_label=source_label,
                    plan_name=plan_name,
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
                    sequence_constraint_patterns=sequence_constraint_patterns,
                    pad_builder=deps.pad,
                    rng=rng,
                )

                sequence_constraints_eval = _evaluate_sequence_constraints(
                    final_seq=final_seq,
                    compiled_sequence_constraints=compiled_sequence_constraints,
                    fixed_elements_dump=fixed_elements_dump,
                    source_label=source_label,
                    plan_name=plan_name,
                    sampling_library_index=int(sampling_library_index),
                    sampling_library_hash=str(sampling_library_hash),
                )
                promoter_detail = dict(sequence_constraints_eval.promoter_detail)
                sequence_validation = dict(sequence_constraints_eval.sequence_validation)
                if sequence_constraints_eval.rejection_detail is not None:
                    failed_solutions = reject_sequence_validation_failure(
                        rejection_context=rejection_context,
                        rejection_detail=dict(sequence_constraints_eval.rejection_detail),
                        rejection_event_payload=sequence_constraints_eval.rejection_event_payload,
                        validation_error=sequence_constraints_eval.error,
                        final_seq=final_seq,
                        used_tf_counts=used_tf_counts,
                        used_tf_list=used_tf_list,
                        sampling_library_index=int(sampling_library_index),
                        sampling_library_hash=str(sampling_library_hash),
                        solver_status=solver_status,
                        solver_objective=solver_objective,
                        solver_solve_time_s=solver_solve_time_s,
                        library_tfbs=library_tfbs,
                        library_tfs=library_tfs,
                        library_site_ids=library_site_ids,
                        library_sources=library_sources,
                        failed_solutions=failed_solutions,
                        max_failed_solutions=max_failed_solutions,
                        source_label=source_label,
                        plan_name=plan_name,
                        events_path=events_path,
                        emit_event=_emit_event,
                        logger=log,
                    )
                    continue

                global_generated, local_generated, produced_this_library, duplicate_records, accepted = (
                    persist_candidate_solution(
                        solution_output_context=solution_output_context,
                        progress_context=progress_context,
                        sol=sol,
                        seq=seq,
                        final_seq=final_seq,
                        used_tfbs=used_tfbs,
                        used_tfbs_detail=used_tfbs_detail,
                        used_tf_counts=used_tf_counts,
                        used_tf_list=used_tf_list,
                        pad_meta=pad_meta,
                        covers_all=covers_all,
                        covers_required=covers_required,
                        tfbs_parts=tfbs_parts,
                        regulator_labels=regulator_labels,
                        library_for_opt=library_for_opt,
                        sampling_info=sampling_info,
                        required_regulators=required_regulators,
                        min_required_regulators=min_required_regulators,
                        sampling_fraction=sampling_fraction,
                        sampling_fraction_pairs=sampling_fraction_pairs,
                        sampling_library_index=int(sampling_library_index),
                        sampling_library_hash=str(sampling_library_hash),
                        library_tfbs=library_tfbs,
                        library_tfs=library_tfs,
                        library_site_ids=library_site_ids,
                        library_sources=library_sources,
                        promoter_detail=promoter_detail,
                        sequence_validation=sequence_validation,
                        solver_status=solver_status,
                        solver_objective=solver_objective,
                        solver_solve_time_s=solver_solve_time_s,
                        apply_pad_offsets=_apply_pad_offsets,
                        global_generated=global_generated,
                        local_generated=local_generated,
                        produced_this_library=produced_this_library,
                        duplicate_records=duplicate_records,
                        duplicate_solutions=duplicate_solutions,
                        failed_solutions=failed_solutions,
                        stall_events=stall_events,
                    )
                )
                if not accepted:
                    continue

                if local_generated >= max_per_subsample or global_generated >= quota:
                    break

        if produced_this_library == 0 and not stall_triggered and stall_seconds > 0:
            now = time.monotonic()
            if (now - last_progress) >= stall_seconds:
                stall_events, stall_triggered = mark_stall_detected(
                    events_path=events_path,
                    source_label=source_label,
                    plan_name=plan_name,
                    stall_seconds=stall_seconds,
                    last_progress=last_progress,
                    now=now,
                    sampling_library_index=int(sampling_library_index),
                    sampling_library_hash=str(sampling_library_hash),
                    stall_events=stall_events,
                    stall_triggered=stall_triggered,
                    emit_event=_emit_event,
                    logger=log,
                )

        return LibraryRunResult(
            produced=produced_this_library,
            stall_triggered=stall_triggered,
            global_generated=int(global_generated),
        )

    def _on_no_solution(library_context: LibraryContext, reason: str) -> None:
        attempt_index = next_attempt_index()
        detail = {"stall_seconds": stall_seconds} if reason == "stall_no_solution" else {}
        _append_attempt(
            tables_root,
            run_id=run_id,
            input_name=source_label,
            plan_name=plan_name,
            attempt_index=attempt_index,
            status="failed",
            reason=reason,
            detail=detail,
            sequence=None,
            used_tf_counts=None,
            used_tf_list=[],
            sampling_library_index=int(library_context.sampling_library_index),
            sampling_library_hash=str(library_context.sampling_library_hash),
            solver_status=None,
            solver_objective=None,
            solver_solve_time_s=None,
            dense_arrays_version=dense_arrays_version,
            dense_arrays_version_source=dense_arrays_version_source,
            library_tfbs=list(library_context.library_tfbs),
            library_tfs=list(library_context.library_tfs),
            library_site_ids=list(library_context.library_site_ids),
            library_sources=list(library_context.library_sources),
            attempts_buffer=attempts_buffer,
        )

    def _on_resample(library_context: LibraryContext, reason: str, produced_this_library: int) -> None:
        if events_path is None:
            return
        try:
            _emit_event(
                events_path,
                event="RESAMPLE_TRIGGERED",
                payload={
                    "input_name": source_label,
                    "plan_name": plan_name,
                    "reason": reason,
                    "produced_this_library": int(produced_this_library),
                    "library_index": int(library_context.sampling_library_index),
                    "library_hash": str(library_context.sampling_library_hash),
                },
            )
        except Exception as exc:
            raise RuntimeError("Failed to emit RESAMPLE_TRIGGERED event.") from exc

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
        build_next_library=_build_next_library,
        run_library=_run_library,
        on_no_solution=_on_no_solution,
        on_resample=_on_resample,
        already_generated=already_generated,
        one_subsample_only=one_subsample_only,
        plan_start=plan_start,
    )
    produced_total_this_call = sampling_result.generated
    global_generated = int(already_generated) + int(produced_total_this_call)
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
                "duplicates_skipped": duplicate_records,
                "failed_solutions": failed_solutions,
                "total_resamples": counters.total_resamples,
                "libraries_built": max(0, libraries_built - libraries_built_start),
                "stall_events": stall_events,
                "failed_min_count_per_tf": failed_min_count_per_tf,
                "failed_required_regulators": failed_required_regulators,
                "failed_min_count_by_regulator": failed_min_count_by_regulator,
                "failed_min_required_regulators": failed_min_required_regulators,
                "duplicate_solutions": duplicate_solutions,
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
        "duplicates_skipped": duplicate_records,
        "failed_solutions": failed_solutions,
        "total_resamples": counters.total_resamples,
        "libraries_built": max(0, libraries_built - libraries_built_start),
        "stall_events": stall_events,
        "failed_min_count_per_tf": failed_min_count_per_tf,
        "failed_required_regulators": failed_required_regulators,
        "failed_min_count_by_regulator": failed_min_count_by_regulator,
        "failed_min_required_regulators": failed_min_required_regulators,
        "duplicate_solutions": duplicate_solutions,
        "leaderboard_latest": snapshot,
    }
