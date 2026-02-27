"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/stage_b_solution_persistence.py

Stage-B accepted-solution persistence and progress accounting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from ...utils.sequence_utils import gc_fraction
from ..metadata import build_metadata
from .attempts import _flush_attempts, _flush_solutions
from .solution_outputs import record_solution_outputs
from .stage_b_solution_types import StageBProgressContext, StageBSolutionOutputContext
from .usage_tracking import _update_usage_summary


def write_solution_output(
    *,
    context: StageBSolutionOutputContext,
    sol,
    seq: str,
    final_seq: str,
    used_tfbs: list[str],
    used_tfbs_detail: list[dict],
    used_tf_counts: dict[str, int],
    used_tf_list: list[str],
    pad_meta: dict,
    covers_all: bool,
    covers_required: bool,
    tfbs_parts: list[str],
    regulator_labels: list[str],
    library_for_opt: list[str],
    sampling_info: dict,
    required_regulators: list[str],
    min_required_regulators: int | None,
    sampling_fraction: float | None,
    sampling_fraction_pairs: float | None,
    sampling_library_index: int,
    sampling_library_hash: str,
    library_tfbs: list[str],
    library_tfs: list[str],
    library_site_ids: list[str],
    library_sources: list[str],
    promoter_detail: dict,
    sequence_validation: dict,
    solver_status: str | None,
    solver_objective: float | None,
    solver_solve_time_s: float | None,
    apply_pad_offsets,
) -> tuple[str, list[dict]]:
    adjusted_tfbs_detail = apply_pad_offsets(used_tfbs_detail, pad_meta)
    gc_core = gc_fraction(seq)
    gc_total = gc_fraction(final_seq)
    created_at = datetime.now(timezone.utc).isoformat()
    derived = build_metadata(
        sol=sol,
        plan_name=context.plan_name,
        tfbs_parts=tfbs_parts,
        regulator_labels=regulator_labels,
        library_for_opt=library_for_opt,
        fixed_elements=context.fixed_elements,
        chosen_solver=context.chosen_solver,
        solver_strategy=context.solver_strategy,
        solver_attempt_timeout_seconds=context.solver_attempt_timeout_seconds,
        solver_threads=context.solver_threads,
        solver_strands=context.solver_strands,
        seq_len=context.seq_len,
        actual_length=len(final_seq),
        pad_meta=pad_meta,
        sampling_meta=sampling_info,
        schema_version=context.schema_version,
        created_at=created_at,
        run_id=context.run_id,
        run_root=context.run_root,
        run_config_path=context.run_config_path,
        run_config_sha256=context.run_config_sha256,
        random_seed=context.random_seed,
        policy_pad=context.policy_pad,
        policy_sampling=context.policy_sampling,
        policy_solver=context.policy_solver,
        input_meta=context.input_meta,
        fixed_elements_dump=context.fixed_elements_dump,
        used_tfbs=used_tfbs,
        used_tfbs_detail=adjusted_tfbs_detail,
        used_tf_counts=used_tf_counts,
        used_tf_list=used_tf_list,
        min_count_per_tf=context.min_count_per_tf,
        covers_all_tfs_in_solution=bool(covers_all),
        required_regulators=required_regulators,
        min_required_regulators=min_required_regulators,
        min_count_by_regulator=context.plan_min_count_by_regulator,
        covers_required_regulators=bool(covers_required),
        gc_core=gc_core,
        gc_total=gc_total,
        promoter_detail=promoter_detail,
        sequence_validation=sequence_validation,
        input_row_count=context.input_row_count,
        input_tf_count=context.input_tf_count,
        input_tfbs_count=context.input_tfbs_count,
        input_tf_tfbs_pair_count=context.input_tf_tfbs_pair_count,
        sampling_fraction=sampling_fraction,
        sampling_fraction_pairs=sampling_fraction_pairs,
        sampling_library_index=int(sampling_library_index),
        sampling_library_hash=str(sampling_library_hash),
        solver_status=solver_status,
        solver_objective=solver_objective,
        solver_solve_time_s=solver_solve_time_s,
        dense_arrays_version=context.dense_arrays_version,
        dense_arrays_version_source=context.dense_arrays_version_source,
        final_sequence=final_seq,
    )
    if not derived:
        return "skipped_no_metadata", adjusted_tfbs_detail
    accepted = record_solution_outputs(
        sinks=context.sinks,
        final_seq=final_seq,
        derived=derived,
        source_label=context.source_label,
        plan_name=context.plan_name,
        output_bio_type=context.output_bio_type,
        output_alphabet=context.output_alphabet,
        tables_root=context.tables_root,
        run_id=context.run_id,
        next_attempt_index=context.next_attempt_index,
        used_tf_counts=used_tf_counts,
        used_tf_list=used_tf_list,
        sampling_library_index=int(sampling_library_index),
        sampling_library_hash=str(sampling_library_hash),
        solver_status=solver_status,
        solver_objective=solver_objective,
        solver_solve_time_s=solver_solve_time_s,
        dense_arrays_version=context.dense_arrays_version,
        dense_arrays_version_source=context.dense_arrays_version_source,
        library_tfbs=library_tfbs,
        library_tfs=library_tfs,
        library_site_ids=library_site_ids,
        library_sources=library_sources,
        attempts_buffer=context.attempts_buffer,
        solution_rows=context.solution_rows,
        composition_rows=context.composition_rows,
        events_path=context.events_path,
        used_tfbs=used_tfbs,
        used_tfbs_detail=adjusted_tfbs_detail,
    )
    if not accepted:
        return "duplicate", adjusted_tfbs_detail
    return "accepted", adjusted_tfbs_detail


def persist_candidate_solution(
    *,
    solution_output_context: StageBSolutionOutputContext,
    progress_context: StageBProgressContext,
    sol,
    seq: str,
    final_seq: str,
    used_tfbs: list[str],
    used_tfbs_detail: list[dict],
    used_tf_counts: dict[str, int],
    used_tf_list: list[str],
    pad_meta: dict,
    covers_all: bool,
    covers_required: bool,
    tfbs_parts: list[str],
    regulator_labels: list[str],
    library_for_opt: list[str],
    sampling_info: dict,
    required_regulators: list[str],
    min_required_regulators: int | None,
    sampling_fraction: float | None,
    sampling_fraction_pairs: float | None,
    sampling_library_index: int,
    sampling_library_hash: str,
    library_tfbs: list[str],
    library_tfs: list[str],
    library_site_ids: list[str],
    library_sources: list[str],
    promoter_detail: dict,
    sequence_validation: dict,
    solver_status: str | None,
    solver_objective: float | None,
    solver_solve_time_s: float | None,
    apply_pad_offsets,
    global_generated: int,
    local_generated: int,
    produced_this_library: int,
    duplicate_records: int,
    duplicate_solutions: int,
    failed_solutions: int,
    stall_events: int,
    write_output_fn: Callable[..., tuple[str, list[dict]]] = write_solution_output,
    record_progress_fn: Callable[..., tuple[int, int, int]] | None = None,
) -> tuple[int, int, int, int, bool]:
    if record_progress_fn is None:
        record_progress_fn = record_accepted_solution_progress
    solution_status, adjusted_tfbs_detail = write_output_fn(
        context=solution_output_context,
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
        apply_pad_offsets=apply_pad_offsets,
    )
    if solution_status == "skipped_no_metadata":
        return global_generated, local_generated, produced_this_library, duplicate_records, False
    if solution_status == "duplicate":
        return global_generated, local_generated, produced_this_library, int(duplicate_records) + 1, False

    global_generated, local_generated, produced_this_library = record_progress_fn(
        context=progress_context,
        global_generated=global_generated,
        local_generated=local_generated,
        produced_this_library=produced_this_library,
        sol=sol,
        final_seq=final_seq,
        used_tfbs_detail=adjusted_tfbs_detail,
        used_tf_list=used_tf_list,
        sampling_library_index=int(sampling_library_index),
        library_tfs=library_tfs,
        library_tfbs=library_tfbs,
        duplicate_records=duplicate_records,
        duplicate_solutions=duplicate_solutions,
        failed_solutions=failed_solutions,
        stall_events=stall_events,
    )
    return global_generated, local_generated, produced_this_library, duplicate_records, True


def record_accepted_solution_progress(
    *,
    context: StageBProgressContext,
    global_generated: int,
    local_generated: int,
    produced_this_library: int,
    sol,
    final_seq: str,
    used_tfbs_detail: list[dict],
    used_tf_list: list[str],
    sampling_library_index: int,
    library_tfs: list[str],
    library_tfbs: list[str],
    duplicate_records: int,
    duplicate_solutions: int,
    failed_solutions: int,
    stall_events: int,
) -> tuple[int, int, int]:
    global_generated += 1
    local_generated += 1
    produced_this_library += 1
    _update_usage_summary(context.usage_counts, context.tf_usage_counts, used_tfbs_detail)

    if context.checkpoint_every > 0 and global_generated % max(1, context.checkpoint_every) == 0:
        _flush_attempts(context.tables_root, context.attempts_buffer)
        if context.solution_rows is not None:
            _flush_solutions(context.tables_root, context.solution_rows)
        if context.state_counts is not None:
            context.state_counts[(context.source_label, context.plan_name)] = int(global_generated)
            if context.write_state is not None:
                context.write_state()

    global_total_generated = None
    if context.state_counts is not None:
        context.state_counts[(context.source_label, context.plan_name)] = int(global_generated)
        if context.total_quota is not None:
            global_total_generated = sum(context.state_counts.values())

    context.progress_reporter.record_solution(
        global_generated=global_generated,
        local_generated=local_generated,
        library_index=int(sampling_library_index),
        sol=sol,
        library_tfs=library_tfs,
        library_tfbs=library_tfbs,
        used_tfbs_detail=used_tfbs_detail,
        used_tf_list=used_tf_list,
        final_seq=final_seq,
        counters=context.counters,
        duplicate_records=duplicate_records,
        duplicate_solutions=duplicate_solutions,
        failed_solutions=failed_solutions,
        stall_events=stall_events,
        usage_counts=context.usage_counts,
        tf_usage_counts=context.tf_usage_counts,
        tf_usage_display=context.diagnostics.map_tf_usage(context.tf_usage_counts),
        tfbs_usage_display=context.diagnostics.map_tfbs_usage(context.usage_counts),
        global_total_generated=global_total_generated,
        global_total_quota=context.total_quota,
    )
    context.progress_reporter.record_leaderboard(
        global_generated=global_generated,
        counters=context.counters,
        duplicate_records=duplicate_records,
        duplicate_solutions=duplicate_solutions,
        failed_solutions=failed_solutions,
        stall_events=stall_events,
        failure_counts=context.failure_counts,
        leaderboard_every=context.leaderboard_every,
        log_snapshot=context.diagnostics.log_snapshot,
    )
    if (
        context.leaderboard_every > 0
        and global_generated % max(1, context.leaderboard_every) == 0
        and context.show_solutions
    ):
        context.logger.info(
            "[%s/%s] Example: %s",
            context.source_label,
            context.plan_name,
            final_seq,
        )
    return global_generated, local_generated, produced_this_library
