"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/stage_b_solution_rejections.py

Stage-B rejection handling, duplicate checks, and failure-cap enforcement.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .attempts import _log_rejection
from .stage_b_solution_types import StageBRejectionContext


def enforce_failed_solution_cap(
    *,
    failed_solutions: int,
    max_failed_solutions: int,
    source_label: str,
    plan_name: str,
    cause: Exception | None = None,
) -> None:
    if max_failed_solutions <= 0 or failed_solutions <= max_failed_solutions:
        return
    message = f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
    if cause is not None:
        raise RuntimeError(message) from cause
    raise RuntimeError(message)


def reject_solution_attempt(
    *,
    context: StageBRejectionContext,
    reason: str,
    detail: dict,
    sequence: str,
    used_tf_counts: dict[str, int],
    used_tf_list: list[str],
    sampling_library_index: int,
    sampling_library_hash: str,
    solver_status: str | None,
    solver_objective: float | None,
    solver_solve_time_s: float | None,
    library_tfbs: list[str],
    library_tfs: list[str],
    library_site_ids: list[str],
    library_sources: list[str],
) -> None:
    attempt_index = context.next_attempt_index()
    _log_rejection(
        context.tables_root,
        run_id=context.run_id,
        input_name=context.source_label,
        plan_name=context.plan_name,
        attempt_index=attempt_index,
        reason=reason,
        detail=detail,
        sequence=sequence,
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
    )


def reject_solution_and_enforce_cap(
    *,
    context: StageBRejectionContext,
    reason: str,
    detail: dict,
    sequence: str,
    used_tf_counts: dict[str, int],
    used_tf_list: list[str],
    sampling_library_index: int,
    sampling_library_hash: str,
    solver_status: str | None,
    solver_objective: float | None,
    solver_solve_time_s: float | None,
    library_tfbs: list[str],
    library_tfs: list[str],
    library_site_ids: list[str],
    library_sources: list[str],
    failed_solutions: int,
    max_failed_solutions: int,
    source_label: str,
    plan_name: str,
    cause: Exception | None = None,
) -> int:
    rejected_solutions = int(failed_solutions) + 1
    reject_solution_attempt(
        context=context,
        reason=reason,
        detail=detail,
        sequence=sequence,
        used_tf_counts=used_tf_counts,
        used_tf_list=used_tf_list,
        sampling_library_index=sampling_library_index,
        sampling_library_hash=sampling_library_hash,
        solver_status=solver_status,
        solver_objective=solver_objective,
        solver_solve_time_s=solver_solve_time_s,
        library_tfbs=library_tfbs,
        library_tfs=library_tfs,
        library_site_ids=library_site_ids,
        library_sources=library_sources,
    )
    enforce_failed_solution_cap(
        failed_solutions=rejected_solutions,
        max_failed_solutions=max_failed_solutions,
        source_label=source_label,
        plan_name=plan_name,
        cause=cause,
    )
    return rejected_solutions


def emit_sequence_validation_failed_event(
    *,
    events_path: Path | None,
    payload: dict,
    emit_event: Callable[..., None],
    logger: Any,
) -> None:
    if events_path is None:
        return
    try:
        emit_event(
            events_path,
            event="SEQUENCE_VALIDATION_FAILED",
            payload=payload,
        )
    except Exception:
        logger.debug("Failed to emit SEQUENCE_VALIDATION_FAILED event.", exc_info=True)


def mark_stall_detected(
    *,
    events_path: Path | None,
    source_label: str,
    plan_name: str,
    stall_seconds: int,
    last_progress: float,
    now: float,
    sampling_library_index: int,
    sampling_library_hash: str,
    stall_events: int,
    stall_triggered: bool,
    emit_event: Callable[..., None],
    logger: Any,
) -> tuple[int, bool]:
    if stall_triggered:
        return int(stall_events), True
    logger.info(
        "[%s/%s] Stall (> %ds) with no solutions; will resample.",
        source_label,
        plan_name,
        stall_seconds,
    )
    updated_stall_events = int(stall_events) + 1
    if events_path is not None:
        try:
            emit_event(
                events_path,
                event="STALL_DETECTED",
                payload={
                    "input_name": source_label,
                    "plan_name": plan_name,
                    "stall_seconds": float(now - last_progress),
                    "library_index": int(sampling_library_index),
                    "library_hash": str(sampling_library_hash),
                },
            )
        except Exception as exc:
            raise RuntimeError("Failed to emit STALL_DETECTED event.") from exc
    return updated_stall_events, True


def handle_duplicate_sequence(
    *,
    sequence: str,
    fingerprints: set[str],
    duplicate_solutions: int,
    consecutive_dup: int,
    max_dupes: int,
    source_label: str,
    plan_name: str,
    logger: Any,
) -> tuple[bool, bool, int, int]:
    if sequence not in fingerprints:
        fingerprints.add(sequence)
        return False, False, int(duplicate_solutions), 0
    updated_duplicate_solutions = int(duplicate_solutions) + 1
    updated_consecutive_dup = int(consecutive_dup) + 1
    if updated_consecutive_dup >= int(max_dupes):
        logger.info(
            "[%s/%s] Duplicate guard (>= %d in a row); will resample.",
            source_label,
            plan_name,
            max_dupes,
        )
        return False, True, updated_duplicate_solutions, updated_consecutive_dup
    return True, False, updated_duplicate_solutions, updated_consecutive_dup


def reject_solution_requirement_failure(
    *,
    rejection_context: StageBRejectionContext,
    rejection_reason: str,
    rejection_detail: dict,
    sequence: str,
    used_tf_counts: dict[str, int],
    used_tf_list: list[str],
    sampling_library_index: int,
    sampling_library_hash: str,
    solver_status: str | None,
    solver_objective: float | None,
    solver_solve_time_s: float | None,
    library_tfbs: list[str],
    library_tfs: list[str],
    library_site_ids: list[str],
    library_sources: list[str],
    failed_solutions: int,
    max_failed_solutions: int,
    source_label: str,
    plan_name: str,
    failed_min_count_per_tf: int,
    failed_required_regulators: int,
    failed_min_count_by_regulator: int,
    diagnostics: Any,
) -> tuple[int, int, int, int]:
    updated_failed_min_count_per_tf = int(failed_min_count_per_tf)
    updated_failed_required_regulators = int(failed_required_regulators)
    updated_failed_min_count_by_regulator = int(failed_min_count_by_regulator)
    if rejection_reason == "min_count_per_tf":
        updated_failed_min_count_per_tf += 1
    elif rejection_reason == "required_regulators":
        updated_failed_required_regulators += 1
    elif rejection_reason == "min_count_by_regulator":
        updated_failed_min_count_by_regulator += 1
    else:
        raise RuntimeError(f"Unsupported requirement rejection reason: {rejection_reason}")

    diagnostics.record_site_failures(rejection_reason)
    updated_failed_solutions = reject_solution_and_enforce_cap(
        context=rejection_context,
        reason=rejection_reason,
        detail=rejection_detail,
        sequence=sequence,
        used_tf_counts=used_tf_counts,
        used_tf_list=used_tf_list,
        sampling_library_index=sampling_library_index,
        sampling_library_hash=sampling_library_hash,
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
    )
    return (
        updated_failed_solutions,
        updated_failed_min_count_per_tf,
        updated_failed_required_regulators,
        updated_failed_min_count_by_regulator,
    )


def reject_sequence_validation_failure(
    *,
    rejection_context: StageBRejectionContext,
    rejection_detail: dict,
    rejection_event_payload: dict | None,
    validation_error: Exception | None,
    final_seq: str,
    used_tf_counts: dict[str, int],
    used_tf_list: list[str],
    sampling_library_index: int,
    sampling_library_hash: str,
    solver_status: str | None,
    solver_objective: float | None,
    solver_solve_time_s: float | None,
    library_tfbs: list[str],
    library_tfs: list[str],
    library_site_ids: list[str],
    library_sources: list[str],
    failed_solutions: int,
    max_failed_solutions: int,
    source_label: str,
    plan_name: str,
    events_path: Path | None,
    emit_event: Callable[..., None],
    logger: Any,
) -> int:
    updated_failed_solutions = reject_solution_and_enforce_cap(
        context=rejection_context,
        reason="sequence_validation_failed",
        detail=dict(rejection_detail),
        sequence=final_seq,
        used_tf_counts=used_tf_counts,
        used_tf_list=used_tf_list,
        sampling_library_index=sampling_library_index,
        sampling_library_hash=sampling_library_hash,
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
        cause=validation_error,
    )
    if rejection_event_payload is not None:
        emit_sequence_validation_failed_event(
            events_path=events_path,
            payload=dict(rejection_event_payload),
            emit_event=emit_event,
            logger=logger,
        )
    if max_failed_solutions == 0:
        if validation_error is not None:
            raise RuntimeError(
                f"[{source_label}/{plan_name}] sequence validation failed and "
                f"runtime.max_failed_solutions=0 ({validation_error})."
            ) from validation_error
        raise RuntimeError(
            f"[{source_label}/{plan_name}] sequence validation failed and runtime.max_failed_solutions=0."
        )
    return updated_failed_solutions
