"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/study_postprocess.py

Postprocess helpers for Study replay, summarize, and final completion phases.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Protocol

from dnadesign.cruncher.study.manifest import StudyManifestV1, StudyStatusV1, utc_now_iso
from dnadesign.cruncher.study.schema_models import StudySpec

logger = logging.getLogger(__name__)
StudyEventCallback = Callable[[str, dict[str, object]], None]
EmitStudyEventFn = Callable[..., None]
AppendStudyLogFn = Callable[[Path, str], None]
RunMMRSweepFn = Callable[..., object]
SummarizeStudyRunFn = Callable[..., object]
RefreshStudyStatusFn = Callable[..., None]
PersistStudyStateFn = Callable[..., None]
MarkPendingSkippedFn = Callable[..., None]


class _StudyBootstrapLike(Protocol):
    spec: StudySpec
    study_run_dir: Path
    manifest_file: Path
    status_file: Path


def _run_study_replays(
    *,
    bootstrap: _StudyBootstrapLike,
    manifest: StudyManifestV1,
    status: StudyStatusV1,
    aborted: bool,
    on_event: StudyEventCallback | None,
    emit_event_fn: EmitStudyEventFn,
    append_log_fn: AppendStudyLogFn,
    run_mmr_sweep_fn: RunMMRSweepFn,
    refresh_status_fn: RefreshStudyStatusFn,
    persist_state_fn: PersistStudyStateFn,
    mark_pending_as_skipped_fn: MarkPendingSkippedFn,
) -> tuple[bool, bool]:
    any_errors = False
    if not bootstrap.spec.replays.mmr_sweep.enabled or aborted:
        return any_errors, aborted
    replay_candidates = [item for item in manifest.trial_runs if item.status == "success" and bool(item.run_dir)]
    total_replays = len(replay_candidates)
    emit_event_fn(on_event, "study_replay_phase_started", total_runs=int(total_replays))
    logger.info("Study replay phase starting: %d successful trial run(s).", total_replays)
    append_log_fn(
        bootstrap.study_run_dir,
        f"REPLAY_START total={total_replays}",
    )
    completed = 0
    for idx, trial_run in enumerate(manifest.trial_runs):
        if trial_run.status != "success" or not trial_run.run_dir:
            continue
        run_dir = Path(trial_run.run_dir)
        logger.info(
            "Study replay run %d/%d: trial=%s seed=%d target_set=%d",
            completed + 1,
            total_replays,
            trial_run.trial_id,
            int(trial_run.seed),
            int(trial_run.target_set_index),
        )
        try:
            run_mmr_sweep_fn(
                run_dir,
                pool_size_values=bootstrap.spec.replays.mmr_sweep.pool_size_values,
                diversity_values=bootstrap.spec.replays.mmr_sweep.diversity_values,
            )
            completed += 1
            emit_event_fn(
                on_event,
                "study_replay_progress",
                completed_runs=int(completed),
                total_runs=int(total_replays),
            )
            append_log_fn(
                bootstrap.study_run_dir,
                f"REPLAY_DONE trial={trial_run.trial_id} seed={trial_run.seed} target_set={trial_run.target_set_index}",
            )
        except Exception as exc:
            any_errors = True
            trial_run.status = "error"
            trial_run.error = f"MMR replay failed: {exc}"
            trial_run.finished_at = utc_now_iso()
            manifest.trial_runs[idx] = trial_run
            append_log_fn(
                bootstrap.study_run_dir,
                "ERROR replay "
                f"trial={trial_run.trial_id} seed={trial_run.seed} "
                f"target_set={trial_run.target_set_index}: {exc}",
            )
            refresh_status_fn(status, manifest)
            persist_state_fn(
                manifest_file=bootstrap.manifest_file,
                status_file=bootstrap.status_file,
                manifest=manifest,
                status=status,
            )
            if bootstrap.spec.execution.on_trial_error == "abort":
                aborted = True
                mark_pending_as_skipped_fn(
                    manifest,
                    reason="Skipped because execution aborted after replay error.",
                )
                break
    logger.info("Study replay phase complete: %d/%d replay run(s) finished.", completed, total_replays)
    emit_event_fn(
        on_event,
        "study_replay_phase_completed",
        completed_runs=int(completed),
        total_runs=int(total_replays),
        aborted=bool(aborted),
    )
    return any_errors, aborted


def _maybe_summarize_study(
    *,
    bootstrap: _StudyBootstrapLike,
    manifest: StudyManifestV1,
    status: StudyStatusV1,
    on_event: StudyEventCallback | None,
    emit_event_fn: EmitStudyEventFn,
    append_log_fn: AppendStudyLogFn,
    summarize_study_run_fn: SummarizeStudyRunFn,
    persist_state_fn: PersistStudyStateFn,
) -> None:
    if not bootstrap.spec.execution.summarize_after_run:
        emit_event_fn(on_event, "study_summarize_skipped", reason="summarize_after_run=false")
        return
    has_non_success = any(item.status != "success" for item in manifest.trial_runs)
    if has_non_success:
        warning = (
            "Summary skipped due trial errors. "
            "Run `cruncher study summarize --allow-partial --run <study_run_dir>` to summarize successes."
        )
        if warning not in status.warnings:
            status.warnings.append(warning)
        status.updated_at = utc_now_iso()
        persist_state_fn(
            manifest_file=bootstrap.manifest_file,
            status_file=bootstrap.status_file,
            manifest=manifest,
            status=status,
        )
        emit_event_fn(on_event, "study_summarize_skipped", reason="non_success_trial_status")
        return
    emit_event_fn(on_event, "study_summarize_phase_started")
    logger.info("Study summarize phase starting.")
    append_log_fn(bootstrap.study_run_dir, "SUMMARIZE_START")
    summarize_study_run_fn(bootstrap.study_run_dir, allow_partial=False)
    append_log_fn(bootstrap.study_run_dir, "SUMMARIZE_DONE")
    logger.info("Study summarize phase complete.")
    emit_event_fn(on_event, "study_summarize_phase_completed")


def _finalize_study_completion(
    *,
    bootstrap: _StudyBootstrapLike,
    manifest: StudyManifestV1,
    status: StudyStatusV1,
    aborted: bool,
    fatal_exc: Exception | None,
    on_event: StudyEventCallback | None,
    emit_event_fn: EmitStudyEventFn,
    refresh_status_fn: RefreshStudyStatusFn,
    persist_state_fn: PersistStudyStateFn,
) -> None:
    refresh_status_fn(status, manifest, final=True, failed=aborted or fatal_exc is not None)
    persist_state_fn(
        manifest_file=bootstrap.manifest_file,
        status_file=bootstrap.status_file,
        manifest=manifest,
        status=status,
    )
    emit_event_fn(
        on_event,
        "study_completed",
        study_name=manifest.study_name,
        study_id=manifest.study_id,
        status=status.status,
        total_runs=int(status.total_runs),
        success_runs=int(status.success_runs),
        error_runs=int(status.error_runs),
        skipped_runs=int(status.skipped_runs),
        pending_runs=int(status.pending_runs),
        running_runs=int(status.running_runs),
        failed=bool(aborted or fatal_exc is not None),
    )
