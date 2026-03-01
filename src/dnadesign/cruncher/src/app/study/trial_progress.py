"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/study/trial_progress.py

Provide shared trial progress and worker lifecycle helpers for study execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

from dnadesign.cruncher.study.manifest import StudyManifestV1, StudyStatusV1, StudyTrialRun, utc_now_iso


@dataclass(frozen=True)
class _TrialWorkerProcess:
    process: multiprocessing.Process
    result_path: Path
    trial_index: int


def _trial_run_is_complete(trial_run: StudyTrialRun) -> bool:
    return trial_run.status == "success" and bool(trial_run.run_dir)


def _log_parallel_trial_progress(
    *,
    logger: logging.Logger,
    status: StudyStatusV1,
    total_runs: int,
    queued_count: int,
) -> None:
    completed = int(status.success_runs + status.error_runs + status.skipped_runs)
    logger.info(
        "Study trial progress: completed=%d/%d success=%d error=%d running=%d queued=%d.",
        completed,
        int(total_runs),
        int(status.success_runs),
        int(status.error_runs),
        int(status.running_runs),
        int(queued_count),
    )


def _study_trial_progress_payload(
    *,
    status: StudyStatusV1,
    total_runs: int,
    queued_count: int,
    worker_count: int,
    active_trial_ids: list[str],
) -> dict[str, object]:
    completed = int(status.success_runs + status.error_runs + status.skipped_runs)
    return {
        "completed_runs": int(completed),
        "total_runs": int(total_runs),
        "success_runs": int(status.success_runs),
        "error_runs": int(status.error_runs),
        "running_runs": int(status.running_runs),
        "queued_runs": int(queued_count),
        "worker_count": int(worker_count),
        "active_trial_ids": list(active_trial_ids),
    }


def _emit_study_trial_progress(
    on_event,
    *,
    status: StudyStatusV1,
    total_runs: int,
    queued_count: int,
    worker_count: int,
    active_trial_ids: list[str],
) -> None:
    if on_event is None:
        return
    on_event(
        "study_trial_progress",
        _study_trial_progress_payload(
            status=status,
            total_runs=total_runs,
            queued_count=queued_count,
            worker_count=worker_count,
            active_trial_ids=active_trial_ids,
        ),
    )


def _shutdown_trial_worker_process(process: multiprocessing.Process, *, terminate: bool) -> None:
    if terminate and process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join()
        return
    process.join()


def _mark_pending_as_skipped(manifest: StudyManifestV1, *, reason: str) -> None:
    for idx, trial_run in enumerate(manifest.trial_runs):
        if trial_run.status == "pending":
            trial_run.status = "skipped"
            trial_run.error = reason
            trial_run.finished_at = utc_now_iso()
            manifest.trial_runs[idx] = trial_run
