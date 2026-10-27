"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/study_state.py

State and persistence helpers for Study run lifecycle management.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dnadesign.cruncher.study.manifest import (
    StudyManifestV1,
    StudyStatusV1,
    StudyTrialRun,
    summarize_trial_statuses,
    utc_now_iso,
    write_study_manifest,
    write_study_status,
)


def _refresh_status(
    status: StudyStatusV1,
    manifest: StudyManifestV1,
    *,
    final: bool = False,
    failed: bool = False,
) -> None:
    counts = summarize_trial_statuses(manifest.trial_runs)
    status.total_runs = int(counts["total_runs"])
    status.pending_runs = int(counts["pending_runs"])
    status.running_runs = int(counts["running_runs"])
    status.success_runs = int(counts["success_runs"])
    status.error_runs = int(counts["error_runs"])
    status.skipped_runs = int(counts["skipped_runs"])
    if failed:
        status.status = "failed"
    elif status.running_runs > 0:
        status.status = "running"
    elif status.error_runs > 0:
        status.status = "completed_with_errors"
    else:
        status.status = "completed"
    status.updated_at = utc_now_iso()
    if final:
        status.finished_at = utc_now_iso()


def _prepare_study_run_dir(*, study_run_dir: Path, resume: bool, force_overwrite: bool) -> None:
    if study_run_dir.exists():
        if not study_run_dir.is_dir():
            raise ValueError(f"Study run path already exists and is not a directory: {study_run_dir}")
        if force_overwrite:
            shutil.rmtree(study_run_dir)
            return
        if not resume:
            raise ValueError(f"Study run directory already exists: {study_run_dir}. Use --resume or --force-overwrite.")
        return
    if resume:
        raise FileNotFoundError(f"Cannot resume missing study run directory: {study_run_dir}")


def _persist_study_state(
    *,
    manifest_file: Path,
    status_file: Path,
    manifest: StudyManifestV1,
    status: StudyStatusV1,
) -> None:
    write_study_manifest(manifest_file, manifest)
    write_study_status(status_file, status)


def _write_trial_run_state(
    *,
    manifest_file: Path,
    status_file: Path,
    manifest: StudyManifestV1,
    status: StudyStatusV1,
    trial_index: int,
    trial_run: StudyTrialRun,
) -> StudyTrialRun:
    manifest.trial_runs[trial_index] = trial_run
    _refresh_status(status, manifest)
    _persist_study_state(
        manifest_file=manifest_file,
        status_file=status_file,
        manifest=manifest,
        status=status,
    )
    return trial_run


def _mark_trial_run_started(
    *,
    manifest_file: Path,
    status_file: Path,
    manifest: StudyManifestV1,
    status: StudyStatusV1,
    trial_index: int,
) -> StudyTrialRun:
    trial_run = manifest.trial_runs[trial_index]
    trial_run.status = "running"
    trial_run.error = None
    trial_run.started_at = utc_now_iso()
    trial_run.finished_at = None
    return _write_trial_run_state(
        manifest_file=manifest_file,
        status_file=status_file,
        manifest=manifest,
        status=status,
        trial_index=trial_index,
        trial_run=trial_run,
    )


def _finalize_trial_run(
    *,
    manifest_file: Path,
    status_file: Path,
    manifest: StudyManifestV1,
    status: StudyStatusV1,
    trial_index: int,
    trial_run: StudyTrialRun,
) -> StudyTrialRun:
    trial_run.finished_at = utc_now_iso()
    return _write_trial_run_state(
        manifest_file=manifest_file,
        status_file=status_file,
        manifest=manifest,
        status=status,
        trial_index=trial_index,
        trial_run=trial_run,
    )
