"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/study/test_study_workflow_hardening.py

Focused hardening checks for Study workflow state transitions and run-path.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dnadesign.cruncher.app.study_workflow as study_workflow
from dnadesign.cruncher.study.manifest import (
    StudyManifestV1,
    StudyStatusV1,
    StudyTrialRun,
    load_study_manifest,
    load_study_status,
)


def _base_manifest() -> StudyManifestV1:
    return StudyManifestV1(
        study_name="demo_study",
        study_id="study123",
        spec_path="/tmp/spec.yaml",
        spec_sha256="specsha",
        base_config_path="/tmp/config.yaml",
        base_config_sha256="cfgsha",
        created_at="2026-04-23T00:00:00+00:00",
        trial_runs=[
            StudyTrialRun(
                trial_id="BASE",
                seed=11,
                target_set_index=1,
                target_tfs=["lexA", "cpxR"],
            )
        ],
    )


def test_prepare_study_run_dir_rejects_existing_file_path(tmp_path: Path) -> None:
    blocked_path = tmp_path / "outputs" / "studies" / "demo" / "study123"
    blocked_path.parent.mkdir(parents=True, exist_ok=True)
    blocked_path.write_text("blocked\n")

    with pytest.raises(ValueError, match="not a directory"):
        study_workflow._prepare_study_run_dir(
            study_run_dir=blocked_path,
            resume=False,
            force_overwrite=False,
        )


def test_trial_run_state_helpers_refresh_and_persist(tmp_path: Path) -> None:
    manifest = _base_manifest()
    status = StudyStatusV1(
        study_name=manifest.study_name,
        study_id=manifest.study_id,
        status="running",
    )
    manifest_file = tmp_path / "study_manifest.json"
    status_file = tmp_path / "study_status.json"

    trial_run = study_workflow._mark_trial_run_started(
        manifest_file=manifest_file,
        status_file=status_file,
        manifest=manifest,
        status=status,
        trial_index=0,
    )

    assert trial_run.status == "running"
    assert trial_run.started_at is not None
    assert trial_run.finished_at is None

    persisted_running_status = load_study_status(status_file)
    assert persisted_running_status.total_runs == 1
    assert persisted_running_status.pending_runs == 0
    assert persisted_running_status.running_runs == 1
    assert persisted_running_status.status == "running"

    trial_run.status = "success"
    trial_run.run_dir = str(tmp_path / "trial_run")
    study_workflow._finalize_trial_run(
        manifest_file=manifest_file,
        status_file=status_file,
        manifest=manifest,
        status=status,
        trial_index=0,
        trial_run=trial_run,
    )

    persisted_manifest = load_study_manifest(manifest_file)
    persisted_status = load_study_status(status_file)
    assert persisted_manifest.trial_runs[0].status == "success"
    assert persisted_manifest.trial_runs[0].run_dir == str(tmp_path / "trial_run")
    assert persisted_manifest.trial_runs[0].finished_at is not None
    assert persisted_status.running_runs == 0
    assert persisted_status.success_runs == 1
    assert persisted_status.status == "completed"
