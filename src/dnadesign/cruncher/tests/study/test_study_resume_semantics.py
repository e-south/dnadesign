"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/study/test_study_resume_semantics.py

Validate Study resume behavior from partially-complete manifest state.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.study_workflow import run_study
from dnadesign.cruncher.study.layout import study_manifest_path, study_status_path
from dnadesign.cruncher.study.manifest import (
    load_study_manifest,
    load_study_status,
    write_study_manifest,
    write_study_status,
)
from dnadesign.cruncher.tests.study._helpers import write_study_spec, write_workspace_config


def test_study_resume_recovers_running_trial(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "smoke.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=False,
        seeds=[11, 12],
        trials=[{"id": "L6", "factors": {"sample.sequence_length": 6}}],
    )

    run_dir = run_study(spec_path)
    manifest_file = study_manifest_path(run_dir)
    status_file = study_status_path(run_dir)
    manifest = load_study_manifest(manifest_file)
    status = load_study_status(status_file)
    assert status.success_runs == 2

    # Simulate an interrupted run with one trial marked as running.
    interrupted = manifest.trial_runs[0]
    interrupted.status = "running"
    interrupted.run_dir = None
    interrupted.error = "interrupted"
    interrupted.finished_at = None
    manifest.trial_runs[0] = interrupted
    write_study_manifest(manifest_file, manifest)
    status.status = "running"
    status.running_runs = 1
    status.success_runs = 1
    write_study_status(status_file, status)

    resumed_dir = run_study(spec_path, resume=True)
    assert resumed_dir == run_dir

    resumed_manifest = load_study_manifest(manifest_file)
    resumed_status = load_study_status(status_file)
    assert resumed_status.status == "completed"
    assert resumed_status.success_runs == 2
    assert resumed_status.error_runs == 0
    assert all(item.status == "success" for item in resumed_manifest.trial_runs)
