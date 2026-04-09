"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/test_ci_workflow_contract.py

Tests for CI workflow contract semantics enforced by the repository.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _workflow() -> dict:
    workflow_path = Path(__file__).resolve().parents[4] / ".github" / "workflows" / "ci.yaml"
    return yaml.safe_load(workflow_path.read_text(encoding="utf-8"))


def test_ci_workflow_uses_core_and_external_integration_lane_ids() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert "secrets-hygiene" in jobs
    assert "core-lint-test-build" in jobs
    assert "external-integration" in jobs
    assert "ci-gate" in jobs


def test_core_lane_installs_ffmpeg() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["core-lint-test-build"]["steps"]
    ffmpeg_step = next(step for step in steps if step.get("name") == "Install FFmpeg")
    condition = str(ffmpeg_step.get("if", ""))
    run_script = str(ffmpeg_step.get("run", ""))
    assert "run-full-core == 'true'" in condition
    assert "baserender" in condition
    assert "command -v ffmpeg" in run_script
    assert "timeout 300 bash -lc" in run_script
    assert "sudo apt-get update" in run_script
    assert "apt-get install -y --no-install-recommends ffmpeg" in run_script
    assert "continuing without video-output coverage" in run_script


def test_ci_gate_is_the_required_aggregate_check() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert "lint-test-build" not in jobs
    gate_job = jobs["ci-gate"]
    assert gate_job["if"] == "always()"
    assert gate_job["needs"] == [
        "detect-ci-scope",
        "secrets-hygiene",
        "core-lint-test-build",
        "external-integration",
        "quality-score-inputs",
    ]


def test_secrets_hygiene_job_runs_baseline_and_full_tree_scans() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["secrets-hygiene"]["steps"]
    baseline_step = next(step for step in steps if step.get("name") == "Detect-secrets baseline path hygiene")
    full_tree_step = next(step for step in steps if step.get("name") == "Detect-secrets full-tree check")

    baseline_run = str(baseline_step.get("run", ""))
    assert "uv run python -m dnadesign.devtools.secrets_baseline_check" in baseline_run
    assert "--repo-root ." in baseline_run
    assert "--baseline .secrets.baseline" in baseline_run

    full_tree_run = str(full_tree_step.get("run", ""))
    assert "uv run pre-commit run detect-secrets --all-files" in full_tree_run


def test_scope_outputs_expose_core_external_integration_keys() -> None:
    workflow = _workflow()
    outputs = workflow["jobs"]["detect-ci-scope"]["outputs"]
    assert "run-external-integration" in outputs
    assert "run-full-core" in outputs
    assert "external-integration-tools-csv" in outputs


def test_quality_entropy_job_uses_locked_uv_environment() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["quality-entropy"]["steps"]
    step_names = {step.get("name") for step in steps}
    assert "Install uv" in step_names
    assert "Install dependencies (locked)" in step_names

    entropy_step = next(step for step in steps if step.get("name") == "Build quality entropy report")
    run_script = str(entropy_step.get("run", ""))
    assert "uv run python -m dnadesign.devtools.quality_entropy" in run_script
