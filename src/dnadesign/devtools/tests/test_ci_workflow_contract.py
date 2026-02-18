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
    assert "core-lint-test-build" in jobs
    assert "external-integration" in jobs
    assert "ci-gate" in jobs


def test_core_lane_installs_ffmpeg() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["core-lint-test-build"]["steps"]
    assert any(step.get("name") == "Install FFmpeg" for step in steps)


def test_lint_test_build_check_mirrors_core_lane_result() -> None:
    workflow = _workflow()
    lint_job = workflow["jobs"]["lint-test-build"]
    assert lint_job["needs"] == "core-lint-test-build"
    assert lint_job["if"] == "always()"


def test_scope_outputs_expose_core_external_integration_keys() -> None:
    workflow = _workflow()
    outputs = workflow["jobs"]["detect-ci-scope"]["outputs"]
    assert "run-external-integration" in outputs
    assert "run-full-core" in outputs
    assert "external-integration-tools-csv" in outputs
