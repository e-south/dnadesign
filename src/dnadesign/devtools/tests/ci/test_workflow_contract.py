"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/ci/test_workflow_contract.py

Tests for CI workflow contract semantics enforced by the repository.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _workflow(filename: str = "ci.yaml") -> dict:
    current = Path(__file__).resolve()
    repo_root = next(parent for parent in current.parents if (parent / "pyproject.toml").exists())
    workflow_path = repo_root / ".github" / "workflows" / filename
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
    install_step = next(step for step in steps if step.get("name") == "Install dependencies (locked)")
    assert install_step["run"] == "uv sync --locked --only-group lint --no-install-project"

    assert "PYTHONPATH=src uv run --no-sync python -m dnadesign.devtools.security.secrets_baseline" in baseline_run
    assert "--repo-root ." in baseline_run
    assert "--baseline .secrets.baseline" in baseline_run

    full_tree_run = str(full_tree_step.get("run", ""))
    assert "uv run --no-sync pre-commit run detect-secrets --all-files" in full_tree_run


def test_core_precommit_defers_detect_secrets_to_dedicated_lane() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["core-lint-test-build"]["steps"]
    precommit_step = next(step for step in steps if step.get("name") == "Pre-commit (non-Ruff hooks)")

    assert set(precommit_step["env"]["SKIP"].split(",")) == {"ruff-check", "ruff-format", "detect-secrets"}


def test_scope_outputs_expose_core_external_integration_keys() -> None:
    workflow = _workflow()
    outputs = workflow["jobs"]["detect-ci-scope"]["outputs"]
    assert "run-external-integration" in outputs
    assert "run-full-core" in outputs
    assert "external-integration-tools-csv" in outputs


def test_fresh_ci_lanes_recreate_changed_file_list_before_resolving_test_targets() -> None:
    workflow = _workflow()

    for job_name in ("core-lint-test-build", "external-integration"):
        steps = workflow["jobs"][job_name]["steps"]
        collect_indices = [
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Collect changed files for test target resolution"
        ]
        assert len(collect_indices) == 1
        collect_index = collect_indices[0]
        target_index = next(
            index
            for index, step in enumerate(steps)
            if "dnadesign.devtools.ci.test_targets" in str(step.get("run", ""))
        )
        collect_run = str(steps[collect_index].get("run", ""))
        checkout_step = next(step for step in steps[:collect_index] if step.get("uses") == "actions/checkout@v5")
        assert checkout_step.get("with", {}).get("fetch-depth") == 0
        assert "dnadesign.devtools.ci.changed_files" in collect_run
        assert "--output-file .ci_changed_files.txt" in collect_run
        assert collect_index < target_index


def test_quality_entropy_job_uses_stdlib_only_runtime() -> None:
    workflow = _workflow("quality-entropy.yaml")
    steps = workflow["jobs"]["report"]["steps"]
    step_names = {step.get("name") for step in steps}
    assert "Install uv" not in step_names
    assert "Install dependencies (locked)" not in step_names

    entropy_step = next(step for step in steps if step.get("name") == "Build quality entropy report")
    run_script = str(entropy_step.get("run", ""))
    assert "PYTHONPATH=src python3 -S -m dnadesign.devtools.quality.entropy" in run_script


def test_ci_jobs_install_only_the_locked_dependency_groups_they_use() -> None:
    workflow = _workflow()

    expected_sync = {
        "secrets-hygiene": "uv sync --locked --only-group lint --no-install-project",
        "core-lint-test-build": "uv sync --locked --dev",
        "external-integration": "uv sync --locked --group test",
    }
    for job_name, expected in expected_sync.items():
        steps = workflow["jobs"][job_name]["steps"]
        install_step = next(step for step in steps if step.get("name") == "Install dependencies (locked)")
        assert install_step["run"] == expected


def test_scheduled_entropy_is_isolated_from_full_ci() -> None:
    ci_workflow = _workflow()
    entropy_workflow = _workflow("quality-entropy.yaml")
    ci_triggers = ci_workflow.get("on", ci_workflow.get(True))
    entropy_triggers = entropy_workflow.get("on", entropy_workflow.get(True))

    assert "schedule" not in ci_triggers
    assert "workflow_dispatch" in ci_triggers
    assert entropy_triggers["schedule"] == [{"cron": "15 07 * * 1"}]
    assert "workflow_dispatch" in entropy_triggers
    assert set(entropy_workflow["jobs"]) == {"report"}
    assert entropy_workflow["concurrency"]["group"].startswith("quality-entropy-")


def test_no_tool_smoke_targets_include_workflow_contracts() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["core-lint-test-build"]["steps"]
    test_step = next(step for step in steps if step.get("name") == "Tests (core lane) + coverage report")

    assert "src/dnadesign/devtools/tests/ci/test_workflow_contract.py" in test_step["run"]


def test_core_installs_ffmpeg_only_after_static_gates() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["core-lint-test-build"]["steps"]
    step_index = {step.get("name"): index for index, step in enumerate(steps)}

    assert step_index["Format check"] < step_index["Install FFmpeg"] < step_index["Tests (core lane) + coverage report"]
