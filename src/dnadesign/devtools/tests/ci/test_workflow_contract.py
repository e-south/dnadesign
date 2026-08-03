"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/ci/test_workflow_contract.py

Tests for CI workflow contract semantics enforced by the repository.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    return next(parent for parent in current.parents if (parent / "pyproject.toml").exists())


def _workflow(filename: str = "ci.yaml") -> dict:
    workflow_path = _repo_root() / ".github" / "workflows" / filename
    return yaml.safe_load(workflow_path.read_text(encoding="utf-8"))


def _workflow_paths() -> list[Path]:
    workflow_dir = _repo_root() / ".github" / "workflows"
    return sorted((*workflow_dir.glob("*.yaml"), *workflow_dir.glob("*.yml")))


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


def test_secrets_hygiene_job_rejects_tracked_personal_operator_data() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["secrets-hygiene"]["steps"]
    privacy_step = next(step for step in steps if step.get("name") == "Tracked text privacy check")

    assert privacy_step["run"] == (
        "PYTHONPATH=src uv run --no-sync python -m dnadesign.devtools.security.tracked_text_privacy --repo-root ."
    )


def test_core_precommit_defers_detect_secrets_to_dedicated_lane() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["core-lint-test-build"]["steps"]
    precommit_step = next(step for step in steps if step.get("name") == "Pre-commit (non-Ruff hooks)")
    precommit_run = str(precommit_step.get("run", ""))

    assert set(precommit_step["env"]["SKIP"].split(",")) == {"ruff-check", "ruff-format", "detect-secrets"}
    assert 'base_sha="${{ github.event.pull_request.base.sha }}"' in precommit_run
    assert 'git cat-file -e "${base_sha}^{commit}"' in precommit_run
    assert '--from-ref "${base_sha}" --to-ref "${{ github.sha }}"' in precommit_run
    assert "git fetch" not in precommit_run


def test_core_lane_always_runs_dependency_security_contract() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["core-lint-test-build"]["steps"]
    security_step = next(step for step in steps if step.get("name") == "Dependency security contracts")

    assert security_step["run"] == (
        "uv run pytest -q src/dnadesign/devtools/tests/security/test_dependency_security_contract.py"
    )


def test_core_lane_fails_fast_on_package_layout_contracts() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["core-lint-test-build"]["steps"]
    step_names = [str(step.get("name", "")) for step in steps]
    layout_step = next(step for step in steps if step.get("name") == "Package layout contracts")

    assert layout_step["run"] == "uv run pytest -q src/dnadesign/devtools/tests/package/test_layout.py"
    assert step_names.index("Package layout contracts") < step_names.index("Tests (core lane) + coverage report")


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
        checkout_step = next(
            step for step in steps[:collect_index] if str(step.get("uses", "")).startswith("actions/checkout@")
        )
        assert checkout_step.get("with", {}).get("fetch-depth") == 2
        assert "dnadesign.devtools.ci.changed_files" in collect_run
        assert '--base-sha "${{ github.event.pull_request.base.sha }}"' in collect_run
        assert "--output-file .ci_changed_files.txt" in collect_run
        assert collect_index < target_index


def test_scope_detection_uses_two_commit_shallow_checkout() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["detect-ci-scope"]["steps"]
    checkout_step = next(step for step in steps if str(step.get("uses", "")).startswith("actions/checkout@"))

    assert checkout_step.get("with", {}).get("fetch-depth") == 2


def test_scope_detection_pins_the_event_base_snapshot() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["detect-ci-scope"]["steps"]
    collect_step = next(step for step in steps if step.get("name") == "Collect changed files for scope detection")
    collect_run = str(collect_step.get("run", ""))

    assert '--base-sha "${{ github.event.pull_request.base.sha }}"' in collect_run
    assert '--head-sha "${{ github.sha }}"' in collect_run


def test_third_party_workflow_actions_are_pinned_to_full_commit_shas() -> None:
    unpinned: list[str] = []
    for workflow_path in _workflow_paths():
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        for job_id, job in workflow.get("jobs", {}).items():
            for step in job.get("steps", ()):
                action_ref = str(step.get("uses", "")).strip()
                if action_ref and re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", action_ref) is None:
                    unpinned.append(f"{workflow_path.name}:{job_id}:{action_ref}")

    assert unpinned == []


def test_uv_action_caches_are_pruned_before_persistence() -> None:
    workflow = _workflow()
    uv_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", ())
        if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    ]

    assert len(uv_steps) == 3
    for step in uv_steps:
        assert step["with"]["enable-cache"] is True
        assert step["with"]["prune-cache"] is True


def test_dependency_review_workflow_is_pr_only_and_least_privilege() -> None:
    workflow = _workflow("dependency-review.yaml")
    triggers = workflow.get("on", workflow.get(True))

    assert set(triggers) == {"pull_request"}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert "pull_request.number" in workflow["concurrency"]["group"]
    assert set(workflow["jobs"]) == {"dependency-review"}

    job = workflow["jobs"]["dependency-review"]
    assert "permissions" not in job
    review_step = next(step for step in job["steps"] if step["uses"].startswith("actions/dependency-review-action@"))
    assert review_step["with"]["fail-on-severity"] == "moderate"


def test_checkout_and_artifact_actions_use_current_node24_releases() -> None:
    expected_refs = {
        "actions/checkout": "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/upload-artifact": "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "actions/download-artifact": "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
    }
    observed_refs: dict[str, list[str]] = {action: [] for action in expected_refs}
    for workflow_path in _workflow_paths():
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        for job in workflow.get("jobs", {}).values():
            for step in job.get("steps", ()):
                action_ref = str(step.get("uses", ""))
                action = action_ref.partition("@")[0]
                if action in observed_refs:
                    observed_refs[action].append(action_ref)

    for action, expected_ref in expected_refs.items():
        assert observed_refs[action], f"expected at least one {action} use"
        assert set(observed_refs[action]) == {expected_ref}


def test_codecov_uploads_use_node24_compatible_action() -> None:
    workflow = _workflow()
    expected_ref = "codecov/codecov-action@fb8b3582c8e4def4969c97caa2f19720cb33a72f"

    upload_refs = [
        str(step.get("uses", ""))
        for job in workflow["jobs"].values()
        for step in job.get("steps", ())
        if str(step.get("uses", "")).startswith("codecov/codecov-action@")
    ]

    assert upload_refs == [expected_ref, expected_ref]


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


def test_coverage_jobs_use_python_sysmon_core() -> None:
    workflow = _workflow()

    for job_name in ("core-lint-test-build", "external-integration"):
        assert workflow["jobs"][job_name]["env"]["COVERAGE_CORE"] == "sysmon"


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


def test_scoped_core_lanes_run_workflow_contract_before_target_resolution() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["core-lint-test-build"]["steps"]
    test_step = next(step for step in steps if step.get("name") == "Tests (core lane) + coverage report")
    run_script = str(test_step["run"])
    match = re.search(r"invariant_targets=\(\n(?P<targets>.*?)\n\s*\)", run_script, flags=re.DOTALL)

    assert match is not None, "core lane must declare invariant_targets as a shell array"
    invariant_targets = [line.strip().strip("\"'") for line in match.group("targets").splitlines() if line.strip()]
    assert invariant_targets == ["src/dnadesign/devtools/tests/ci/test_workflow_contract.py"]
    invariant_run = 'uv run pytest -q --durations=25 "${invariant_targets[@]}"'
    assert invariant_run in run_script
    assert run_script.index(invariant_run) < run_script.index('affected_csv="')


def test_core_lane_declared_test_targets_exist() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["core-lint-test-build"]["steps"]
    test_step = next(step for step in steps if step.get("name") == "Tests (core lane) + coverage report")
    run_script = str(test_step["run"])

    targets: list[str] = []
    for array_name in ("invariant_targets", "smoke_targets"):
        match = re.search(rf"{array_name}=\(\n(?P<targets>.*?)\n\s*\)", run_script, flags=re.DOTALL)
        assert match is not None, f"core lane must declare {array_name} as a shell array"
        targets.extend(line.strip().strip("\"'") for line in match.group("targets").splitlines() if line.strip())

    assert targets, "core lane declared test targets must not be empty"
    missing = [target for target in targets if not (_repo_root() / target).is_file()]
    assert missing == [], f"core lane declared test targets do not exist: {missing}"


def test_core_pytest_collects_the_complete_failure_set() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["core-lint-test-build"]["steps"]
    test_step = next(step for step in steps if step.get("name") == "Tests (core lane) + coverage report")
    run_script = str(test_step["run"])

    assert "--maxfail" not in run_script
    assert re.search(r"(^|\s)-x(?:\s|$)", run_script) is None


def test_core_installs_ffmpeg_only_after_static_gates() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["core-lint-test-build"]["steps"]
    step_index = {step.get("name"): index for index, step in enumerate(steps)}

    assert step_index["Format check"] < step_index["Install FFmpeg"] < step_index["Tests (core lane) + coverage report"]
