"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_operations_contracts.py

Tests for documentation operations contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from dnadesign.devtools.docs.checks import (
    _find_legacy_contract_surface_doc_issues,
    _find_operational_runbook_path_issues,
    _find_ops_deprecated_semantics_issues,
    _find_runbook_demo_snippet_issues,
    _find_shared_utils_path_issues,
    _find_stale_overlay_guard_term_issues,
    _find_transient_operational_artifact_path_issues,
    main,
)
from dnadesign.devtools.tests.docs.check_test_support import (
    _git_add,
    _git_init,
    _write,
)


def test_runbook_demo_snippet_check_flags_missing_shell_and_yaml_comments(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "tutorials" / "demo.md",
        "\n".join(
            [
                "## Demo",
                "",
                "```bash",
                "uv run alpha do-work",
                "```",
                "",
                "```yaml",
                "alpha:",
                "  enabled: true",
                "```",
                "",
            ]
        ),
    )

    issues = _find_runbook_demo_snippet_issues(tmp_path)

    assert any("command in shell block needs an explanatory comment" in issue for issue in issues)
    assert any("yaml key/value in runbook/demo snippets needs a right-side inline comment" in issue for issue in issues)


def test_runbook_demo_snippet_check_accepts_commented_shell_and_yaml_blocks(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "tutorials" / "demo.md",
        "\n".join(
            [
                "## Demo",
                "",
                "```bash",
                "# Run the demo command.",
                "uv run alpha do-work",
                "```",
                "",
                "```yaml",
                "alpha:",
                "  enabled: true  # Toggle demo mode.",
                "```",
                "",
            ]
        ),
    )

    issues = _find_runbook_demo_snippet_issues(tmp_path)

    assert issues == []


def test_find_operational_runbook_path_issues_flags_repo_root_runbook(tmp_path: Path) -> None:
    _write(
        tmp_path / "stress_ethanol_cipro.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: study_stress_ethanol_cipro",
                "  workflow_id: densegen_batch_submit",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
                "  logging:",
                "    stdout_dir: /tmp/workspace/outputs/logs/ops/sge/study_stress_ethanol_cipro",
                "  densegen:",
                "    config: /tmp/workspace/config.yaml",
                "    qsub_template: docs/bu-scc/jobs/densegen-cpu.qsub",
                "  resources:",
                "    pe_omp: 16",
                "    h_rt: 08:00:00",
                "    mem_per_core: 8G",
            ]
        )
        + "\n",
    )
    _git_init(tmp_path)
    _git_add(tmp_path, "stress_ethanol_cipro.yaml")

    issues = _find_operational_runbook_path_issues(tmp_path)

    assert any("operational runbook path is outside allowed locations" in issue for issue in issues)


def test_find_operational_runbook_path_issues_rejects_malformed_tracked_yaml(tmp_path: Path) -> None:
    _write(tmp_path / "broken.yaml", "runbook:\n  workflow_id: [broken\n")
    _git_init(tmp_path)
    _git_add(tmp_path, "broken.yaml")

    with pytest.raises(ValueError, match="operational runbook yaml is invalid"):
        _find_operational_runbook_path_issues(tmp_path)


def test_find_operational_runbook_path_issues_ignores_untracked_yaml_noise_in_git_repo(tmp_path: Path) -> None:
    _write(
        tmp_path / "stress_ethanol_cipro.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: study_stress_ethanol_cipro",
                "  workflow_id: densegen_batch_submit",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
                "  logging:",
                "    stdout_dir: /tmp/workspace/outputs/logs/ops/sge/study_stress_ethanol_cipro",
                "  densegen:",
                "    config: /tmp/workspace/config.yaml",
                "    qsub_template: docs/bu-scc/jobs/densegen-cpu.qsub",
                "  resources:",
                "    pe_omp: 16",
                "    h_rt: 08:00:00",
                "    mem_per_core: 8G",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "scratch" / "nested" / "noise.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: generated_noise",
                "  workflow_id: densegen_batch_submit",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
            ]
        )
        + "\n",
    )
    _git_init(tmp_path)
    _git_add(tmp_path, "stress_ethanol_cipro.yaml")

    issues = _find_operational_runbook_path_issues(tmp_path)

    assert any("stress_ethanol_cipro.yaml" in issue for issue in issues)
    assert not any("scratch/nested/noise.yaml" in issue for issue in issues)


def test_find_operational_runbook_path_issues_allows_packaged_presets(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "runbooks" / "presets" / "densegen_demo.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: study_stress_ethanol_cipro",
                "  workflow_id: densegen_batch_submit",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
                "  logging:",
                "    stdout_dir: /tmp/workspace/outputs/logs/ops/sge/study_stress_ethanol_cipro",
                "  densegen:",
                "    config: /tmp/workspace/config.yaml",
                "    qsub_template: docs/bu-scc/jobs/densegen-cpu.qsub",
                "  resources:",
                "    pe_omp: 16",
                "    h_rt: 08:00:00",
                "    mem_per_core: 8G",
            ]
        )
        + "\n",
    )

    issues = _find_operational_runbook_path_issues(tmp_path)

    assert issues == []


def test_find_operational_runbook_path_issues_allows_workspace_runbooks_dir(tmp_path: Path) -> None:
    _write(
        tmp_path / "workspace" / "outputs" / "logs" / "ops" / "runbooks" / "densegen_demo.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: densegen_demo",
                "  workflow_id: densegen_batch_submit",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
                "  logging:",
                "    stdout_dir: /tmp/workspace/outputs/logs/ops/sge/densegen_demo",
            ]
        )
        + "\n",
    )

    issues = _find_operational_runbook_path_issues(tmp_path)

    assert issues == []


def test_find_operational_runbook_path_issues_skips_generated_output_yaml_noise(tmp_path: Path) -> None:
    _write(
        tmp_path / "workspace" / "outputs" / "usr_datasets" / "registry.yaml",
        "\n".join(
            [
                "runbook:",
                "  schema_version: 1",
                "  id: generated_noise",
                "  workflow_id: densegen_batch_submit",
                "  project: dunlop",
                "  workspace_root: /tmp/workspace",
            ]
        )
        + "\n",
    )

    issues = _find_operational_runbook_path_issues(tmp_path)

    assert issues == []


def test_find_shared_utils_path_issues_flags_top_level_utils_package(tmp_path: Path) -> None:
    disallowed_utils_path = tmp_path / "src" / "dnadesign" / "utils"
    disallowed_utils_path.mkdir(parents=True, exist_ok=True)

    issues = _find_shared_utils_path_issues(tmp_path)

    assert any("shared utils package is not allowed" in issue for issue in issues)


def test_find_shared_utils_path_issues_allows_tool_local_utils(tmp_path: Path) -> None:
    allowed_tool_utils_path = tmp_path / "src" / "dnadesign" / "densegen" / "src" / "utils"
    allowed_tool_utils_path.mkdir(parents=True, exist_ok=True)

    issues = _find_shared_utils_path_issues(tmp_path)

    assert issues == []


def test_find_transient_operational_artifact_path_issues_flags_repo_root_codex_tmp(tmp_path: Path) -> None:
    _write(tmp_path / ".codex_tmp" / "audit_notify" / "records.parquet", "placeholder\n")

    issues = _find_transient_operational_artifact_path_issues(tmp_path)

    assert any("transient operational artifact directory is not allowed at repo root" in issue for issue in issues)


def test_find_transient_operational_artifact_path_issues_flags_repo_root_outputs(tmp_path: Path) -> None:
    _write(tmp_path / "outputs" / "thread" / "artifact.yaml", "placeholder\n")

    issues = _find_transient_operational_artifact_path_issues(tmp_path)

    assert any("generated artifact directory is not allowed at repository root" in issue for issue in issues)


def test_find_transient_operational_artifact_path_issues_allows_workspace_nested_temp_dirs(tmp_path: Path) -> None:
    _write(
        tmp_path
        / "src"
        / "dnadesign"
        / "densegen"
        / "workspaces"
        / "study"
        / "outputs"
        / "tmp"
        / ".codex_tmp"
        / "state.json",
        "{}\n",
    )

    issues = _find_transient_operational_artifact_path_issues(tmp_path)

    assert issues == []


def test_main_fails_when_repo_root_contains_transient_operational_dir(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(tmp_path / ".tmp_ops" / "scratch.log", "placeholder\n")

    rc = main(["--repo-root", str(tmp_path)])

    assert rc == 1


def test_find_stale_overlay_guard_term_issues_flags_old_ops_guard_terms(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "orchestration/runbooks.md",
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Owner:** maintainers",
                f"**Last verified:** {dt.date.today().isoformat()}",
                "",
                "Use densegen-overlay-guard with densegen.overlay_guard.namespace.",
                "",
            ]
        )
        + "\n",
    )

    issues = _find_stale_overlay_guard_term_issues(tmp_path)

    assert any("densegen-overlay-guard" in issue for issue in issues)
    assert any("densegen.overlay_guard.namespace" in issue for issue in issues)


def test_find_stale_overlay_guard_term_issues_accepts_usr_overlay_guard_terms(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "orchestration/runbooks.md",
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Owner:** maintainers",
                f"**Last verified:** {dt.date.today().isoformat()}",
                "",
                "Use usr-overlay-guard with densegen.overlay_guard.overlay_namespace.",
                "",
            ]
        )
        + "\n",
    )

    issues = _find_stale_overlay_guard_term_issues(tmp_path)

    assert issues == []


def test_ops_deprecated_semantics_check_flags_legacy_terms(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "orchestration/runbooks.md",
        "\n".join(
            [
                "## Ops runbook",
                "",
                "Use `densegen_batch_with_notify_slack`.",
                "",
                "The precedents surface remains available.",
                "Use infer_local_runtime and notify_profile_doctor.",
                "Read notify.profile.*.details.setup_command after infer_validate_config.",
            ]
        )
        + "\n",
    )

    issues = _find_ops_deprecated_semantics_issues(tmp_path)

    assert any("with_notify_slack" in issue for issue in issues)
    assert any("precedents" in issue for issue in issues)
    assert any("infer_local_runtime" in issue for issue in issues)
    assert any("notify_profile_doctor" in issue for issue in issues)
    assert any("details.setup_command" in issue for issue in issues)


def test_legacy_contract_surface_docs_check_flags_repo_root_contract_references(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "README.md", "## Docs\n\nUse `dnadesign._contracts` and `src/dnadesign/usr_roots.py`.\n")

    issues = _find_legacy_contract_surface_doc_issues(tmp_path)

    assert any("dnadesign._contracts" in issue for issue in issues)
    assert any("src/dnadesign/usr_roots.py" in issue for issue in issues)
