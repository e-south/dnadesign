"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_workspaces_reset_cli.py

CLI contract tests for `cruncher workspaces reset`.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def _seed_workspace(root: Path) -> None:
    configs = root / "configs"
    inputs = root / "inputs" / "local_motifs"
    outputs = root / "outputs" / "analysis"
    cache = root / ".cruncher" / "locks"
    pycache = root / "outputs" / "__pycache__"

    configs.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    pycache.mkdir(parents=True, exist_ok=True)

    (configs / "config.yaml").write_text("cruncher: {schema_version: 3, workspace: {out_dir: outputs}}\n")
    (configs / "runbook.yaml").write_text("runbook: {schema_version: 1, name: demo, steps: []}\n")
    (configs / "studies.yaml").write_text("study: {}\n")
    (inputs / "lexA.txt").write_text("MEME version 5\n")
    (root / "runbook.md").write_text("# demo\n")
    (outputs / "artifact.txt").write_text("artifact\n")
    (cache / "config.lock.json").write_text("{}\n")
    (pycache / "cached.pyc").write_bytes(b"pyc")
    (root / ".DS_Store").write_text("ds\n")


def test_workspaces_reset_dry_run_reports_targets_without_deleting(tmp_path: Path) -> None:
    workspace = tmp_path / "demo"
    _seed_workspace(workspace)

    result = runner.invoke(
        app,
        ["workspaces", "reset", "--root", str(workspace)],
        env={"CRUNCHER_NONINTERACTIVE": "1"},
    )

    assert result.exit_code == 0
    assert "Dry run only. Re-run with --confirm to reset workspace state." in result.output
    assert (workspace / "outputs").exists()
    assert (workspace / ".cruncher").exists()
    assert (workspace / "inputs").exists()
    assert (workspace / "configs").exists()


def test_workspaces_reset_confirm_removes_generated_state_and_preserves_inputs_and_configs(tmp_path: Path) -> None:
    workspace = tmp_path / "demo"
    _seed_workspace(workspace)

    result = runner.invoke(
        app,
        ["workspaces", "reset", "--root", str(workspace), "--confirm"],
        env={"CRUNCHER_NONINTERACTIVE": "1"},
    )

    assert result.exit_code == 0
    assert not (workspace / "outputs").exists()
    assert not (workspace / ".cruncher").exists()
    assert not (workspace / ".DS_Store").exists()
    assert (workspace / "inputs" / "local_motifs" / "lexA.txt").exists()
    assert (workspace / "configs" / "config.yaml").exists()
    assert (workspace / "configs" / "runbook.yaml").exists()
    assert (workspace / "runbook.md").exists()


def test_workspaces_reset_requires_workspace_layout(tmp_path: Path) -> None:
    not_workspace = tmp_path / "not_workspace"
    not_workspace.mkdir(parents=True, exist_ok=True)
    (not_workspace / "random.txt").write_text("x\n")

    result = runner.invoke(
        app,
        ["workspaces", "reset", "--root", str(not_workspace)],
        env={"CRUNCHER_NONINTERACTIVE": "1"},
    )

    assert result.exit_code == 1
    assert "Workspace root must contain configs/runbook.yaml or configs/config.yaml" in result.output


def test_workspaces_reset_all_workspaces_dry_run_reports_targets_without_deleting(tmp_path: Path) -> None:
    workspaces_root = tmp_path / "workspaces"
    workspace_a = workspaces_root / "demo_a"
    workspace_b = workspaces_root / "demo_b"
    non_workspace = workspaces_root / "notes"
    _seed_workspace(workspace_a)
    _seed_workspace(workspace_b)
    non_workspace.mkdir(parents=True, exist_ok=True)
    (non_workspace / "readme.txt").write_text("notes\n")

    result = runner.invoke(
        app,
        ["workspaces", "reset", "--root", str(workspaces_root), "--all-workspaces"],
        env={"CRUNCHER_NONINTERACTIVE": "1"},
    )

    assert result.exit_code == 0
    assert "Workspace reset root set:" in result.output
    assert "demo_a" in result.output
    assert "demo_b" in result.output
    assert "Dry run only. Re-run with --confirm to reset workspace state." in result.output
    assert (workspace_a / "outputs").exists()
    assert (workspace_b / "outputs").exists()


def test_workspaces_reset_all_workspaces_confirm_removes_generated_state(tmp_path: Path) -> None:
    workspaces_root = tmp_path / "workspaces"
    workspace_a = workspaces_root / "demo_a"
    workspace_b = workspaces_root / "demo_b"
    _seed_workspace(workspace_a)
    _seed_workspace(workspace_b)

    result = runner.invoke(
        app,
        ["workspaces", "reset", "--root", str(workspaces_root), "--all-workspaces", "--confirm"],
        env={"CRUNCHER_NONINTERACTIVE": "1"},
    )

    assert result.exit_code == 0
    assert not (workspace_a / "outputs").exists()
    assert not (workspace_a / ".cruncher").exists()
    assert not (workspace_b / "outputs").exists()
    assert not (workspace_b / ".cruncher").exists()
    assert (workspace_a / "configs" / "config.yaml").exists()
    assert (workspace_b / "configs" / "config.yaml").exists()
