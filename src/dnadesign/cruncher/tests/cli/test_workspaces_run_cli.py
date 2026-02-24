"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_workspaces_run_cli.py

CLI contract tests for `cruncher workspaces run`.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml
from typer.testing import CliRunner

import dnadesign.cruncher.cli.commands.workspaces as workspaces_cli
from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def _write_workspace(tmp_path: Path, name: str) -> Path:
    workspace = tmp_path / "workspaces" / name
    config = workspace / "configs" / "config.yaml"
    runbook = workspace / "configs" / "runbook.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("cruncher: {schema_version: 3, workspace: {out_dir: outputs, regulator_sets: [[lexA,cpxR]]}}\n")
    runbook.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": name,
                    "steps": [{"id": "lock", "run": ["lock", "-c", "configs/config.yaml"]}],
                }
            }
        )
    )
    return workspace


def _write_runbook_only_workspace(tmp_path: Path, name: str) -> Path:
    workspace = tmp_path / "workspaces" / name
    runbook = workspace / "configs" / "runbook.yaml"
    runbook.parent.mkdir(parents=True, exist_ok=True)
    runbook.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": name,
                    "steps": [{"id": "portfolio_run", "run": ["portfolio", "run", "--spec", "configs/p.yaml"]}],
                }
            }
        )
    )
    return workspace


def test_workspaces_run_resolves_runbook_from_workspace_selector(tmp_path: Path, monkeypatch) -> None:
    workspace = _write_workspace(tmp_path, "demo")
    captured: dict[str, object] = {}

    def _fake_run_workspace_runbook(path: Path, *, step_ids=None, dry_run=False):
        captured["path"] = path
        captured["step_ids"] = list(step_ids or [])
        captured["dry_run"] = dry_run
        return SimpleNamespace(
            runbook_path=path,
            workspace_root=path.parent.parent,
            executed_step_ids=list(step_ids or ["lock"]),
        )

    monkeypatch.setattr(workspaces_cli, "run_workspace_runbook", _fake_run_workspace_runbook)

    result = runner.invoke(
        app,
        ["workspaces", "run", "--workspace", "demo", "--step", "lock", "--dry-run"],
        env={"CRUNCHER_WORKSPACE_ROOTS": str(tmp_path / "workspaces"), "CRUNCHER_NONINTERACTIVE": "1"},
    )

    assert result.exit_code == 0
    assert captured["path"] == (workspace / "configs" / "runbook.yaml").resolve()
    assert captured["step_ids"] == ["lock"]
    assert captured["dry_run"] is True


def test_workspaces_run_resolves_relative_runbook_against_workspace_selector(tmp_path: Path, monkeypatch) -> None:
    workspace = _write_workspace(tmp_path, "demo")
    captured: dict[str, object] = {}

    def _fake_run_workspace_runbook(path: Path, *, step_ids=None, dry_run=False):
        captured["path"] = path
        captured["step_ids"] = list(step_ids or [])
        captured["dry_run"] = dry_run
        return SimpleNamespace(
            runbook_path=path,
            workspace_root=path.parent.parent,
            executed_step_ids=list(step_ids or ["lock"]),
        )

    monkeypatch.setattr(workspaces_cli, "run_workspace_runbook", _fake_run_workspace_runbook)

    result = runner.invoke(
        app,
        ["workspaces", "run", "--workspace", "demo", "--runbook", "configs/runbook.yaml", "--dry-run"],
        env={
            "CRUNCHER_WORKSPACE_ROOTS": str(tmp_path / "workspaces"),
            "CRUNCHER_CWD": str(tmp_path),
            "CRUNCHER_NONINTERACTIVE": "1",
        },
    )

    assert result.exit_code == 0
    assert captured["path"] == (workspace / "configs" / "runbook.yaml").resolve()
    assert captured["step_ids"] == []
    assert captured["dry_run"] is True


def test_workspaces_run_rejects_relative_runbook_that_escapes_workspace(tmp_path: Path, monkeypatch) -> None:
    _write_workspace(tmp_path, "demo")
    captured: dict[str, object] = {}

    def _fake_run_workspace_runbook(path: Path, *, step_ids=None, dry_run=False):
        captured["path"] = path
        return SimpleNamespace(
            runbook_path=path,
            workspace_root=path.parent.parent,
            executed_step_ids=list(step_ids or ["lock"]),
        )

    monkeypatch.setattr(workspaces_cli, "run_workspace_runbook", _fake_run_workspace_runbook)

    result = runner.invoke(
        app,
        ["workspaces", "run", "--workspace", "demo", "--runbook", "../other_runbook.yaml", "--dry-run"],
        env={
            "CRUNCHER_WORKSPACE_ROOTS": str(tmp_path / "workspaces"),
            "CRUNCHER_CWD": str(tmp_path),
            "CRUNCHER_NONINTERACTIVE": "1",
        },
    )

    assert result.exit_code == 1
    assert "Relative --runbook must resolve inside the selected workspace" in result.output
    assert "path" not in captured


def test_workspaces_run_defaults_to_cwd_runbook_for_runbook_only_workspace(tmp_path: Path, monkeypatch) -> None:
    workspace = _write_runbook_only_workspace(tmp_path, "portfolio_tmp")
    captured: dict[str, object] = {}

    def _fake_run_workspace_runbook(path: Path, *, step_ids=None, dry_run=False):
        captured["path"] = path
        captured["step_ids"] = list(step_ids or [])
        captured["dry_run"] = dry_run
        return SimpleNamespace(
            runbook_path=path,
            workspace_root=path.parent.parent,
            executed_step_ids=list(step_ids or ["portfolio_run"]),
        )

    monkeypatch.setattr(workspaces_cli, "run_workspace_runbook", _fake_run_workspace_runbook)

    result = runner.invoke(
        app,
        ["workspaces", "run", "--dry-run"],
        env={"CRUNCHER_CWD": str(workspace), "CRUNCHER_NONINTERACTIVE": "1"},
    )

    assert result.exit_code == 0
    assert captured["path"] == (workspace / "configs" / "runbook.yaml").resolve()
    assert captured["step_ids"] == []
    assert captured["dry_run"] is True


def test_workspaces_run_defaults_to_cwd_workspace_runbook(tmp_path: Path, monkeypatch) -> None:
    workspace = _write_workspace(tmp_path, "demo")
    captured: dict[str, object] = {}

    def _fake_run_workspace_runbook(path: Path, *, step_ids=None, dry_run=False):
        captured["path"] = path
        return SimpleNamespace(
            runbook_path=path,
            workspace_root=path.parent.parent,
            executed_step_ids=list(step_ids or ["lock"]),
        )

    monkeypatch.setattr(workspaces_cli, "run_workspace_runbook", _fake_run_workspace_runbook)

    result = runner.invoke(
        app,
        ["workspaces", "run"],
        env={"CRUNCHER_CWD": str(workspace), "CRUNCHER_NONINTERACTIVE": "1"},
    )

    assert result.exit_code == 0
    assert captured["path"] == (workspace / "configs" / "runbook.yaml").resolve()


def test_workspaces_run_resolves_runbook_only_workspace_by_name(tmp_path: Path, monkeypatch) -> None:
    _write_workspace(tmp_path, "demo")
    workspace = _write_runbook_only_workspace(tmp_path, "portfolio_tmp")
    captured: dict[str, object] = {}

    def _fake_run_workspace_runbook(path: Path, *, step_ids=None, dry_run=False):
        captured["path"] = path
        captured["dry_run"] = dry_run
        return SimpleNamespace(
            runbook_path=path,
            workspace_root=path.parent.parent,
            executed_step_ids=list(step_ids or ["portfolio_run"]),
        )

    monkeypatch.setattr(workspaces_cli, "run_workspace_runbook", _fake_run_workspace_runbook)

    result = runner.invoke(
        app,
        ["workspaces", "run", "--workspace", "portfolio_tmp", "--dry-run"],
        env={"CRUNCHER_WORKSPACE_ROOTS": str(tmp_path / "workspaces"), "CRUNCHER_NONINTERACTIVE": "1"},
    )

    assert result.exit_code == 0
    assert captured["path"] == (workspace / "configs" / "runbook.yaml").resolve()
    assert captured["dry_run"] is True


def test_workspaces_list_includes_runbook_only_workspaces(tmp_path: Path) -> None:
    _write_workspace(tmp_path, "demo")
    _write_runbook_only_workspace(tmp_path, "portfolio_tmp")

    result = runner.invoke(
        app,
        ["workspaces", "list"],
        env={
            "CRUNCHER_WORKSPACE_ROOTS": str(tmp_path / "workspaces"),
            "CRUNCHER_NONINTERACTIVE": "1",
            "COLUMNS": "240",
        },
    )

    assert result.exit_code == 0
    assert "portfolio_tmp" in result.output
    assert "runbook-only" in result.output


def test_workspaces_list_supports_explicit_root_option(tmp_path: Path) -> None:
    _write_workspace(tmp_path, "demo")
    _write_runbook_only_workspace(tmp_path, "portfolio_tmp")
    root = (tmp_path / "workspaces").resolve()

    result = runner.invoke(
        app,
        ["workspaces", "list", "--root", str(root)],
        env={
            "CRUNCHER_NONINTERACTIVE": "1",
            "COLUMNS": "240",
        },
    )

    assert result.exit_code == 0
    assert "demo" in result.output
    assert "portfolio_tmp" in result.output
