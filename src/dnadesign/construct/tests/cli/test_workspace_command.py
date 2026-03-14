"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/cli/test_workspace_command.py

Workspace command contracts for construct CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.construct.cli import app

_RUNNER = CliRunner()


def test_workspace_where_uses_env_root_when_set(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    monkeypatch.setenv("CONSTRUCT_WORKSPACE_ROOT", root.as_posix())

    result = _RUNNER.invoke(app, ["workspace", "where"])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert f"workspace_root: {root.resolve()}" in output
    assert "workspace_root_source: env" in output


def test_workspace_init_creates_default_layout_and_config(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix()])

    workspace_dir = root / "demo_construct"
    assert result.exit_code == 0, result.stdout
    assert (workspace_dir / "config.yaml").is_file()
    assert (workspace_dir / "inputs" / "README.md").is_file()
    assert (workspace_dir / "inputs" / "anchor_manifest.template.yaml").is_file()
    assert (workspace_dir / "inputs" / "template.fa").is_file()
    assert (workspace_dir / "outputs" / "usr_datasets").is_dir()
    assert (workspace_dir / "outputs" / "logs" / "ops" / "audit").is_dir()
    output = result.stdout or ""
    assert "construct validate config --config" in output
    assert "construct run --config" in output


def test_workspace_init_rejects_path_like_workspace_id(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "bad/name", "--root", root.as_posix()])

    assert result.exit_code == 2
    assert "workspace id must be a simple directory name" in (result.stdout or "")


def test_workspace_init_fails_if_workspace_already_exists(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    workspace_dir = root / "demo_construct"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix()])

    assert result.exit_code == 2
    assert "workspace already exists" in (result.stdout or "")
