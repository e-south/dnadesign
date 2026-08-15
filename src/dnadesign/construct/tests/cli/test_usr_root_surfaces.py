"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/cli/test_usr_root_surfaces.py

Tests the shared explicit USR coordinate on every Construct execution surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.construct.src.cli import app

_RUNNER = CliRunner()


@pytest.mark.parametrize(
    "command",
    (
        ["run", "--help"],
        ["validate", "config", "--help"],
        ["workspace", "run-project", "--help"],
        ["workspace", "validate-project", "--help"],
    ),
)
def test_construct_execution_surfaces_expose_the_same_usr_root_option(command: list[str]) -> None:
    result = _RUNNER.invoke(app, command)

    assert result.exit_code == 0, result.stdout
    assert "--usr-root" in result.stdout


def test_validate_config_rejects_usr_root_without_runtime(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("invalid: true\n", encoding="utf-8")

    result = _RUNNER.invoke(
        app,
        ["validate", "config", "--config", str(config), "--usr-root", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert result.stdout == "Error: --usr-root requires --runtime.\n"


def test_workspace_validate_project_rejects_usr_root_without_runtime(tmp_path: Path) -> None:
    result = _RUNNER.invoke(
        app,
        [
            "workspace",
            "validate-project",
            "--workspace",
            str(tmp_path / "missing-workspace"),
            "--project",
            "missing-project",
            "--usr-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert result.stdout == "Error: --usr-root requires --runtime.\n"
