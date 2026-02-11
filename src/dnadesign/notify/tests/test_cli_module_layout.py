"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_module_layout.py

Module layout contract tests for Notify CLI command decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib

from typer.testing import CliRunner

from dnadesign.notify.cli import app


def test_notify_cli_command_modules_importable() -> None:
    assert importlib.import_module("dnadesign.notify.cli_commands.usr_events")
    assert importlib.import_module("dnadesign.notify.cli_commands.spool")
    assert importlib.import_module("dnadesign.notify.cli_commands.profile")
    assert importlib.import_module("dnadesign.notify.cli_commands.providers")


def test_notify_help_lists_usr_events_and_spool_groups() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "usr-events" in result.stdout
    assert "spool" in result.stdout
    assert "profile" in result.stdout
