"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_cli_command_surface.py

Ensure top-level CLI surface excludes tool-specific commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typer.testing import CliRunner

from dnadesign.usr.src.cli import app


def test_top_level_excludes_tool_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    help_text = result.stdout
    assert "convert-legacy" not in help_text
    assert "make-mock" not in help_text
    assert "add-demo-cols" not in help_text
    assert "repair-densegen" not in help_text
    assert "merge-datasets" not in help_text
    assert "dedupe-sequences" not in help_text
    assert "plot" not in help_text

    assert "legacy" in help_text
    assert "densegen" in help_text
    assert "maintenance" in help_text
    assert "namespace" in help_text
    assert "events" in help_text
    assert "state" in help_text
