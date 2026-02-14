"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_secret_ref_help.py

CLI help contract tests for secret_ref backend hints in Notify commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typer.testing import CliRunner

from dnadesign.notify.cli import app


def test_notify_help_secret_ref_mentions_file_backend() -> None:
    runner = CliRunner()
    for args in (
        ["send", "--help"],
        ["usr-events", "watch", "--help"],
        ["spool", "drain", "--help"],
    ):
        result = runner.invoke(app, list(args))
        assert result.exit_code == 0
        assert "file:///path" in result.stdout
