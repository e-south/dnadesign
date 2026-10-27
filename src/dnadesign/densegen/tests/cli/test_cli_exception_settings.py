"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_exception_settings.py

Regression tests for CLI exception settings DenseGen CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.cli import main as cli


def test_typer_apps_disable_local_variable_tracebacks() -> None:
    assert cli.app.pretty_exceptions_show_locals is False
    assert cli.inspect_app.pretty_exceptions_show_locals is False
    assert cli.stage_a_app.pretty_exceptions_show_locals is False
    assert cli.stage_b_app.pretty_exceptions_show_locals is False
    assert cli.workspace_app.pretty_exceptions_show_locals is False
    assert cli.notebook_app.pretty_exceptions_show_locals is False
