"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/tests/cli/test_help.py

Regression tests for help Permuter CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typer.main import get_command
from typer.testing import CliRunner

from dnadesign.permuter.src.cli.app import app


def test_cli_help_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "workspace" in result.output
    assert "--job" not in result.output

    click_app = get_command(app)
    run_options = {opt for param in click_app.commands["run"].params for opt in param.opts}
    assert "--workspace" in run_options
    assert "--job" not in run_options


def test_plot_help_does_not_advertise_internal_helpers() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["plot", "--help"])
    assert result.exit_code == 0
    assert "window_score_mass" not in result.output
    assert "ranked_variants" in result.output
