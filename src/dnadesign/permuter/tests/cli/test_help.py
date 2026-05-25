"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/cli/test_help.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typer.testing import CliRunner

from dnadesign.permuter.src.cli.app import app


def test_cli_help_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--workspace" in result.output
    assert "--job" not in result.output


def test_plot_help_does_not_advertise_internal_helpers() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["plot", "--help"])
    assert result.exit_code == 0
    assert "window_score_mass" not in result.output
    assert "ranked_variants" in result.output
