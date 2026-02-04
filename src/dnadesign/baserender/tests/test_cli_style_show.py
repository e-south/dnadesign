"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_cli_style_show.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typer.testing import CliRunner

from dnadesign.baserender.src.cli import app


def test_cli_style_show_default_includes_presentation_defaults():
    runner = CliRunner()
    res = runner.invoke(app, ["style", "show"])
    assert res.exit_code == 0
    # Key from presentation_default.yml
    assert "track_spacing: 35" in res.stdout


def test_cli_style_show_presentation_default_matches_default():
    runner = CliRunner()
    res_default = runner.invoke(app, ["style", "show"])
    res_preset = runner.invoke(app, ["style", "show", "--preset", "presentation_default"])
    assert res_default.exit_code == 0
    assert res_preset.exit_code == 0
    assert res_default.stdout == res_preset.stdout
