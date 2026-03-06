"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/cli/test_adapters_commands.py

CLI adapter command contracts for infer bootstrap visibility.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typer.testing import CliRunner

from dnadesign.infer.cli import app

_RUNNER = CliRunner()


def test_adapters_list_reports_registered_default_model_ids() -> None:
    result = _RUNNER.invoke(app, ["adapters", "list"])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "evo2_7b" in output
    assert "evo2_1b_base" in output


def test_adapters_fns_reports_registered_default_namespaced_functions() -> None:
    result = _RUNNER.invoke(app, ["adapters", "fns"])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "evo2.logits" in output
    assert "evo2.embedding" in output
    assert "evo2.log_likelihood" in output
    assert "evo2.generate" in output

