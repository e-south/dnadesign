"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_report_plots.py

CLI guardrails for DenseGen notebook command surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app
from dnadesign.densegen.tests.config_fixtures import write_minimal_config


def test_report_command_removed() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code != 0
    assert "No such command 'report'" in result.output


def test_notebook_generate_writes_workspace_notebook(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    assert notebook_path.exists()
    content = notebook_path.read_text()
    assert "DenseGen Run Notebook" in content
    assert 'run_root / "outputs" / "tables" / "records.parquet"' in content
    assert 'run_root / "outputs" / "tables" / "dense_arrays.parquet"' in content


def test_notebook_run_requires_existing_notebook(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "run", "-c", str(cfg_path)])
    assert result.exit_code == 1
    assert "No notebook found" in result.output
    assert "outputs/notebooks/densegen_run_overview.py" in result.output


def test_notebook_generate_passes_marimo_check(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    assert notebook_path.exists()

    check_result = subprocess.run(
        [sys.executable, "-m", "marimo", "check", str(notebook_path)],
        capture_output=True,
        text=True,
    )
    assert check_result.returncode == 0, check_result.stdout + check_result.stderr
    assert "warning[" not in check_result.stdout
