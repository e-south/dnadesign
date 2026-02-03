"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_cli_config_option.py

CLI config flag handling tests for DenseGen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import textwrap
from contextlib import contextmanager
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app


@contextmanager
def _chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _write_min_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo
                  type: binding_sites
                  path: inputs.csv

              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/dense_arrays.parquet

              generation:
                sequence_length: 10
                quota: 1
                plan:
                  - name: default
                    quota: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups:
                        - name: all
                          members: [TF1]
                          min_required: 1

              solver:
                backend: CBC
                strategy: iterate

              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n"
    )


def test_validate_accepts_config_after_command(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_min_config(cfg_path)
    runner = CliRunner()
    result = runner.invoke(app, ["validate-config", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "Config is valid" in result.output


def test_validate_reports_invalid_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("densegen:\n  inputs: []\n")
    runner = CliRunner()
    result = runner.invoke(app, ["validate-config", "-c", str(cfg_path)])
    assert result.exit_code != 0, result.output
    assert "Config error" in result.output


def test_validate_uses_cwd_config_when_missing_flag(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_min_config(cfg_path)
    with _chdir(tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["validate-config"])
        assert result.exit_code == 0, result.output
        assert "Config is valid" in result.output


def test_validate_missing_cwd_config_reports_actionable_error(tmp_path: Path) -> None:
    with _chdir(tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["validate-config"])
        assert result.exit_code != 0, result.output
        assert "No config found" in result.output
