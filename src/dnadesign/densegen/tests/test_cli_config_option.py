from __future__ import annotations

import textwrap
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app


def _write_min_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.4"
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
                  path: outputs/dense_arrays.parquet

              generation:
                sequence_length: 10
                quota: 1
                plan:
                  - name: default
                    quota: 1

              solver:
                backend: CBC
                strategy: iterate

              logging:
                log_dir: logs
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
