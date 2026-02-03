"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_cli_workspace_init.py

Workspace init and Stage-B guardrail tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app


def _write_template_config(path: Path) -> None:
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
                  path: inputs/sites.csv
            """
        ).strip()
        + "\n"
    )


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
                      include_inputs: [demo]
                    regulator_constraints:
                      groups: []

              solver:
                backend: CBC
                strategy: iterate

              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n"
    )


def test_workspace_init_warns_on_relative_inputs_without_copy(tmp_path: Path) -> None:
    template_path = tmp_path / "template.yaml"
    _write_template_config(template_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workspace",
            "init",
            "--id",
            "demo_run",
            "--root",
            str(tmp_path),
            "--template",
            str(template_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Workspace uses file-based inputs with relative paths" in result.output
    assert (tmp_path / "demo_run" / "config.yaml").exists()


def test_stage_b_reports_missing_pool_manifest(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_min_config(cfg_path)
    pool_dir = tmp_path / "outputs" / "pools"
    pool_dir.mkdir(parents=True)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "stage-b",
            "build-libraries",
            "-c",
            str(cfg_path),
            "--pool",
            str(pool_dir),
        ],
    )
    assert result.exit_code != 0, result.output
    assert "Pool manifest not found" in result.output
    normalized = " ".join(result.output.split())
    assert "dense stage-a build-pool" in normalized
