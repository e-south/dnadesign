# ABOUTME: CLI coverage for Stage-A build-pool length summaries.
# ABOUTME: Ensures pooled TFBS length stats are surfaced in stdout.
from __future__ import annotations

import textwrap
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app


def test_stage_a_build_pool_reports_length_summary(tmp_path: Path) -> None:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    (inputs_dir / "sites.csv").write_text(
        textwrap.dedent(
            """
            tf,tfbs
            TF1,AAAAAAAAAA
            TF2,CCCCCCCCCCCC
            TF3,GGGGGGGGGGGGGG
            """
        ).strip()
        + "\n"
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            densegen:
              schema_version: "2.5"
              run:
                id: demo
                root: "."
              inputs:
                - name: toy_sites
                  type: binding_sites
                  path: {inputs_dir / "sites.csv"}
                  format: csv
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/dense_arrays.parquet
              generation:
                sequence_length: 30
                quota: 1
                plan:
                  - name: default
                    quota: 1
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n"
    )
    runner = CliRunner()
    result = runner.invoke(app, ["stage-a", "build-pool", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "TFBS length summary" in result.output
    assert "toy_sites" in result.output
