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
              schema_version: "2.5"
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

              solver:
                strategy: approximate

              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n"
    )


def test_describe_outputs_summary(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_min_config(cfg_path)
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "config", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "Config" in result.output
    assert "Pad" in result.output
    assert "See `dense inspect inputs`" in result.output
