"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_describe.py

CLI describe command tests for DenseGen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

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
                  path: outputs/tables/records.parquet

              generation:
                sequence_length: 10
                plan:
                  - name: default
                    quota: 1
                    sampling:
                      include_inputs: [demo]
                    regulator_constraints:
                      groups: []

              solver:
                strategy: approximate

              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n"
    )


def _write_pwm_mmr_config(path: Path) -> None:
    inputs_dir = path.parent / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    meme_path = inputs_dir / "motifs.meme"
    meme_path.write_text(
        textwrap.dedent(
            """
            MEME version 4

            ALPHABET= ACGT

            Background letter frequencies
            A 0.25 C 0.25 G 0.25 T 0.25

            MOTIF M1
            letter-probability matrix: alength= 4 w= 3 nsites= 20 E= 0
            0.8 0.1 0.05 0.05
            0.1 0.7 0.1 0.1
            0.1 0.1 0.7 0.1
            """
        ).strip()
        + "\n"
    )
    path.write_text(
        textwrap.dedent(
            f"""
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_pwm
                  type: pwm_meme
                  path: {meme_path}
                  sampling:
                    n_sites: 3
                    mining:
                      batch_size: 10
                      budget:
                        mode: fixed_candidates
                        candidates: 100
                    selection:
                      policy: mmr
                      alpha: 0.5
                      pool:
                        min_score_norm: 0.85
                        max_candidates: 5000
                        relevance_norm: minmax_raw_score
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
              generation:
                sequence_length: 30
                plan:
                  - name: demo_plan
                    quota: 1
                    sampling:
                      include_inputs: [demo_pwm]
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


def test_describe_outputs_summary(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_min_config(cfg_path)
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "config", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "Config" in result.output
    assert "Pad" in result.output
    assert "See `" in result.output
    assert "dense inspect inputs" in result.output


def test_inspect_config_shows_mmr_pool_details(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_pwm_mmr_config(cfg_path)
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "config", "-c", str(cfg_path)], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "mmr(" in result.output
    assert "min=0.85" not in result.output
    assert "rel=minmax_raw_score" not in result.output
    assert "cap=5000" in result.output
