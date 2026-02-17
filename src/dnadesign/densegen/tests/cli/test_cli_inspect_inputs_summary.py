"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_inspect_inputs_summary.py

CLI inspect inputs summary tests for DenseGen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.cli.main import app


def _write_config(path: Path, *, meme_path: Path) -> None:
    payload = textwrap.dedent(
        f"""
        densegen:
          schema_version: "2.9"
          run:
            id: demo
            root: "."
          inputs:
            - name: demo_sites
              type: binding_sites
              path: inputs.csv
            - name: demo_pwm
              type: pwm_meme
              path: {meme_path.as_posix()}
              sampling:
                strategy: stochastic
                n_sites: 2
                mining:
                  batch_size: 50
                  budget:
                    mode: fixed_candidates
                    candidates: 100
                length:
                  policy: exact
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
              - name: demo_plan
                quota: 1
                sampling:
                  include_inputs: [demo_sites, demo_pwm]
                regulator_constraints:
                  groups:
                    - name: core
                      members: [lexA_CTGTATAWAWWHACA, cpxR_MANWWHTTTAM]
                      min_required: 2
          solver:
            backend: CBC
            strategy: iterate
          logging:
            log_dir: outputs/logs
        """
    ).strip()
    payload += "\n"
    path.write_text(payload)


def _write_meme(path: Path) -> None:
    path.write_text(
        """
        MEME version 4

        ALPHABET= ACGT

        Background letter frequencies
        A 0.25 C 0.25 G 0.25 T 0.25

        MOTIF lexA_CTGTATAWAWWHACA
        letter-probability matrix: alength= 4 w= 3 nsites= 20 E= 0
        0.8 0.1 0.05 0.05
        0.1 0.7 0.1 0.1
        0.1 0.1 0.7 0.1

        MOTIF cpxR_MANWWHTTTAM
        letter-probability matrix: alength= 4 w= 2 nsites= 10 E= 0
        0.6 0.2 0.1 0.1
        0.2 0.6 0.1 0.1
        """.strip()
        + "\n"
    )


def test_inspect_inputs_uses_clear_labels(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    meme_path = tmp_path / "motifs.meme"
    _write_meme(meme_path)
    _write_config(cfg_path, meme_path=meme_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "inputs", "-c", str(cfg_path)], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "Stage-A input sources" in result.output
    assert "motifs" in result.output
    assert "lexA,cpxR" in result.output
    assert "stage-a pool" in result.output.lower()
    assert "uv run dense stage-a build-pool --fresh -c " in result.output
    assert " && " not in result.output


def test_inspect_inputs_can_show_motif_ids(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    meme_path = tmp_path / "motifs.meme"
    _write_meme(meme_path)
    _write_config(cfg_path, meme_path=meme_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["inspect", "inputs", "--show-motif-ids", "-c", str(cfg_path)],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    assert "lexA_CTGTATAWAWWHACA" in result.output
    assert "cpxR_MANWWHTTTAM" in result.output


def test_inspect_inputs_absolute_paths(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    meme_path = tmp_path / "motifs.meme"
    _write_meme(meme_path)
    _write_config(cfg_path, meme_path=meme_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "inputs", "--absolute", "-c", str(cfg_path)], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert str(tmp_path) in result.output
