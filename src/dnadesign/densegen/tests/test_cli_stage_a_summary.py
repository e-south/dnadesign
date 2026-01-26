"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_cli_stage_a_summary.py

CLI coverage for Stage-A build-pool summaries.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# ABOUTME: CLI coverage for Stage-A build-pool length summaries.
# ABOUTME: Ensures pooled TFBS length stats are surfaced in stdout.
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.densegen.src.cli import _format_tier_counts, app
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


def _write_stage_a_config(tmp_path: Path) -> Path:
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
    return cfg_path


def _write_pwm_stage_a_config(tmp_path: Path) -> Path:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
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

            MOTIF M2
            letter-probability matrix: alength= 4 w= 2 nsites= 10 E= 0
            0.6 0.2 0.1 0.1
            0.2 0.6 0.1 0.1
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
                - name: demo_pwm
                  type: pwm_meme
                  path: {meme_path}
                  sampling:
                    strategy: consensus
                    n_sites: 1
                    oversample_factor: 1
                    scoring_backend: fimo
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
    return cfg_path


def test_stage_a_build_pool_reports_sampling_recap(tmp_path: Path) -> None:
    cfg_path = _write_stage_a_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["stage-a", "build-pool", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "Stage-A sampling recap" in result.output
    assert "Input: toy_sites" in result.output
    assert "candidates" in result.output
    assert "tiers" in result.output
    assert "score" in result.output
    assert "retained" in result.output
    assert "provided:" in result.output


def test_stage_a_build_pool_accepts_fresh_flag(tmp_path: Path) -> None:
    cfg_path = _write_stage_a_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["stage-a", "build-pool", "--fresh", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output


def test_stage_a_build_pool_logs_initialized(tmp_path: Path) -> None:
    cfg_path = _write_stage_a_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["stage-a", "build-pool", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "Logging initialized" in result.output


def test_stage_a_build_pool_reports_plan(tmp_path: Path) -> None:
    cfg_path = _write_pwm_stage_a_config(tmp_path)
    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")
    runner = CliRunner()
    result = runner.invoke(app, ["stage-a", "build-pool", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "Stage-A plan" in result.output
    assert "M1" in result.output
    assert "M2" in result.output


def test_tier_rows_include_zero_counts() -> None:
    label = _format_tier_counts([2, 0, 1], [1, 0, 0])
    assert label == "t0 2/1 | t1 0/0 | t2 1/0"
