"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/cli/test_validate_command.py

Validation command hardening tests for infer CLI config parsing behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.infer.cli import app

_RUNNER = CliRunner()


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_validate_config_rejects_unknown_fields_with_config_exit_code(tmp_path: Path) -> None:
    bad = _write(
        tmp_path / "bad_extra.yaml",
        """
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
  typo_field: 123
jobs:
  - id: j1
    operation: extract
    ingest:
      source: sequences
    outputs:
      - id: ll
        fn: evo2.log_likelihood
        format: float
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", bad.as_posix()])

    assert result.exit_code == 2
    output = (result.stdout or "").lower()
    assert "extra inputs are not permitted" in output or "extra_forbidden" in output


def test_validate_config_rejects_wrong_type_with_config_exit_code(tmp_path: Path) -> None:
    bad = _write(
        tmp_path / "bad_type.yaml",
        """
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
  batch_size: not_an_int
jobs:
  - id: j1
    operation: extract
    ingest:
      source: sequences
    outputs:
      - id: ll
        fn: evo2.log_likelihood
        format: float
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", bad.as_posix()])

    assert result.exit_code == 2
    output = (result.stdout or "").lower()
    assert "valid integer" in output


def test_validate_config_requires_explicit_path_or_cwd_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    result = _RUNNER.invoke(app, ["validate", "config"])

    assert result.exit_code == 2
    assert "No config found. Pass --config or place config.yaml in the current directory." in (result.stdout or "")


def test_validate_config_rejects_usr_ingest_path_field(tmp_path: Path) -> None:
    bad = _write(
        tmp_path / "bad_usr_path.yaml",
        """
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
jobs:
  - id: j1
    operation: extract
    ingest:
      source: usr
      dataset: demo_dataset
      path: inputs/records.jsonl
    outputs:
      - id: ll
        fn: evo2.log_likelihood
        format: float
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", bad.as_posix()])

    assert result.exit_code == 2
    assert "ingest.path is not allowed for source='usr'" in (result.stdout or "")
