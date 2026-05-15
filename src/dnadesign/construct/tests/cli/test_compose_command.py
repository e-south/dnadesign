"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/cli/test_compose_command.py

CLI tests for Construct linear ssDNA composition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.construct.cli import app

_RUNNER = CliRunner()


def _write_minimal_composition_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "minimal_composition.yaml"
    config_path.write_text(
        """
contract: linear_ssdna_composition_v1
schema_version: 1
composition_id: synthetic_x3
units:
  - unit_id: synthetic_unit
    repeat_count: 3
    segments:
      - segment_id: left
        sequence: AAAA
      - segment_id: payload
        sequence: ACGT
      - segment_id: payload_rc
        sequence: ACGT
        transform:
          kind: reverse_complement
          source_segment_id: payload
      - segment_id: right
        sequence: TTTT
    annotations:
      - annotation_id: payload_annotation
        role: payload
        location:
          basis: segment
          segment_id: payload
          start: 0
          end: 4
output:
  artifact_bundle: artifacts/synthetic_x3
""",
        encoding="utf-8",
    )
    return config_path


def test_compose_validate_reports_summary_json(tmp_path: Path) -> None:
    config_path = _write_minimal_composition_config(tmp_path)

    result = _RUNNER.invoke(app, ["compose", "validate", "--config", config_path.as_posix(), "--format", "json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["composition_id"] == "synthetic_x3"
    assert payload["unit_count"] == 1
    assert payload["expanded_copy_count"] == 3


def test_compose_run_writes_bundle_and_reports_json(tmp_path: Path) -> None:
    config_path = _write_minimal_composition_config(tmp_path)

    result = _RUNNER.invoke(app, ["compose", "run", "--config", config_path.as_posix(), "--format", "json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["composition"]["composition_id"] == "synthetic_x3"
    assert payload["composition"]["sequence_length"] == 48
    assert Path(payload["composition"]["artifact_bundle"]) == tmp_path / "artifacts" / "synthetic_x3"
    assert Path(payload["artifacts"]["genbank"]) == tmp_path / "artifacts" / "synthetic_x3" / "sequence.gb"
    assert payload["artifacts"]["finder_reveal"].startswith("open -R ")
    assert (tmp_path / "artifacts" / "synthetic_x3" / "assembled_sequence.json").exists()


def test_compose_run_text_reports_genbank_finder_reveal(tmp_path: Path) -> None:
    config_path = _write_minimal_composition_config(tmp_path)

    result = _RUNNER.invoke(app, ["compose", "run", "--config", config_path.as_posix()])

    assert result.exit_code == 0, result.stdout
    assert f"genbank: {tmp_path / 'artifacts' / 'synthetic_x3' / 'sequence.gb'}" in result.stdout
    assert "finder_reveal: open -R " in result.stdout
