"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/cli/test_validate_command.py

CLI validation command contracts for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.construct.cli import app
from dnadesign.usr import Dataset

_RUNNER = CliRunner()


def _write_registry(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "registry.yaml").write_text(
        """
namespaces:
  usr_state:
    owner: usr
    description: Reserved record-state overlay (masked/qc/split/lineage).
    columns:
      - name: usr_state__masked
        type: bool
      - name: usr_state__qc_status
        type: string
      - name: usr_state__split
        type: string
      - name: usr_state__supersedes
        type: string
      - name: usr_state__lineage
        type: list<string>
""",
        encoding="utf-8",
    )


def test_validate_config_accepts_minimal_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
job:
  id: demo_job
  input:
    source: usr
    dataset: anchors_demo
  template:
    id: template_demo
    sequence: AAAATTTTCCCCGGGG
    circular: true
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        start: 4
        end: 8
        orientation: forward
        expected_template_sequence: TTTT
  realize:
    mode: window
    focal_part: anchor
    focal_point: center
    window_bp: 8
  output:
    dataset: anchors_demo_constructed
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", config_path.as_posix()])

    assert result.exit_code == 0, result.stdout
    assert "Config OK:" in (result.stdout or "")
    assert "job_id: demo_job" in (result.stdout or "")


def test_validate_config_rejects_missing_input_driven_part(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
job:
  id: demo_job
  input:
    source: usr
    dataset: anchors_demo
  template:
    id: template_demo
    sequence: AAAATTTTCCCCGGGG
  parts:
    - name: literal_only
      role: helper
      sequence:
        source: literal
        literal: ACGT
      placement:
        kind: replace
        start: 4
        end: 8
        orientation: forward
        expected_template_sequence: TTTT
  realize:
    mode: full_construct
  output:
    dataset: anchors_demo_constructed
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", config_path.as_posix()])

    assert result.exit_code == 1
    assert "must include at least one source='input_field' part" in (result.stdout or "")


def test_validate_config_runtime_reports_preflight_summary(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    _write_registry(usr_root)
    dataset = Dataset(usr_root, "anchors_demo")
    dataset.init(source="test", notes="validate runtime")
    dataset.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
job:
  id: runtime_demo
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
  template:
    id: template_demo
    sequence: AAAATTTTCCCCGGGG
    circular: true
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        start: 8
        end: 12
        orientation: forward
        expected_template_sequence: CCCC
  realize:
    mode: window
    focal_part: anchor
    focal_point: center
    window_bp: 8
  output:
    dataset: anchors_demo_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        ["validate", "config", "--config", config_path.as_posix(), "--runtime"],
    )

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "template_id: template_demo" in output
    assert "template_length: 16" in output
    assert "template_circular: true" in output
    assert "rows_total: 1" in output
