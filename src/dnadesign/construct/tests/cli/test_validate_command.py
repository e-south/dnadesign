"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/cli/test_validate_command.py

CLI validation command schema contracts for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.construct.src.cli import app

_RUNNER = CliRunner()


def test_validate_config_accepts_minimal_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
job:
  id: demo_job
  input:
    source:
      kind: usr
      dataset: anchors_demo
      root: outputs/usr_datasets
  template:
    id: template_demo
    source:
      kind: literal
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
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 8
        guards:
          replaced_sequence: TTTT
  realize:
    mode: window
    focal_part: anchor
    window:
      semantics: fixed_total
      reference: center
      direction: symmetric
      size_bp: 8
      offset_bp: 0
  output:
    target:
      kind: usr
      dataset: anchors_demo_constructed
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", config_path.as_posix()])

    assert result.exit_code == 0, result.stdout
    assert "Config OK:" in (result.stdout or "")
    assert "job_id: demo_job" in (result.stdout or "")


def test_validate_config_rejects_usr_input_without_explicit_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config_missing_root.yaml"
    config_path.write_text(
        """
job:
  id: demo_job
  input:
    source:
      kind: usr
      dataset: anchors_demo
  template:
    id: template_demo
    source:
      kind: literal
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
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 8
        guards:
          replaced_sequence: TTTT
  realize:
    mode: window
    focal_part: anchor
    window:
      semantics: fixed_total
      reference: center
      direction: symmetric
      size_bp: 8
      offset_bp: 0
  output:
    target:
      kind: usr
      dataset: anchors_demo_constructed
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", config_path.as_posix()])

    assert result.exit_code == 1
    assert "job.input.source.root is required for construct jobs that read USR datasets" in (result.stdout or "")


def test_validate_config_rejects_missing_input_driven_part(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
job:
  id: demo_job
  input:
    source:
      kind: usr
      dataset: anchors_demo
      root: outputs/usr_datasets
  template:
    id: template_demo
    source:
      kind: literal
      sequence: AAAATTTTCCCCGGGG
  parts:
    - name: literal_only
      role: helper
      sequence:
        source: literal
        literal: ACGT
      placement:
        kind: replace
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 8
        guards:
          replaced_sequence: TTTT
  realize:
    mode: full_construct
  output:
    target:
      kind: usr
      dataset: anchors_demo_constructed
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", config_path.as_posix()])

    assert result.exit_code == 1
    assert "must include at least one source='input_field' part" in (result.stdout or "")


def test_validate_config_accepts_explicit_window_block(tmp_path: Path) -> None:
    config_path = tmp_path / "config_window_block.yaml"
    config_path.write_text(
        """
job:
  id: demo_window_block
  input:
    source:
      kind: usr
      dataset: anchors_demo
      root: outputs/usr_datasets
  template:
    id: template_demo
    source:
      kind: literal
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
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 8
        guards:
          replaced_sequence: TTTT
  realize:
    mode: window
    focal_part: anchor
    window:
      semantics: fixed_total
      reference: center
      direction: symmetric
      size_bp: 8
      offset_bp: 0
  output:
    target:
      kind: usr
      dataset: anchors_demo_constructed
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", config_path.as_posix()])

    assert result.exit_code == 0, result.stdout
    assert "Config OK:" in (result.stdout or "")


def test_validate_config_rejects_legacy_window_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config_legacy_window.yaml"
    config_path.write_text(
        """
job:
  id: demo_legacy_window
  input:
    source:
      kind: usr
      dataset: anchors_demo
      root: outputs/usr_datasets
  template:
    id: template_demo
    source:
      kind: literal
      sequence: AAAATTTTCCCCGGGG
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 8
        guards:
          replaced_sequence: TTTT
  realize:
    mode: window
    focal_part: anchor
    window:
      semantics: fixed_total
      reference: center
      direction: symmetric
      size_bp: 8
      offset_bp: 0
    window_bp: 8
  output:
    target:
      kind: usr
      dataset: anchors_demo_constructed
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", config_path.as_posix()])

    assert result.exit_code == 1
    assert "realize.window_bp is no longer supported" in (result.stdout or "")
