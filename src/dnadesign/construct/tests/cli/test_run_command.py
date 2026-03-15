"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/cli/test_run_command.py

CLI run command contracts for construct.

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


def _write_valid_config(*, tmp_path: Path, usr_root: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
job:
  id: cli_run_demo
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
    return config_path


def test_run_command_reports_dry_run_summary(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    _write_registry(usr_root)
    dataset = Dataset(usr_root, "anchors_demo")
    dataset.init(source="test", notes="run cli")
    dataset.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")
    config_path = _write_valid_config(tmp_path=tmp_path, usr_root=usr_root)

    result = _RUNNER.invoke(app, ["run", "--config", config_path.as_posix(), "--dry-run"])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "Config validated (dry run): job=cli_run_demo rows=1 output_collisions=0" in output
    assert f"output_root: {usr_root}" in output
    assert "output_dataset: anchors_demo_constructed" in output
    assert "spec_id:" in output


def test_run_command_writes_output_dataset(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    _write_registry(usr_root)
    dataset = Dataset(usr_root, "anchors_demo")
    dataset.init(source="test", notes="run cli")
    dataset.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")
    config_path = _write_valid_config(tmp_path=tmp_path, usr_root=usr_root)

    result = _RUNNER.invoke(app, ["run", "--config", config_path.as_posix()])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "Construct run complete: job=cli_run_demo rows_planned=1 rows_written=1 rows_skipped_existing=0" in output
    frame = Dataset(usr_root, "anchors_demo_constructed").head(n=5)
    assert len(frame) == 1
    assert frame.iloc[0]["construct__job"] == "cli_run_demo"


def test_run_command_shapes_construct_errors(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    _write_registry(usr_root)
    dataset = Dataset(usr_root, "anchors_demo")
    dataset.init(source="test", notes="run cli")
    dataset.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text(
        f"""
job:
  id: cli_run_bad
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
  template:
    id: template_demo
    path: {tmp_path.as_posix()}
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        start: 0
        end: 4
        orientation: forward
        expected_template_sequence: AAAA
  realize:
    mode: full_construct
  output:
    dataset: anchors_demo_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["run", "--config", config_path.as_posix()])

    assert result.exit_code == 1
    assert "Error: Template path must resolve to a readable file" in (result.stdout or "")
