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
    assert f"input_root: {usr_root}" in output
    assert f"output_root: {usr_root}" in output
    assert "template_id: template_demo" in output
    assert "template_kind: literal" in output
    assert "template_length: 16" in output
    assert "template_circular: true" in output
    assert "template_sha256:" in output
    assert "realize_mode: window" in output
    assert "focal_part: anchor" in output
    assert "focal_point: center" in output
    assert "anchor_offset_bp: 0" in output
    assert "window_bp: 8" in output
    assert "spec_id:" in output
    assert "output_on_conflict: error" in output
    assert "existing_output_collisions: 0" in output
    assert "placement: part=anchor" in output
    assert "template_start=8" in output
    assert "template_end=12" in output
    assert "expected_template_sequence=CCCC" in output
    assert "rows_total: 1" in output
    assert "output_id=" in output


def test_validate_config_runtime_shields_template_path_io_errors(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    _write_registry(usr_root)
    dataset = Dataset(usr_root, "anchors_demo")
    dataset.init(source="test", notes="validate runtime")
    dataset.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")
    template_dir = tmp_path / "template_dir"
    template_dir.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "config_path_template.yaml"
    config_path.write_text(
        f"""
job:
  id: runtime_path_template_error
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
  template:
    id: template_demo
    kind: path
    path: {template_dir.as_posix()}
    circular: false
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

    result = _RUNNER.invoke(app, ["validate", "config", "--config", config_path.as_posix(), "--runtime"])

    assert result.exit_code == 1
    assert "Template path must resolve to a readable file" in (result.stdout or "")


def test_validate_config_runtime_rejects_same_start_mixed_intervals(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    _write_registry(usr_root)
    dataset = Dataset(usr_root, "anchors_demo")
    dataset.init(source="test", notes="validate runtime")
    dataset.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "config_same_start_mixed.yaml"
    config_path.write_text(
        f"""
job:
  id: runtime_same_start_mixed
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: template_demo
    sequence: AAAATTTTCCCCGGGG
    circular: false
  parts:
    - name: replace_anchor
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
    - name: insert_tag
      role: helper
      sequence:
        source: literal
        literal: GG
      placement:
        kind: insert
        start: 4
        end: 4
        orientation: forward
  realize:
    mode: full_construct
  output:
    dataset: anchors_demo_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", config_path.as_posix(), "--runtime"])

    assert result.exit_code == 1
    assert "Same-start placements with different intervals are ambiguous" in (result.stdout or "")


def test_validate_config_runtime_reports_usr_template_details(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    _write_registry(usr_root)
    dataset = Dataset(usr_root, "anchors_demo")
    dataset.init(source="test", notes="validate runtime")
    dataset.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")
    templates = Dataset(usr_root, "templates_demo")
    templates.init(source="test", notes="template runtime")
    templates.add_sequences(["AAAATTTTCCCCGGGG"], bio_type="dna", alphabet="dna_4", source="test")
    template_id = templates.head(n=1).iloc[0]["id"]

    config_path = tmp_path / "config_usr_template.yaml"
    config_path.write_text(
        f"""
job:
  id: runtime_usr_template
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
  template:
    id: template_demo
    kind: usr
    dataset: templates_demo
    root: {usr_root.as_posix()}
    record_id: {template_id}
    field: sequence
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
    assert f"input_root: {usr_root}" in output
    assert f"output_root: {usr_root}" in output
    assert "template_kind: usr" in output
    assert "template_dataset: templates_demo" in output
    assert f"template_record_id: {template_id}" in output


def test_validate_config_runtime_normalizes_usr_package_root(tmp_path: Path) -> None:
    usr_pkg_root = tmp_path / "usr"
    usr_pkg_root.mkdir(parents=True, exist_ok=True)
    (usr_pkg_root / "__init__.py").write_text("# stub\n", encoding="utf-8")
    usr_root = usr_pkg_root / "datasets"
    _write_registry(usr_root)
    dataset = Dataset(usr_root, "anchors_demo")
    dataset.init(source="test", notes="normalize runtime")
    dataset.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "config_pkg_root.yaml"
    config_path.write_text(
        f"""
job:
  id: runtime_pkg_root
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_pkg_root.as_posix()}
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
    root: {usr_pkg_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        ["validate", "config", "--config", config_path.as_posix(), "--runtime"],
    )

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert f"input_root: {usr_root}" in output
    assert f"output_root: {usr_root}" in output
