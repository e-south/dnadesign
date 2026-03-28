"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/cli/test_validate_command.py

CLI validation command contracts for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.construct.cli import app
from dnadesign.usr import Dataset
from dnadesign.usr import SchemaError as USRSchemaError

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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
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
          start: 8
          end: 12
        guards:
          replaced_sequence: CCCC
          upstream_sequence: TTTT
          downstream_sequence: GGGG
          require_unique_forward_matches: true
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
    assert "window_semantics: fixed_total" in output
    assert "window_reference: center" in output
    assert "window_direction: symmetric" in output
    assert "window_size_bp: 8" in output
    assert "window_offset_bp: 0" in output
    assert "spec_id:" in output
    assert "output_on_conflict: error" in output
    assert "existing_output_collisions: 0" in output
    assert "placement: part=anchor" in output
    assert "template_start=8" in output
    assert "template_end=12" in output
    assert "template_span_bp=4" in output
    assert "locator_kind=coordinates" in output
    assert "guard_mode=replaced_sequence_and_context" in output
    assert "guard_require_unique_forward_matches=true" in output
    assert "template_sequence=CCCC" in output
    assert "guard_replaced_sequence=CCCC" in output
    assert "guard_upstream_sequence=TTTT" in output
    assert "observed_guard_upstream_sequence=TTTT" in output
    assert "guard_downstream_sequence=GGGG" in output
    assert "observed_guard_downstream_sequence=GGGG" in output
    assert "rows_total: 1" in output
    assert "output_id=" in output


def test_validate_config_runtime_json_reports_preflight_payload(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    _write_registry(usr_root)
    dataset = Dataset(usr_root, "anchors_demo")
    dataset.init(source="test", notes="validate runtime json")
    dataset.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "config_runtime_json.yaml"
    config_path.write_text(
        f"""
job:
  id: runtime_demo_json
  input:
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
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
          start: 8
          end: 12
        guards:
          replaced_sequence: CCCC
          upstream_sequence: TTTT
          downstream_sequence: GGGG
          require_unique_forward_matches: true
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
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        ["validate", "config", "--config", config_path.as_posix(), "--runtime", "--format", "json"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["job"]["id"] == "runtime_demo_json"
    assert payload["runtime_preflight"]["template_id"] == "template_demo"
    placement = payload["runtime_preflight"]["placements"][0]
    assert placement["template_span_bp"] == 4
    assert placement["locator_kind"] == "coordinates"
    assert placement["guard_mode"] == "replaced_sequence_and_context"
    assert placement["guard_require_unique_forward_matches"] is True


def test_validate_config_runtime_json_reports_flank_locator_payload(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    _write_registry(usr_root)
    dataset = Dataset(usr_root, "anchors_demo")
    dataset.init(source="test", notes="validate runtime flank json")
    dataset.add_sequences(["AC"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "config_runtime_flanks.yaml"
    config_path.write_text(
        f"""
job:
  id: runtime_demo_flanks
  input:
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
  template:
    id: template_demo
    source:
      kind: literal
      sequence: AAAACCCCGGGGTTTT
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
          kind: flanks
          upstream_sequence: AAAA
          downstream_sequence: GGGG
        guards:
          replaced_sequence: CCCC
          replaced_span_bp: 4
  realize:
    mode: full_construct
  output:
    target:
      kind: usr
      dataset: anchors_demo_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        ["validate", "config", "--config", config_path.as_posix(), "--runtime", "--format", "json"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    placement = payload["runtime_preflight"]["placements"][0]
    assert placement["locator_kind"] == "flanks"
    assert placement["locator_upstream_sequence"] == "AAAA"
    assert placement["locator_downstream_sequence"] == "GGGG"
    assert placement["template_start"] == 4
    assert placement["template_end"] == 8
    assert placement["guard_replaced_span_bp"] == 4


def test_validate_config_runtime_shapes_usr_preflight_errors(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config_runtime_usr_error.yaml"
    config_path.write_text(
        """
job:
  id: runtime_usr_error
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
  output:
    target:
      kind: usr
      dataset: anchors_demo_constructed
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "dnadesign.construct.src.api._runtime_preflight_from_config",
        lambda path: (_ for _ in ()).throw(USRSchemaError("registry schema mismatch")),
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", config_path.as_posix(), "--runtime"])

    assert result.exit_code == 1
    assert "construct preflight failed while reading USR inputs: registry schema mismatch" in (result.stdout or "")


def test_validate_config_runtime_missing_input_dataset_suggests_seed_or_import(tmp_path: Path) -> None:
    config_path = tmp_path / "config_missing_dataset.yaml"
    config_path.write_text(
        f"""
job:
  id: runtime_missing_dataset
  input:
    source:
      kind: usr
      dataset: anchors_demo
      root: {(tmp_path / "usr_root").as_posix()}
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
    mode: full_construct
  output:
    target:
      kind: usr
      dataset: anchors_demo_constructed
      root: {(tmp_path / "usr_root").as_posix()}
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", config_path.as_posix(), "--runtime"])

    assert result.exit_code == 1
    output = result.stdout or ""
    assert "Input dataset not initialized:" in output
    assert "Seed or import the required dataset before runtime validation or run." in output


def test_validate_config_runtime_missing_input_dataset_reports_json_error(tmp_path: Path) -> None:
    config_path = tmp_path / "config_missing_dataset.yaml"
    config_path.write_text(
        f"""
job:
  id: runtime_missing_dataset
  input:
    source:
      kind: usr
      dataset: anchors_demo
      root: {(tmp_path / "usr_root").as_posix()}
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
    mode: full_construct
  output:
    target:
      kind: usr
      dataset: anchors_demo_constructed
      root: {(tmp_path / "usr_root").as_posix()}
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        ["validate", "config", "--config", config_path.as_posix(), "--runtime", "--format", "json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["code"] == 1
    assert payload["error_type"] == "ValidationError"
    assert "Input dataset not initialized:" in payload["error"]
    assert "Seed or import the required dataset before runtime validation or run." in payload["error"]


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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
  template:
    id: template_demo
    source:
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
        orientation: forward
        locator:
          kind: coordinates
          start: 0
          end: 4
        guards:
          replaced_sequence: AAAA
  realize:
    mode: full_construct
  output:
    target:
      kind: usr
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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
  template:
    id: template_demo
    source:
      kind: literal
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
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 8
        guards:
          replaced_sequence: TTTT
    - name: insert_tag
      role: helper
      sequence:
        source: literal
        literal: GG
      placement:
        kind: insert
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 4
  realize:
    mode: full_construct
  output:
    target:
      kind: usr
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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
  template:
    id: template_demo
    source:
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
        orientation: forward
        locator:
          kind: coordinates
          start: 8
          end: 12
        guards:
          replaced_sequence: CCCC
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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_pkg_root.as_posix()}
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
          start: 8
          end: 12
        guards:
          replaced_sequence: CCCC
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
