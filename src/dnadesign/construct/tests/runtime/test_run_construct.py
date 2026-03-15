"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_run_construct.py

Runtime realization tests for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.construct.src.api import preflight_from_config, run_from_config
from dnadesign.construct.src.errors import ValidationError
from dnadesign.usr import Dataset


def _write_registry(root: Path) -> None:
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


def test_run_construct_realizes_multi_part_linear_window(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_linear
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    sequence: AAAATTTTCCCCGGGG
    circular: false
  parts:
    - name: tag
      role: helper
      sequence:
        source: literal
        literal: GG
      placement:
        kind: insert
        start: 4
        end: 4
        orientation: forward
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
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.records_total == 1
    output_ds = Dataset(usr_root, "anchors_constructed")
    frame = output_ds.head(n=5)
    assert frame.iloc[0]["sequence"] == "TTACGTGG"
    assert frame.iloc[0]["construct__anchor_id"]
    assert frame.iloc[0]["construct__part_count"] == 2
    assert list(frame.iloc[0]["construct__part_names"]) == ["tag", "anchor"]
    assert frame.iloc[0]["construct__template_kind"] == "literal"


def test_run_construct_supports_circular_window_wrap(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["GG"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_circular
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: circular_template
    sequence: AAAACCCC
    circular: true
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        start: 6
        end: 8
        orientation: forward
        expected_template_sequence: CC
  realize:
    mode: window
    focal_part: anchor
    focal_point: center
    window_bp: 6
  output:
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.records_total == 1
    output_ds = Dataset(usr_root, "anchors_constructed")
    frame = output_ds.head(n=5)
    assert frame.iloc[0]["sequence"] == "CCGGAA"
    assert bool(frame.iloc[0]["construct__template_circular"]) is True
    assert frame.iloc[0]["construct__window_start"] == 4
    assert frame.iloc[0]["construct__window_end"] == 2
    assert list(frame.iloc[0]["construct__part_kinds"]) == ["replace"]


def test_run_construct_supports_negative_anchor_offset_on_circular_window(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["GG"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_negative_offset.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_negative_offset
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: circular_template
    sequence: AAAACCCC
    circular: true
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        start: 6
        end: 8
        orientation: forward
        expected_template_sequence: CC
  realize:
    mode: window
    focal_part: anchor
    focal_point: center
    anchor_offset_bp: -2
    window_bp: 6
  output:
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.records_total == 1
    output_ds = Dataset(usr_root, "anchors_constructed")
    frame = output_ds.head(n=5)
    assert frame.iloc[0]["sequence"] == "AACCGG"
    assert frame.iloc[0]["construct__window_start"] == 2
    assert frame.iloc[0]["construct__window_end"] == 0


def test_run_construct_supports_reverse_complement_orientation(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["AGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_reverse_complement
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    sequence: AAAACCCC
    circular: false
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
        orientation: reverse_complement
        expected_template_sequence: CCCC
  realize:
    mode: full_construct
  output:
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.records_total == 1
    output_ds = Dataset(usr_root, "anchors_constructed")
    frame = output_ds.head(n=5)
    assert frame.iloc[0]["sequence"] == "AAAAACT"
    assert list(frame.iloc[0]["construct__part_orientations"]) == ["reverse_complement"]


def test_run_construct_rejects_mismatched_expected_template_sequence(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["GG"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_mismatch
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    sequence: AAAACCCC
    circular: false
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
    mode: full_construct
  output:
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="expected template interval"):
        run_from_config(config_path)


def test_run_construct_supports_usr_backed_template_records(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["GG"], bio_type="dna", alphabet="dna_4", source="test")

    template_ds = Dataset(usr_root, "templates_demo")
    template_ds.init(source="test", notes="template test")
    template_ds.add_sequences(["AAAACCCC"], bio_type="dna", alphabet="dna_4", source="test")
    template_id = template_ds.head(n=1).iloc[0]["id"]

    config_path = tmp_path / "construct_usr_template.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_usr_template
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: circular_template
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
        start: 6
        end: 8
        orientation: forward
        expected_template_sequence: CC
  realize:
    mode: full_construct
  output:
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.records_total == 1
    output_ds = Dataset(usr_root, "anchors_constructed")
    frame = output_ds.head(n=5)
    assert frame.iloc[0]["sequence"] == "AAAACCGG"
    assert frame.iloc[0]["construct__template_kind"] == "usr"
    assert frame.iloc[0]["construct__template_dataset"] == "templates_demo"
    assert frame.iloc[0]["construct__template_record_id"] == template_id


def test_run_construct_rejects_multi_record_fasta_template(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["GG"], bio_type="dna", alphabet="dna_4", source="test")

    fasta_path = tmp_path / "multi.fa"
    fasta_path.write_text(">first\nAAAA\n>second\nCCCC\n", encoding="utf-8")

    config_path = tmp_path / "construct_multi_fasta.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_multi_fasta
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: fasta_template
    kind: path
    path: {fasta_path.as_posix()}
    circular: false
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        start: 2
        end: 4
        orientation: forward
        expected_template_sequence: AA
  realize:
    mode: full_construct
  output:
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="exactly one record"):
        run_from_config(config_path)


def test_run_construct_rejects_registry_type_drift_before_write(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    (usr_root / "registry.yaml").write_text(
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
  construct:
    owner: construct
    description: Construct lineage overlays for realized DNA sequences.
    columns:
      - name: construct__template_length
        type: string
""",
        encoding="utf-8",
    )

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_bad_registry.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_bad_registry
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
    mode: full_construct
  output:
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="construct__template_length"):
        run_from_config(config_path)


def test_run_construct_preserves_equal_coordinate_part_order_in_output_and_metadata(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["AC"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_equal_coordinate_order.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_equal_coordinate_order
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    sequence: AAAACCCC
    circular: false
  parts:
    - name: z_insert
      role: helper
      sequence:
        source: literal
        literal: GG
      placement:
        kind: insert
        start: 4
        end: 4
        orientation: forward
    - name: a_insert
      role: helper
      sequence:
        source: literal
        literal: TT
      placement:
        kind: insert
        start: 4
        end: 4
        orientation: forward
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        start: 6
        end: 8
        orientation: forward
        expected_template_sequence: CC
  realize:
    mode: full_construct
  output:
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    run_from_config(config_path)

    output_ds = Dataset(usr_root, "anchors_constructed")
    frame = output_ds.head(n=5)
    assert frame.iloc[0]["sequence"] == "AAAAGGTTCCAC"
    assert list(frame.iloc[0]["construct__part_names"]) == ["z_insert", "a_insert", "anchor"]


def test_preflight_reports_equal_coordinate_insert_order_consistently_with_lineage(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["AC"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_equal_coordinate_preflight.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_equal_coordinate_preflight
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    sequence: AAAACCCC
    circular: false
  parts:
    - name: z_insert
      role: helper
      sequence:
        source: literal
        literal: GG
      placement:
        kind: insert
        start: 4
        end: 4
        orientation: forward
    - name: a_insert
      role: helper
      sequence:
        source: literal
        literal: TT
      placement:
        kind: insert
        start: 4
        end: 4
        orientation: forward
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        start: 6
        end: 8
        orientation: forward
        expected_template_sequence: CC
  realize:
    mode: full_construct
  output:
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    preflight = preflight_from_config(config_path)
    assert [placement.part_name for placement in preflight.placements] == ["z_insert", "a_insert", "anchor"]

    run_from_config(config_path)
    frame = Dataset(usr_root, "anchors_constructed").head(n=5)
    assert list(frame.iloc[0]["construct__part_names"]) == ["z_insert", "a_insert", "anchor"]


def test_preflight_rejects_same_start_mixed_intervals(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_same_start_mixed.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_same_start_mixed
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
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
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Same-start placements with different intervals are ambiguous"):
        preflight_from_config(config_path)


def test_preflight_spec_id_changes_with_selected_input_ids(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGT", "TGCA"], bio_type="dna", alphabet="dna_4", source="test")
    input_frame = input_ds.head(n=10)
    first_id = input_frame.iloc[0]["id"]
    second_id = input_frame.iloc[1]["id"]

    first_config = tmp_path / "spec_first.yaml"
    second_config = tmp_path / "spec_second.yaml"
    for config_path, selected_id in ((first_config, first_id), (second_config, second_id)):
        config_path.write_text(
            f"""
job:
  id: demo_spec_ids
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
    ids: [{selected_id}]
  template:
    id: linear_template
    sequence: AAAATTTTCCCCGGGG
    circular: false
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
    mode: full_construct
  output:
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
            encoding="utf-8",
        )

    assert preflight_from_config(first_config).spec_id != preflight_from_config(second_config).spec_id


def test_preflight_rejects_same_input_and_output_dataset_without_opt_in(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "same_input_output.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_same_dataset
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    sequence: AAAATTTTCCCCGGGG
    circular: false
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
    mode: full_construct
  output:
    dataset: anchors_demo
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="same root/dataset as input"):
        preflight_from_config(config_path)


def test_preflight_detects_existing_output_collisions_and_ignore_mode_skips_them(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "collision.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_collision
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    sequence: AAAATTTTCCCCGGGG
    circular: false
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
    mode: full_construct
  output:
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )
    ignore_config = tmp_path / "collision_ignore.yaml"
    ignore_config.write_text(
        config_path.read_text(encoding="utf-8")
        .replace(
            "root: " + usr_root.as_posix(),
            "root: " + usr_root.as_posix(),
            2,
        )
        .replace(
            "  output:\n    dataset: anchors_constructed\n    root: " + usr_root.as_posix(),
            "  output:\n    dataset: anchors_constructed\n    root: "
            + usr_root.as_posix()
            + "\n    on_conflict: ignore",
        ),
        encoding="utf-8",
    )

    first = run_from_config(config_path)
    assert first.records_written == 1

    with pytest.raises(ValidationError, match="already exist"):
        preflight_from_config(config_path)

    preflight = preflight_from_config(ignore_config)
    assert preflight.existing_output_collisions == 1
    assert preflight.output_on_conflict == "ignore"

    second = run_from_config(ignore_config)
    assert second.records_written == 0
    assert second.records_skipped_existing == 1


def test_preflight_rejects_duplicate_planned_output_ids(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")
    anchor_id = str(input_ds.head(n=1).iloc[0]["id"])

    config_path = tmp_path / "duplicate_outputs.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_duplicate_outputs
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
    ids: [{anchor_id}, {anchor_id}]
  template:
    id: linear_template
    sequence: AAAATTTTCCCCGGGG
    circular: false
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
    mode: full_construct
  output:
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="duplicate planned output id"):
        preflight_from_config(config_path)


def test_run_construct_can_append_new_rows_to_existing_output_dataset(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["GG"], bio_type="dna", alphabet="dna_4", source="test")

    first_config = tmp_path / "append_first.yaml"
    second_config = tmp_path / "append_second.yaml"
    first_config.write_text(
        f"""
job:
  id: demo_append_first
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    sequence: AAAACCCC
    circular: false
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        start: 6
        end: 8
        orientation: forward
        expected_template_sequence: CC
  realize:
    mode: full_construct
  output:
    dataset: anchors_constructed
    root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )
    second_config.write_text(
        first_config.read_text(encoding="utf-8")
        .replace("demo_append_first", "demo_append_second")
        .replace("start: 6", "start: 4")
        .replace("end: 8", "end: 6"),
        encoding="utf-8",
    )

    run_from_config(first_config)
    result = run_from_config(second_config)

    assert result.records_written == 1
    output_ds = Dataset(usr_root, "anchors_constructed")
    frame = output_ds.head(n=10)
    assert len(frame) == 2
    assert set(frame["sequence"]) == {"AAAACCGG", "AAAAGGCC"}
    assert "construct__job" in frame.columns


def test_run_construct_can_append_into_input_dataset_when_allowed(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["GG"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "same_dataset_allowed.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_same_dataset_allowed
  input:
    source: usr
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    sequence: AAAACCCC
    circular: false
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        start: 6
        end: 8
        orientation: forward
        expected_template_sequence: CC
  realize:
    mode: full_construct
  output:
    dataset: anchors_demo
    root: {usr_root.as_posix()}
    allow_same_as_input: true
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.records_written == 1
    frame = Dataset(usr_root, "anchors_demo").head(n=10)
    assert len(frame) == 2
    assert "AAAACCGG" in set(frame["sequence"])
    assert "construct__job" in frame.columns
