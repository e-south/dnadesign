"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_run_construct_preflight.py

Runtime preflight and collision tests for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.construct.src.contracts.errors import ValidationError
from dnadesign.construct.src.interfaces.api import preflight_from_config, run_from_config
from dnadesign.construct.tests.runtime.run_construct_helpers import write_registry as _write_registry
from dnadesign.usr import Dataset


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
    mode: full_construct
  output:
    target:
      kind: usr
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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    source:
      kind: literal
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
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 4
    - name: a_insert
      role: helper
      sequence:
        source: literal
        literal: TT
      placement:
        kind: insert
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 4
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
          start: 6
          end: 8
        guards:
          replaced_sequence: CC
  realize:
    mode: full_construct
  output:
    target:
      kind: usr
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    run_from_config(config_path)

    output_ds = Dataset(usr_root, "anchors_constructed")
    frame = output_ds.head(n=5)
    assert frame.iloc[0]["sequence"] == "AAAAGGTTCCAC"
    assert [part["name"] for part in frame.iloc[0]["construct__parts"]] == ["z_insert", "a_insert", "anchor"]


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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    source:
      kind: literal
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
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 4
    - name: a_insert
      role: helper
      sequence:
        source: literal
        literal: TT
      placement:
        kind: insert
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 4
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
          start: 6
          end: 8
        guards:
          replaced_sequence: CC
  realize:
    mode: full_construct
  output:
    target:
      kind: usr
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    preflight = preflight_from_config(config_path)
    assert [placement.part_name for placement in preflight.placements] == ["z_insert", "a_insert", "anchor"]

    run_from_config(config_path)
    frame = Dataset(usr_root, "anchors_constructed").head(n=5)
    assert [part["name"] for part in frame.iloc[0]["construct__parts"]] == ["z_insert", "a_insert", "anchor"]


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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
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
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Same-start placements with different intervals are ambiguous"):
        preflight_from_config(config_path)


def test_preflight_rejects_partial_overlap(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_partial_overlap.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_partial_overlap
  input:
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    source:
      kind: literal
      sequence: AAAATTTTCCCCGGGG
    circular: false
  parts:
    - name: helper
      role: helper
      sequence:
        source: literal
        literal: GG
      placement:
        kind: replace
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 8
        guards:
          replaced_sequence: TTTT
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
          start: 6
          end: 10
        guards:
          replaced_sequence: TTCC
  realize:
    mode: full_construct
  output:
    target:
      kind: usr
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="overlaps prior placement"):
        preflight_from_config(config_path)


def test_preflight_rejects_fixed_total_window_shorter_than_focal_part(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGTAA"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_fixed_total_too_small.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_fixed_total_too_small
  input:
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    source:
      kind: literal
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
      size_bp: 5
      offset_bp: 0
  output:
    target:
      kind: usr
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="exceeds fixed_total window size_bp=5"):
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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
    ids: [{selected_id}]
  template:
    id: linear_template
    source:
      kind: literal
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
        orientation: forward
        locator:
          kind: coordinates
          start: 8
          end: 12
        guards:
          replaced_sequence: CCCC
  realize:
    mode: full_construct
  output:
    target:
      kind: usr
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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    source:
      kind: literal
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
        orientation: forward
        locator:
          kind: coordinates
          start: 8
          end: 12
        guards:
          replaced_sequence: CCCC
  realize:
    mode: full_construct
  output:
    target:
      kind: usr
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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
  template:
    id: linear_template
    source:
      kind: literal
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
        orientation: forward
        locator:
          kind: coordinates
          start: 8
          end: 12
        guards:
          replaced_sequence: CCCC
  realize:
    mode: full_construct
  output:
    target:
      kind: usr
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
            "  output:\n    target:\n      kind: usr\n      dataset: anchors_constructed\n      root: "
            + usr_root.as_posix(),
            "  output:\n    target:\n      kind: usr\n      dataset: anchors_constructed\n      root: "
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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
    ids: [{anchor_id}, {anchor_id}]
  template:
    id: linear_template
    source:
      kind: literal
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
        orientation: forward
        locator:
          kind: coordinates
          start: 8
          end: 12
        guards:
          replaced_sequence: CCCC
  realize:
    mode: full_construct
  output:
    target:
      kind: usr
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="duplicate planned output id"):
        preflight_from_config(config_path)
