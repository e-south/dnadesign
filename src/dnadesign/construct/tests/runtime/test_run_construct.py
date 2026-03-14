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

from dnadesign.construct.src.api import run_from_config
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
    assert list(frame.iloc[0]["construct__part_kinds"]) == ["replace"]


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
