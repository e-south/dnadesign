"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_run_construct_persistence.py

Runtime persistence, label carry-forward, flank, and focal-anchor tests for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from dnadesign.construct.src.contracts.errors import ValidationError
from dnadesign.construct.src.interfaces.api import run_from_config
from dnadesign.construct.src.persistence.usr_registry import _ensure_construct_registry
from dnadesign.construct.tests.runtime.run_construct_helpers import write_registry as _write_registry
from dnadesign.usr import Dataset


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


def test_run_construct_carries_forward_upstream_usr_labels(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    _ensure_construct_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    with input_ds.write_session() as session:
        session.init_if_missing(source="test", notes="runtime test")
        result = session.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")
        session.write_overlay(
            "usr_label",
            pa.table(
                {
                    "id": [result.ids[0]],
                    "usr_label__primary": ["J23105"],
                    "usr_label__aliases": [["BBa_J23105"]],
                }
            ),
            overwrite=True,
            note="test labels",
        )

    config_path = tmp_path / "construct_with_labels.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_labels
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
      size_bp: 8
      offset_bp: 0
  output:
    target:
      kind: usr
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.records_total == 1
    output_ds = Dataset(usr_root, "anchors_constructed")
    frame = output_ds.head(n=5)
    assert frame.iloc[0]["usr_label__primary"] == "J23105"
    assert frame.iloc[0]["usr_label__aliases"] == ["BBa_J23105"]
    assert frame.iloc[0]["construct__input_id"]


def test_run_construct_resolves_flank_locator_replace(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["AC"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_flanks.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_flanks
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
      sequence: AAAACCCCGGGGTTTT
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
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.records_total == 1
    frame = Dataset(usr_root, "anchors_constructed").head(n=5)
    assert frame.iloc[0]["sequence"] == "AAAAACGGGGTTTT"
    assert frame.iloc[0]["construct__parts"][0]["template_start"] == 4
    assert frame.iloc[0]["construct__parts"][0]["template_end"] == 8


def test_run_construct_rejects_flank_locator_with_repeated_kmer(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["AC"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_flanks_repeated.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_flanks_repeated
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
      sequence: AAAACCCCAAAAGGGG
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
          kind: flanks
          upstream_sequence: AAAA
          downstream_sequence: GGGG
        guards:
          replaced_span_bp: 4
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

    with pytest.raises(
        ValidationError,
        match="requires exactly one forward-strand match for placement.locator.upstream_sequence",
    ):
        run_from_config(config_path)


def test_run_construct_uses_explicit_focal_part_for_multi_anchor_full_construct(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_multi_anchor_explicit_focal.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_multi_anchor_explicit_focal
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
    - name: anchor_a
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
    - name: anchor_b
      role: anchor
      sequence:
        source: literal
        literal: GG
      placement:
        kind: replace
        orientation: forward
        locator:
          kind: coordinates
          start: 12
          end: 16
        guards:
          replaced_sequence: GGGG
  realize:
    mode: full_construct
    focal_part: anchor_b
  output:
    target:
      kind: usr
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.records_total == 1
    frame = Dataset(usr_root, "anchors_constructed").head(n=5)
    assert frame.iloc[0]["sequence"] == "AAAAACGTCCCCGG"
    assert frame.iloc[0]["construct__anchor_start"] == 12
    assert frame.iloc[0]["construct__anchor_end"] == 14
    assert frame.iloc[0]["construct__focal_part"] == "anchor_b"
    assert [
        (slot["slot_id"], slot["role"], slot["start"], slot["end"]) for slot in frame.iloc[0]["construct__slots"]
    ] == [
        ("anchor_a", "anchor", 4, 8),
        ("anchor_b", "anchor", 12, 14),
    ]
