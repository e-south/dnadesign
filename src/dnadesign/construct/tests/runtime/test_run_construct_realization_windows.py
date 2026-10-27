"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_run_construct_realization_windows.py

Window extraction and focal-anchor realization tests for construct.

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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
  template:
    id: circular_template
    source:
      kind: literal
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
        orientation: forward
        locator:
          kind: coordinates
          start: 6
          end: 8
        guards:
          replaced_sequence: CC
  realize:
    mode: window
    focal_part: anchor
    window:
      semantics: fixed_total
      reference: center
      direction: symmetric
      size_bp: 6
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
    assert frame.iloc[0]["sequence"] == "CCGGAA"
    assert bool(frame.iloc[0]["construct__template_circular"]) is True
    assert frame.iloc[0]["construct__window_start"] == 4
    assert frame.iloc[0]["construct__window_end"] == 2
    assert [part["placement_kind"] for part in frame.iloc[0]["construct__parts"]] == ["replace"]


def test_run_construct_supports_negative_window_offset_on_circular_window(tmp_path: Path) -> None:
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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
  template:
    id: circular_template
    source:
      kind: literal
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
        orientation: forward
        locator:
          kind: coordinates
          start: 6
          end: 8
        guards:
          replaced_sequence: CC
  realize:
    mode: window
    focal_part: anchor
    window:
      semantics: fixed_total
      reference: center
      direction: symmetric
      size_bp: 6
      offset_bp: -2
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
    assert frame.iloc[0]["sequence"] == "AACCGG"
    assert frame.iloc[0]["construct__window_start"] == 2
    assert frame.iloc[0]["construct__window_end"] == 0


def test_run_construct_supports_fixed_total_three_prime_window_semantics(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_three_prime.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_three_prime
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
      reference: start
      direction: three_prime
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

    result = run_from_config(config_path)

    assert result.records_total == 1
    frame = Dataset(usr_root, "anchors_constructed").head(n=5)
    assert frame.iloc[0]["sequence"] == "ACGTG"
    assert frame.iloc[0]["construct__window_semantics"] == "fixed_total"
    assert frame.iloc[0]["construct__window_reference"] == "start"
    assert frame.iloc[0]["construct__window_direction"] == "three_prime"
    assert frame.iloc[0]["construct__window_size_bp"] == 5


def test_preflight_rejects_fixed_total_window_that_clips_anchor_handoff_span(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_clipped_anchor_window.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_clipped_anchor_window
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
      reference: start
      direction: five_prime
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

    with pytest.raises(ValidationError, match="construct__anchor_start/end cannot be emitted"):
        preflight_from_config(config_path)


def test_run_construct_supports_anchor_plus_context_window_semantics(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGTAA"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_anchor_plus_context.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_anchor_plus_context
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
      semantics: anchor_plus_context
      upstream_bp: 2
      downstream_bp: 3
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
    assert frame.iloc[0]["sequence"] == "TTACGTAAGGG"
    assert frame.iloc[0]["construct__window_semantics"] == "anchor_plus_context"
    assert frame.iloc[0]["construct__window_upstream_bp"] == 2
    assert frame.iloc[0]["construct__window_downstream_bp"] == 3
    assert frame.iloc[0]["construct__focal_part_length"] == 6
