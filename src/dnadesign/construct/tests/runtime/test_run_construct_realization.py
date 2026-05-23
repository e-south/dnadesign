"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_run_construct_realization.py

Core runtime realization tests for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.construct.src.interfaces.api import run_from_config
from dnadesign.construct.tests.runtime.run_construct_helpers import write_registry as _write_registry
from dnadesign.usr import Dataset


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
    - name: tag
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
    assert frame.iloc[0]["sequence"] == "TTACGTGG"
    assert frame.iloc[0]["construct__input_id"]
    assert frame.iloc[0]["construct__context_id"] == "demo_linear:linear_template"
    assert frame.iloc[0]["construct__context_kind"] == "template"
    assert frame.iloc[0]["construct__anchor_id"] == frame.iloc[0]["construct__input_id"]
    assert frame.iloc[0]["construct__anchor_start"] == 2
    assert frame.iloc[0]["construct__anchor_end"] == 6
    assert frame.iloc[0]["construct__resolved_length"] == len(frame.iloc[0]["sequence"])
    assert frame.iloc[0]["construct__window_semantics"] == "fixed_total"
    assert [part["name"] for part in frame.iloc[0]["construct__parts"]] == ["tag", "anchor"]
    assert frame.iloc[0]["construct__template_kind"] == "literal"


def test_run_construct_tags_usr_events_with_construct_actor(tmp_path: Path) -> None:
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
  id: actor_demo
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

    run_from_config(config_path)

    output_ds = Dataset(usr_root, "anchors_constructed")
    events = [
        json.loads(line) for line in output_ds.events_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    relevant_events = [event for event in events if event["action"] in {"init", "import_rows", "attach"}]

    assert [event["action"] for event in relevant_events] == ["init", "import_rows", "attach"]
    assert all(event["actor"]["tool"] == "construct" for event in relevant_events)
    assert all(event["actor"]["run_id"] == "construct-actor_demo" for event in relevant_events)
