"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_multi_slot_assembly.py

Runtime tests for public multi-slot Construct assembly.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
import yaml
from Bio.Seq import Seq

from dnadesign.construct.src.contracts.errors import ValidationError
from dnadesign.construct.src.interfaces.api import run_from_config
from dnadesign.construct.src.persistence.usr_registry import _ensure_construct_registry
from dnadesign.usr import Dataset


def _register_candidate_slot_overlay(root: Path) -> None:
    registry_path = root / "registry.yaml"
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    namespaces = payload.setdefault("namespaces", {})
    namespaces["candidate"] = {
        "owner": "study",
        "description": "Test candidate slot sequence overlay.",
        "columns": [
            {"name": "candidate__lnrna_sequence", "type": "string"},
            {"name": "candidate__rt_cds_sequence", "type": "string"},
        ],
    }
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _candidate_dataset(root: Path) -> tuple[Dataset, str]:
    _ensure_construct_registry(root)
    _register_candidate_slot_overlay(root)
    dataset = Dataset(root, "rt_lnrna_candidates")
    dataset.init(source="test", notes="multi-slot candidate rows")
    add_result = dataset.add_sequences(["A"], bio_type="dna", alphabet="dna_4", source="candidate-id-carrier")
    candidate_id = add_result.ids[0]
    dataset.write_overlay(
        "candidate",
        pa.table(
            {
                "id": pa.array([candidate_id], type=pa.string()),
                "candidate__lnrna_sequence": pa.array(["GG"], type=pa.string()),
                "candidate__rt_cds_sequence": pa.array(["AATTAA"], type=pa.string()),
            }
        ),
        key="id",
        overwrite=True,
    )
    return dataset, candidate_id


def test_run_construct_assembles_named_input_slots_without_precomposed_anchor(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _candidate_dataset(usr_root)

    config_path = tmp_path / "construct_multi_slot.yaml"
    config_path.write_text(
        f"""
job:
  id: rt_lnrna_multi_slot
  input:
    source:
      kind: usr
      dataset: rt_lnrna_candidates
      root: {usr_root.as_posix()}
    field: null
  template:
    id: dual_cassette_template
    source:
      kind: literal
      sequence: AAAACCCCGGGGTTTT
    circular: false
  parts:
    - name: lnrna
      role: lnrna_cassette
      sequence:
        source: input_field
        field: candidate__lnrna_sequence
      placement:
        kind: replace
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 8
        guards:
          replaced_sequence: CCCC
    - name: rt_cds
      role: rt_cds
      sequence:
        source: input_field
        field: candidate__rt_cds_sequence
      placement:
        kind: replace
        orientation: forward
        locator:
          kind: coordinates
          start: 12
          end: 16
        guards:
          replaced_sequence: TTTT
  realize:
    mode: full_construct
    focal_part: lnrna
    required_slots: [lnrna, rt_cds]
  output_variants:
    - product_kind: realized_context
      orientation: forward
      recommended_pooling: anchor_mean
    - product_kind: realized_context
      orientation: reverse_complement
      recommended_pooling: anchor_mean
  output:
    target:
      kind: usr
      dataset: rt_lnrna_constructs
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.records_total == 2
    output = Dataset(usr_root, "rt_lnrna_constructs").head(n=10)
    frame = output.sort_values("construct__orientation").reset_index(drop=True)
    forward = frame.iloc[0]
    reverse = frame.iloc[1]
    assert forward["sequence"] == "AAAAGGGGGGAATTAA"
    assert reverse["sequence"] == str(Seq(forward["sequence"]).reverse_complement())
    assert forward["construct__assembly_mode"] == "multi_slot"
    assert forward["construct__slot_count"] == 2
    assert list(forward["construct__input_fields"]) == [
        "candidate__lnrna_sequence",
        "candidate__rt_cds_sequence",
    ]
    assert forward["construct__input_length"] == 8
    assert forward["construct__anchor_start"] == 4
    assert forward["construct__anchor_end"] == 6
    assert [
        (slot["slot_id"], slot["role"], slot["sequence_field"], slot["start"], slot["end"])
        for slot in forward["construct__slots"]
    ] == [
        ("lnrna", "lnrna_cassette", "candidate__lnrna_sequence", 4, 6),
        ("rt_cds", "rt_cds", "candidate__rt_cds_sequence", 10, 16),
    ]
    assert [
        (slot["slot_id"], slot["start"], slot["end"], slot["forward_start"], slot["forward_end"])
        for slot in reverse["construct__slots"]
    ] == [
        ("lnrna", 10, 12, 4, 6),
        ("rt_cds", 0, 6, 10, 16),
    ]


def test_run_construct_rejects_window_that_clips_required_slot(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _candidate_dataset(usr_root)

    config_path = tmp_path / "construct_clipped_required_slot.yaml"
    config_path.write_text(
        f"""
job:
  id: rt_lnrna_multi_slot_clipped
  input:
    source:
      kind: usr
      dataset: rt_lnrna_candidates
      root: {usr_root.as_posix()}
    field: null
  template:
    id: dual_cassette_template
    source:
      kind: literal
      sequence: AAAACCCCGGGGTTTT
    circular: false
  parts:
    - name: lnrna
      role: lnrna_cassette
      sequence:
        source: input_field
        field: candidate__lnrna_sequence
      placement:
        kind: replace
        orientation: forward
        locator:
          kind: coordinates
          start: 4
          end: 8
        guards:
          replaced_sequence: CCCC
    - name: rt_cds
      role: rt_cds
      sequence:
        source: input_field
        field: candidate__rt_cds_sequence
      placement:
        kind: replace
        orientation: forward
        locator:
          kind: coordinates
          start: 12
          end: 16
        guards:
          replaced_sequence: TTTT
  realize:
    mode: window
    focal_part: lnrna
    required_slots: [lnrna, rt_cds]
    window:
      semantics: fixed_total
      reference: start
      direction: three_prime
      size_bp: 8
      offset_bp: 0
  output:
    target:
      kind: usr
      dataset: rt_lnrna_constructs
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="required slot 'rt_cds'"):
        run_from_config(config_path)
