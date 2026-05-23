"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_run_construct_template_sources_and_guards.py

Template source, orientation, and guard tests for construct realization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.construct.src.contracts.errors import ValidationError
from dnadesign.construct.src.interfaces.api import run_from_config
from dnadesign.construct.tests.runtime.run_construct_helpers import write_registry as _write_registry
from dnadesign.usr import Dataset


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
        orientation: reverse_complement
        locator:
          kind: coordinates
          start: 4
          end: 8
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

    result = run_from_config(config_path)

    assert result.records_total == 1
    output_ds = Dataset(usr_root, "anchors_constructed")
    frame = output_ds.head(n=5)
    assert frame.iloc[0]["sequence"] == "AAAAACT"
    assert [part["orientation"] for part in frame.iloc[0]["construct__parts"]] == ["reverse_complement"]


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
          start: 4
          end: 8
        guards:
          replaced_sequence: TTTT
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

    with pytest.raises(ValidationError, match="expected template interval"):
        run_from_config(config_path)


def test_run_construct_accepts_case_insensitive_template_flanks(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["GG"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_flanks.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_flank_contract
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
      sequence: aaaattttccccgggg
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
          start: 4
          end: 8
        guards:
          replaced_sequence: TTTT
          upstream_sequence: AAAA
          downstream_sequence: cCcC
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
    assert str(frame.iloc[0]["sequence"]) == "AAAAGGCCCCGGGG"


def test_run_construct_rejects_mismatched_expected_template_upstream_sequence(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["GG"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_bad_upstream_flank.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_bad_upstream_flank
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
          start: 4
          end: 8
        guards:
          replaced_sequence: TTTT
          upstream_sequence: AAAT
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

    with pytest.raises(ValidationError, match="forward-strand upstream flank"):
        run_from_config(config_path)


def test_run_construct_rejects_mismatched_expected_template_downstream_sequence(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["GG"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_bad_downstream_flank.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_bad_downstream_flank
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
          start: 4
          end: 8
        guards:
          replaced_sequence: TTTT
          downstream_sequence: CCCG
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

    with pytest.raises(ValidationError, match="forward-strand downstream flank"):
        run_from_config(config_path)


def test_run_construct_rejects_non_unique_template_kmer_guards(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["GG"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_non_unique_flank_guard.yaml"
    config_path.write_text(
        f"""
job:
  id: demo_non_unique_flank_guard
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
      sequence: AAAATTTTCCCCAAAAGGGG
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
          start: 4
          end: 8
        guards:
          replaced_sequence: TTTT
          upstream_sequence: AAAA
          require_unique_forward_matches: true
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

    with pytest.raises(ValidationError, match="requires a unique forward-strand match"):
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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
  template:
    id: circular_template
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
    source:
      kind: usr
      dataset: anchors_demo
      root: {usr_root.as_posix()}
    field: sequence
  template:
    id: fasta_template
    source:
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
        orientation: forward
        locator:
          kind: coordinates
          start: 2
          end: 4
        guards:
          replaced_sequence: AA
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

    with pytest.raises(ValidationError, match="exactly one record"):
        run_from_config(config_path)
