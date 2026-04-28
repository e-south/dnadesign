"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_run_construct.py

Runtime realization tests for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest
from Bio.Seq import Seq

from dnadesign.construct.src.annotations import AnnotationFeature, AnnotationInterval
from dnadesign.construct.src.api import preflight_from_config, run_from_config
from dnadesign.construct.src.errors import ValidationError
from dnadesign.construct.src.feature_retention import classify_feature_retention
from dnadesign.construct.src.output_store import _ensure_construct_registry
from dnadesign.usr import Dataset, ensure_sequence_contract_namespaces, load_sequence_views, write_sequence_views
from dnadesign.usr.src.registry.models import SEQ_ANNOT_COLUMNS
from dnadesign.usr.src.registry.typespec import arrow_type_from_str


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


def _seq_annot_table(*, row_id: str, features: list[dict[str, object]]) -> pa.Table:
    seq_annot_type = next(column.type for column in SEQ_ANNOT_COLUMNS if column.name == "seq_annot__features")
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("seq_annot__features", arrow_type_from_str(seq_annot_type)),
        ]
    )
    return pa.table(
        {
            "id": pa.array([row_id], type=pa.string()),
            "seq_annot__features": pa.array([features], type=arrow_type_from_str(seq_annot_type)),
        },
        schema=schema,
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
    assert str(frame.iloc[0]["sequence"]).upper() == "AAAAGGCCCCGGGG"


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


def test_run_construct_normalize_anchor_selects_annotation_pair_midpoint_and_writes_sequence_view(
    tmp_path: Path,
) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)
    ensure_sequence_contract_namespaces(usr_root)

    input_ds = Dataset(usr_root, "annotated_refs")
    input_ds.init(source="test", notes="normalize anchor test")
    add_result = input_ds.add_sequences(["A" * 80], bio_type="dna", alphabet="dna_4", source="test")
    input_ds.write_overlay(
        "seq_annot",
        _seq_annot_table(
            row_id=add_result.ids[0],
            features=[
                {
                    "feature_id": "minus35",
                    "feature_order": 1,
                    "feature_type": "misc_feature",
                    "label": "-35",
                    "role_hint": "sigma70_minus35",
                    "location_raw": "11..16",
                    "location_kind": "exact",
                    "start_0": 10,
                    "end_0": 16,
                    "strand": 1,
                    "intervals_0": [{"start_0": 10, "end_0": 16, "strand": 1, "partial": False}],
                    "is_fuzzy": False,
                    "is_compound": False,
                    "qualifiers": [],
                    "confidence": "high",
                    "source": "fixture",
                },
                {
                    "feature_id": "minus10",
                    "feature_order": 2,
                    "feature_type": "misc_feature",
                    "label": "-10",
                    "role_hint": "sigma70_minus10",
                    "location_raw": "41..46",
                    "location_kind": "exact",
                    "start_0": 40,
                    "end_0": 46,
                    "strand": 1,
                    "intervals_0": [{"start_0": 40, "end_0": 46, "strand": 1, "partial": False}],
                    "is_fuzzy": False,
                    "is_compound": False,
                    "qualifiers": [],
                    "confidence": "high",
                    "source": "fixture",
                },
            ],
        ),
        key="id",
        overwrite=True,
    )

    config_path = tmp_path / "normalize_anchor.yaml"
    config_path.write_text(
        f"""
job:
  id: normalize_anchor_demo
  mode: normalize_anchor
  input:
    source:
      kind: usr
      dataset: annotated_refs
      root: {usr_root.as_posix()}
    field: sequence
  normalize_anchor:
    product_kind: analysis_window
    target_length: 60
    focal_selector:
      kind: chain
      selectors:
        - kind: annotation_pair_midpoint
          first:
            role_hint: sigma70_minus35
            labels: ["-35"]
          second:
            role_hint: sigma70_minus10
            labels: ["-10"]
          confidence: high
        - kind: sequence_midpoint
          allowed: true
    over_length_policy:
      kind: trim
      target_length: 60
    feature_retention_policy:
      fail_if_loses_roles: [sigma70_minus35, sigma70_minus10]
    emit_feature_retention_report: true
    output_sequence_view:
      create: true
      recommended_pooling: core60_mean
  output:
    target:
      kind: usr
      dataset: normalized_refs
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.records_total == 1
    output_ds = Dataset(usr_root, "normalized_refs")
    frame = output_ds.head(n=5)
    assert len(frame.iloc[0]["sequence"]) == 60
    assert frame.iloc[0]["construct__context_kind"] == "analysis_window"
    assert frame.iloc[0]["derived__source_interval_start_0"] == 0
    assert frame.iloc[0]["derived__source_interval_end_0"] == 60
    assert frame.iloc[0]["derived__focal_rule"] == "annotation_pair_midpoint"
    assert frame.iloc[0]["derived__product_kind"] == "analysis_window"
    assert bool(frame.iloc[0]["derived__analysis_only"]) is True

    views = load_sequence_views(output_ds)
    assert len(views) == 1
    assert views[0].product_kind == "analysis_window"
    assert views[0].recommended_pooling == "core60_mean"
    assert views[0].parent_sequence_id == add_result.ids[0]


def test_run_construct_normalize_anchor_fails_on_ambiguous_annotation_pair(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)
    ensure_sequence_contract_namespaces(usr_root)

    input_ds = Dataset(usr_root, "annotated_refs")
    input_ds.init(source="test", notes="normalize anchor ambiguity test")
    add_result = input_ds.add_sequences(["A" * 80], bio_type="dna", alphabet="dna_4", source="test")
    input_ds.write_overlay(
        "seq_annot",
        _seq_annot_table(
            row_id=add_result.ids[0],
            features=[
                {
                    "feature_id": "minus35a",
                    "feature_order": 1,
                    "feature_type": "misc_feature",
                    "label": "-35",
                    "role_hint": "sigma70_minus35",
                    "location_raw": "6..11",
                    "location_kind": "exact",
                    "start_0": 5,
                    "end_0": 11,
                    "strand": 1,
                    "intervals_0": [{"start_0": 5, "end_0": 11, "strand": 1, "partial": False}],
                    "is_fuzzy": False,
                    "is_compound": False,
                    "qualifiers": [],
                    "confidence": "high",
                    "source": "fixture",
                },
                {
                    "feature_id": "minus35b",
                    "feature_order": 2,
                    "feature_type": "misc_feature",
                    "label": "-35",
                    "role_hint": "sigma70_minus35",
                    "location_raw": "13..18",
                    "location_kind": "exact",
                    "start_0": 12,
                    "end_0": 18,
                    "strand": 1,
                    "intervals_0": [{"start_0": 12, "end_0": 18, "strand": 1, "partial": False}],
                    "is_fuzzy": False,
                    "is_compound": False,
                    "qualifiers": [],
                    "confidence": "high",
                    "source": "fixture",
                },
                {
                    "feature_id": "minus10",
                    "feature_order": 3,
                    "feature_type": "misc_feature",
                    "label": "-10",
                    "role_hint": "sigma70_minus10",
                    "location_raw": "41..46",
                    "location_kind": "exact",
                    "start_0": 40,
                    "end_0": 46,
                    "strand": 1,
                    "intervals_0": [{"start_0": 40, "end_0": 46, "strand": 1, "partial": False}],
                    "is_fuzzy": False,
                    "is_compound": False,
                    "qualifiers": [],
                    "confidence": "high",
                    "source": "fixture",
                },
            ],
        ),
        key="id",
        overwrite=True,
    )

    config_path = tmp_path / "normalize_anchor_ambiguous.yaml"
    config_path.write_text(
        f"""
job:
  id: normalize_anchor_ambiguous
  mode: normalize_anchor
  input:
    source:
      kind: usr
      dataset: annotated_refs
      root: {usr_root.as_posix()}
    field: sequence
  normalize_anchor:
    product_kind: analysis_window
    target_length: 60
    focal_selector:
      kind: chain
      selectors:
        - kind: annotation_pair_midpoint
          first:
            role_hint: sigma70_minus35
            labels: ["-35"]
          second:
            role_hint: sigma70_minus10
            labels: ["-10"]
    over_length_policy:
      kind: trim
      target_length: 60
  output:
    target:
      kind: usr
      dataset: normalized_refs
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="matched 2 features"):
        run_from_config(config_path)


def test_run_construct_normalize_anchor_expands_short_sequence_from_template(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)
    ensure_sequence_contract_namespaces(usr_root)

    short_anchor = "ACGT" * 8 + "ACG"
    input_ds = Dataset(usr_root, "short_refs")
    input_ds.init(source="test", notes="normalize anchor short test")
    input_ds.add_sequences([short_anchor], bio_type="dna", alphabet="dna_4", source="test")

    template_sequence = "A" * 15 + short_anchor + "C" * 10
    config_path = tmp_path / "normalize_anchor_expand.yaml"
    config_path.write_text(
        f"""
job:
  id: normalize_anchor_expand
  mode: normalize_anchor
  input:
    source:
      kind: usr
      dataset: short_refs
      root: {usr_root.as_posix()}
    field: sequence
  normalize_anchor:
    product_kind: analysis_window
    target_length: 60
    focal_selector:
      kind: chain
      selectors:
        - kind: sequence_midpoint
          allowed: true
    fallback_policy:
      allow_low_confidence: true
    over_length_policy:
      kind: trim
      target_length: 60
    under_length_policy:
      kind: expand_from_template
      target_length: 60
      template:
        source:
          kind: literal
          sequence: {template_sequence}
      placement_ref: template_fixture
    emit_feature_retention_report: true
  output:
    target:
      kind: usr
      dataset: normalized_refs
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    run_from_config(config_path)

    frame = Dataset(usr_root, "normalized_refs").head(n=5)
    assert len(frame.iloc[0]["sequence"]) == 60
    assert bool(frame.iloc[0]["derived__analysis_only"]) is True
    assert frame.iloc[0]["derived__added_left_bp"] == 15
    assert frame.iloc[0]["derived__added_right_bp"] == 10


def test_run_construct_normalize_anchor_placement_ref_disambiguates_duplicate_template_match(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)
    ensure_sequence_contract_namespaces(usr_root)

    short_anchor = "ACGT" * 8 + "ACG"
    input_ds = Dataset(usr_root, "short_refs")
    input_ds.init(source="test", notes="normalize anchor placement-ref test")
    input_ds.add_sequences([short_anchor], bio_type="dna", alphabet="dna_4", source="test")

    template_sequence = "T" * 5 + short_anchor + "G" * 5 + short_anchor + "C" * 20
    config_path = tmp_path / "normalize_anchor_expand_offset.yaml"
    config_path.write_text(
        f"""
job:
  id: normalize_anchor_expand_offset
  mode: normalize_anchor
  input:
    source:
      kind: usr
      dataset: short_refs
      root: {usr_root.as_posix()}
    field: sequence
  normalize_anchor:
    product_kind: analysis_window
    target_length: 60
    focal_selector:
      kind: chain
      selectors:
        - kind: sequence_midpoint
          allowed: true
    fallback_policy:
      allow_low_confidence: true
    over_length_policy:
      kind: trim
      target_length: 60
    under_length_policy:
      kind: expand_from_template
      target_length: 60
      template:
        source:
          kind: literal
          sequence: {template_sequence}
      placement_ref: offset:5
  output:
    target:
      kind: usr
      dataset: normalized_refs
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    run_from_config(config_path)

    frame = Dataset(usr_root, "normalized_refs").head(n=5)
    assert frame.iloc[0]["sequence"] == template_sequence[:60]
    assert frame.iloc[0]["derived__added_left_bp"] == 5
    assert frame.iloc[0]["derived__added_right_bp"] == 20


def test_run_construct_normalize_anchor_expands_short_sequence_by_replacing_template_interval(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)
    ensure_sequence_contract_namespaces(usr_root)

    short_anchor = "TTGACA" + "G" * 17 + "TATAAT" + "C" * 6
    input_ds = Dataset(usr_root, "short_refs")
    input_ds.init(source="test", notes="normalize anchor replacement test")
    add_result = input_ds.add_sequences([short_anchor], bio_type="dna", alphabet="dna_4", source="test")
    input_ds.write_overlay(
        "seq_annot",
        _seq_annot_table(
            row_id=add_result.ids[0],
            features=[
                {
                    "feature_id": "minus35",
                    "feature_order": 1,
                    "feature_type": "misc_feature",
                    "label": "-35",
                    "role_hint": "sigma70_minus35",
                    "location_raw": "1..6",
                    "location_kind": "exact",
                    "start_0": 0,
                    "end_0": 6,
                    "strand": 1,
                    "intervals_0": [{"start_0": 0, "end_0": 6, "strand": 1, "partial": False}],
                    "is_fuzzy": False,
                    "is_compound": False,
                    "qualifiers": [],
                    "confidence": "high",
                    "source": "fixture",
                },
                {
                    "feature_id": "minus10",
                    "feature_order": 2,
                    "feature_type": "misc_feature",
                    "label": "-10",
                    "role_hint": "sigma70_minus10",
                    "location_raw": "24..29",
                    "location_kind": "exact",
                    "start_0": 23,
                    "end_0": 29,
                    "strand": 1,
                    "intervals_0": [{"start_0": 23, "end_0": 29, "strand": 1, "partial": False}],
                    "is_fuzzy": False,
                    "is_compound": False,
                    "qualifiers": [],
                    "confidence": "high",
                    "source": "fixture",
                },
            ],
        ),
        key="id",
        overwrite=True,
    )

    template_sequence = "G" * 30 + "A" * 92 + "C" * 30
    config_path = tmp_path / "normalize_anchor_replace_interval.yaml"
    config_path.write_text(
        f"""
job:
  id: normalize_anchor_replace_interval
  mode: normalize_anchor
  input:
    source:
      kind: usr
      dataset: short_refs
      root: {usr_root.as_posix()}
    field: sequence
  normalize_anchor:
    product_kind: analysis_window
    target_length: 60
    focal_selector:
      kind: chain
      selectors:
        - kind: annotation_pair_midpoint
          first:
            role_hint: sigma70_minus35
            labels: ["-35"]
          second:
            role_hint: sigma70_minus10
            labels: ["-10"]
          confidence: high
    over_length_policy:
      kind: trim
      target_length: 60
    under_length_policy:
      kind: expand_from_template
      target_length: 60
      template:
        source:
          kind: literal
          sequence: {template_sequence}
      placement_ref: replace:30-122
    feature_retention_policy:
      fail_if_loses_roles: [sigma70_minus35, sigma70_minus10]
    emit_feature_retention_report: true
  output:
    target:
      kind: usr
      dataset: normalized_refs
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    run_from_config(config_path)

    frame = Dataset(usr_root, "normalized_refs").head(n=5)
    assert frame.iloc[0]["sequence"] == "G" * 16 + short_anchor + "C" * 9
    assert frame.iloc[0]["derived__added_left_bp"] == 16
    assert frame.iloc[0]["derived__added_right_bp"] == 9
    assert frame.iloc[0]["derived__focal_rule"] == "annotation_pair_midpoint"


def test_run_construct_normalize_anchor_circular_expansion_wraps_left_context(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)
    ensure_sequence_contract_namespaces(usr_root)

    short_anchor = "AACCGGTT"
    input_ds = Dataset(usr_root, "short_refs")
    input_ds.init(source="test", notes="normalize anchor circular wrap test")
    input_ds.add_sequences([short_anchor], bio_type="dna", alphabet="dna_4", source="test")

    template_sequence = "GG" + short_anchor + "TTTTCCCCAAAAGG"
    expected = template_sequence[-4:] + template_sequence[:16]
    config_path = tmp_path / "normalize_anchor_expand_circular.yaml"
    config_path.write_text(
        f"""
job:
  id: normalize_anchor_expand_circular
  mode: normalize_anchor
  input:
    source:
      kind: usr
      dataset: short_refs
      root: {usr_root.as_posix()}
    field: sequence
  normalize_anchor:
    product_kind: analysis_window
    target_length: 20
    focal_selector:
      kind: chain
      selectors:
        - kind: sequence_midpoint
          allowed: true
    fallback_policy:
      allow_low_confidence: true
    over_length_policy:
      kind: trim
      target_length: 20
    under_length_policy:
      kind: expand_from_template
      target_length: 20
      template:
        source:
          kind: literal
          sequence: {template_sequence}
        circular: true
      placement_ref: offset:2
  output:
    target:
      kind: usr
      dataset: normalized_refs
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    run_from_config(config_path)

    frame = Dataset(usr_root, "normalized_refs").head(n=5)
    assert frame.iloc[0]["sequence"] == expected
    assert frame.iloc[0]["derived__added_left_bp"] == 6
    assert frame.iloc[0]["derived__added_right_bp"] == 6


def test_construct_feature_retention_counts_lost_compound_intervals_as_clipped_bp() -> None:
    feature = AnnotationFeature(
        feature_id="compound_tfbs",
        feature_order=1,
        feature_type="misc_feature",
        label="compound_tfbs",
        role_hint="TFBS",
        start_0=5,
        end_0=40,
        intervals_0=(
            AnnotationInterval(start_0=5, end_0=10, strand=1, partial=False),
            AnnotationInterval(start_0=35, end_0=40, strand=1, partial=False),
        ),
        confidence="high",
    )

    retention = classify_feature_retention(
        features=[feature],
        source_start_0=0,
        source_end_0=20,
    )

    assert retention.clipped[0]["clipped_bp"] == 5
    assert retention.clipped[0]["derived_intervals_0"] == [{"start_0": 5, "end_0": 10, "strand": 1, "partial": False}]


def test_run_construct_normalize_anchor_expansion_offsets_feature_retention_coordinates(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)
    ensure_sequence_contract_namespaces(usr_root)

    short_anchor = "ACGT" * 8 + "ACG"
    input_ds = Dataset(usr_root, "short_refs")
    input_ds.init(source="test", notes="normalize anchor retention offset test")
    add_result = input_ds.add_sequences([short_anchor], bio_type="dna", alphabet="dna_4", source="test")
    input_ds.write_overlay(
        "seq_annot",
        _seq_annot_table(
            row_id=add_result.ids[0],
            features=[
                {
                    "feature_id": "anchor_feature",
                    "feature_order": 1,
                    "feature_type": "misc_feature",
                    "label": "anchor_feature",
                    "role_hint": "TFBS",
                    "location_raw": "6..10",
                    "location_kind": "exact",
                    "start_0": 5,
                    "end_0": 10,
                    "strand": 1,
                    "intervals_0": [{"start_0": 5, "end_0": 10, "strand": 1, "partial": False}],
                    "is_fuzzy": False,
                    "is_compound": False,
                    "qualifiers": [],
                    "confidence": "high",
                    "source": "fixture",
                }
            ],
        ),
        key="id",
        overwrite=True,
    )

    template_sequence = "A" * 15 + short_anchor + "C" * 10
    config_path = tmp_path / "normalize_anchor_expand_retention.yaml"
    config_path.write_text(
        f"""
job:
  id: normalize_anchor_expand_retention
  mode: normalize_anchor
  input:
    source:
      kind: usr
      dataset: short_refs
      root: {usr_root.as_posix()}
    field: sequence
  normalize_anchor:
    product_kind: analysis_window
    target_length: 60
    focal_selector:
      kind: chain
      selectors:
        - kind: sequence_midpoint
          allowed: true
    fallback_policy:
      allow_low_confidence: true
    over_length_policy:
      kind: trim
      target_length: 60
    under_length_policy:
      kind: expand_from_template
      target_length: 60
      template:
        source:
          kind: literal
          sequence: {template_sequence}
      placement_ref: template_fixture
    emit_feature_retention_report: true
  output:
    target:
      kind: usr
      dataset: normalized_refs
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    run_from_config(config_path)

    frame = Dataset(usr_root, "normalized_refs").head(n=5)
    retained = frame.iloc[0]["derived__features_retained"]
    assert retained[0]["derived_intervals_0"] == [{"start_0": 20, "end_0": 25, "strand": 1, "partial": False}]


def test_run_construct_output_variants_emit_forward_and_reverse_complement_views(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["AGTC"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_variants.yaml"
    config_path.write_text(
        f"""
job:
  id: context_variants
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
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.records_total == 2
    output_ds = Dataset(usr_root, "anchors_constructed")
    frame = output_ds.head(n=10).sort_values("construct__orientation").reset_index(drop=True)
    assert list(frame["construct__orientation"]) == ["forward", "reverse_complement"]
    forward_row = frame.iloc[0]
    rc_row = frame.iloc[1]
    assert forward_row["sequence"] == "AAAATTTTAGTCGGGG"
    assert rc_row["sequence"] == str(Seq(forward_row["sequence"]).reverse_complement())
    assert rc_row["sequence"] == "CCCCGACTAAAATTTT"
    assert forward_row["construct__anchor_start"] == 8
    assert forward_row["construct__anchor_end"] == 12
    assert rc_row["construct__anchor_start"] == 4
    assert rc_row["construct__anchor_end"] == 8
    assert rc_row["construct__forward_anchor_start"] == 8
    assert rc_row["construct__forward_anchor_end"] == 12

    views = sorted(load_sequence_views(output_ds), key=lambda view: view.orientation)
    assert [view.orientation for view in views] == ["forward", "reverse_complement"]
    assert [view.product_kind for view in views] == ["realized_context", "realized_context"]
    assert [view.recommended_pooling for view in views] == ["anchor_mean", "anchor_mean"]
    assert [(view.anchor_start_0, view.anchor_end_0) for view in views] == [(8, 12), (4, 8)]
    assert [(view.forward_anchor_start_0, view.forward_anchor_end_0) for view in views] == [
        (8, 12),
        (8, 12),
    ]


def test_run_construct_output_variants_complete_views_for_existing_forward_rows(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["AGTC"], bio_type="dna", alphabet="dna_4", source="test")

    legacy_config_path = tmp_path / "legacy_forward_context.yaml"
    legacy_config_path.write_text(
        f"""
job:
  id: legacy_forward_context
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
  realize:
    mode: full_construct
  output:
    on_conflict: ignore
    target:
      kind: usr
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )
    run_from_config(legacy_config_path)
    output_ds = Dataset(usr_root, "anchors_constructed")
    assert load_sequence_views(output_ds) == []

    variant_config_path = tmp_path / "complete_context_variants.yaml"
    variant_config_path.write_text(
        f"""
job:
  id: complete_context_variants
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
  realize:
    mode: full_construct
  output_variants:
    - product_kind: realized_context
      orientation: forward
      recommended_pooling: anchor_mean
    - product_kind: realized_context
      orientation: reverse_complement
      recommended_pooling: anchor_mean
  output:
    on_conflict: ignore
    target:
      kind: usr
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(variant_config_path)

    assert result.records_total == 2
    assert result.records_written == 1
    assert result.records_skipped_existing == 1
    views = sorted(load_sequence_views(output_ds), key=lambda view: view.orientation)
    assert [view.orientation for view in views] == ["forward", "reverse_complement"]
    assert [view.product_kind for view in views] == ["realized_context", "realized_context"]
    assert [(view.anchor_start_0, view.anchor_end_0) for view in views] == [(8, 12), (4, 8)]

    rerun = run_from_config(variant_config_path)

    assert rerun.records_total == 2
    assert rerun.records_written == 0
    assert rerun.records_skipped_existing == 2
    assert len(load_sequence_views(output_ds)) == 2

    drifted_view = views[0].model_copy(update={"recommended_pooling": "seq_mean"})
    write_sequence_views(output_ds, [drifted_view], conflict_policy="replace")
    with pytest.raises(ValidationError, match="already exists with different metadata"):
        run_from_config(variant_config_path)


def test_run_construct_output_variants_allow_same_sequence_with_distinct_views(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "construct_palindromic_variants.yaml"
    config_path.write_text(
        f"""
job:
  id: context_variants
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
      sequence: CCCC
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
          replaced_sequence: CCCC
  realize:
    mode: full_construct
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
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    output_ds = Dataset(usr_root, "anchors_constructed")
    assert result.records_total == 2
    assert output_ds.head(n=10).shape[0] == 1
    views = sorted(load_sequence_views(output_ds), key=lambda view: view.orientation)
    assert [view.orientation for view in views] == ["forward", "reverse_complement"]
    assert views[0].sequence_id == views[1].sequence_id


def test_run_construct_output_variants_make_carried_aliases_variant_specific(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)
    ensure_sequence_contract_namespaces(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    add_result = input_ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")
    input_ds.write_overlay(
        "usr_label",
        pa.table(
            {
                "id": pa.array([add_result.ids[0]], type=pa.string()),
                "usr_label__primary": pa.array(["anchor_label"], type=pa.string()),
                "usr_label__aliases": pa.array([["legacy_anchor"]], type=pa.list_(pa.string())),
            }
        ),
        key="id",
        overwrite=True,
    )

    config_path = tmp_path / "construct_labeled_variants.yaml"
    config_path.write_text(
        f"""
job:
  id: context_variants
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
      dataset: anchors_constructed
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    run_from_config(config_path)

    output_ds = Dataset(usr_root, "anchors_constructed")
    views = sorted(load_sequence_views(output_ds), key=lambda view: view.orientation)
    assert [view.view_name for view in views] == [
        "anchor_label_realized_context_forward",
        "anchor_label_realized_context_reverse_complement",
    ]
    assert [view.aliases for view in views] == [
        ["legacy_anchor_realized_context_forward"],
        ["legacy_anchor_realized_context_reverse_complement"],
    ]
