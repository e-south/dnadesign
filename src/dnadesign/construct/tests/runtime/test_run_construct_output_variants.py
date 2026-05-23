"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_run_construct_output_variants.py

Output variant and sequence-view runtime tests for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
from Bio.Seq import Seq

from dnadesign.construct.src.contracts.errors import ValidationError
from dnadesign.construct.src.interfaces.api import run_from_config
from dnadesign.construct.tests.runtime.run_construct_helpers import write_registry as _write_registry
from dnadesign.usr import (
    Dataset,
    SequenceViewRecord,
    ensure_sequence_contract_namespaces,
    load_sequence_views,
    write_sequence_views,
)
from dnadesign.usr.src.sequence_views.store import _rows_to_table, _write_sequence_views_atomic, sequence_views_path


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


def test_run_construct_on_conflict_ignore_does_not_duplicate_existing_sequence_id_views(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)

    input_ds = Dataset(usr_root, "anchors_demo")
    input_ds.init(source="test", notes="runtime test")
    input_ds.add_sequences(["GGGG"], bio_type="dna", alphabet="dna_4", source="test")

    output_ds = Dataset(usr_root, "anchors_constructed")
    config_path = tmp_path / "construct_context.yaml"
    config_path.write_text(
        f"""
job:
  id: context
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
          kind: coordinates
          start: 8
          end: 12
  realize:
    mode: full_construct
  output_variants:
    - product_kind: realized_context
      orientation: forward
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

    run_from_config(config_path)
    [current_view] = load_sequence_views(output_ds)
    legacy_payload = current_view.model_dump(mode="python")
    legacy_payload.pop("view_id")
    legacy_payload["derivation_spec_id"] = "legacy_spec"
    legacy_view = SequenceViewRecord.model_validate(legacy_payload)
    _write_sequence_views_atomic(sequence_views_path(output_ds), _rows_to_table([legacy_view]))

    rerun = run_from_config(config_path)
    views = load_sequence_views(output_ds)

    assert rerun.records_written == 0
    assert rerun.records_skipped_existing == 1
    assert len(views) == 1
    assert views[0].sequence_id == current_view.sequence_id
    assert views[0].view_id == legacy_view.view_id


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
