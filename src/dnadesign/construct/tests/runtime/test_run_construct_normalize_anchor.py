"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_run_construct_normalize_anchor.py

End-to-end normalize-anchor runtime tests for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.construct.src.annotations.features import AnnotationFeature, AnnotationInterval
from dnadesign.construct.src.annotations.retention import classify_feature_retention
from dnadesign.construct.src.contracts.errors import ValidationError
from dnadesign.construct.src.interfaces.api import run_from_config
from dnadesign.construct.tests.runtime.run_construct_helpers import seq_annot_table as _seq_annot_table
from dnadesign.construct.tests.runtime.run_construct_helpers import write_registry as _write_registry
from dnadesign.usr import Dataset, ensure_sequence_contract_namespaces, load_sequence_views


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


def test_run_construct_normalize_anchor_selects_fixed_upstream_window_from_sequence_offset(
    tmp_path: Path,
) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)
    ensure_sequence_contract_namespaces(usr_root)

    input_ds = Dataset(usr_root, "native_windows")
    input_ds.init(source="test", notes="fixed upstream window test")
    source_sequence = ("ACGT" * 15) + "T" + ("GCTA" * 5)
    add_result = input_ds.add_sequences([source_sequence], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "normalize_tss_upstream.yaml"
    config_path.write_text(
        f"""
job:
  id: normalize_tss_upstream_demo
  mode: normalize_anchor
  input:
    source:
      kind: usr
      dataset: native_windows
      root: {usr_root.as_posix()}
    field: sequence
  normalize_anchor:
    product_kind: analysis_window
    target_length: 60
    focal_selector:
      kind: chain
      selectors:
        - kind: sequence_offset
          offset_0: 60
          label: tss_offset_0
          confidence: high
    over_length_policy:
      kind: trim
      target_length: 60
      require_focal_inside: false
      window_anchor: upstream_of_focal
    output_sequence_view:
      create: true
      recommended_pooling: core60_mean
  output:
    target:
      kind: usr
      dataset: tss_upstream_core60
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    result = run_from_config(config_path)

    assert result.records_total == 1
    output_ds = Dataset(usr_root, "tss_upstream_core60")
    frame = output_ds.head(n=5)
    assert frame.iloc[0]["sequence"].upper() == source_sequence[:60]
    assert frame.iloc[0]["derived__source_interval_start_0"] == 0
    assert frame.iloc[0]["derived__source_interval_end_0"] == 60
    assert frame.iloc[0]["derived__focal_rule"] == "sequence_offset"
    assert frame.iloc[0]["derived__focal_features"] == ["tss_offset_0"]
    assert frame.iloc[0]["construct__window_direction"] == "upstream"
    assert frame.iloc[0]["construct__window_upstream_bp"] == 60
    assert frame.iloc[0]["construct__window_downstream_bp"] == 0

    views = load_sequence_views(output_ds)
    assert len(views) == 1
    assert views[0].product_kind == "analysis_window"
    assert views[0].parent_sequence_id == add_result.ids[0]
    assert views[0].source_interval_start_0 == 0
    assert views[0].source_interval_end_0 == 60


def test_run_construct_normalize_anchor_fails_when_upstream_offset_lacks_coverage(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    usr_root.mkdir(parents=True, exist_ok=True)
    _write_registry(usr_root)
    ensure_sequence_contract_namespaces(usr_root)

    input_ds = Dataset(usr_root, "short_upstream_window")
    input_ds.init(source="test", notes="short upstream window test")
    input_ds.add_sequences(["A" * 81], bio_type="dna", alphabet="dna_4", source="test")

    config_path = tmp_path / "normalize_tss_upstream_short.yaml"
    config_path.write_text(
        f"""
job:
  id: normalize_tss_upstream_short_demo
  mode: normalize_anchor
  input:
    source:
      kind: usr
      dataset: short_upstream_window
      root: {usr_root.as_posix()}
    field: sequence
  normalize_anchor:
    product_kind: analysis_window
    target_length: 60
    focal_selector:
      kind: chain
      selectors:
        - kind: sequence_offset
          offset_0: 59
          label: tss_offset_0
    over_length_policy:
      kind: trim
      target_length: 60
      require_focal_inside: false
      window_anchor: upstream_of_focal
  output:
    target:
      kind: usr
      dataset: tss_upstream_core60
      root: {usr_root.as_posix()}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="60 upstream bases"):
        run_from_config(config_path)


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
