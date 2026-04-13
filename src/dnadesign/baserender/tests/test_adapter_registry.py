"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_adapter_registry.py

Adapter registry tests for centralized factory and required-source-column contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.baserender.src.adapters import build_adapter, required_source_columns
from dnadesign.baserender.src.adapters.cruncher_best_window import CruncherBestWindowAdapter
from dnadesign.baserender.src.adapters.duplex_sequence_v1 import DuplexSequenceV1Adapter
from dnadesign.baserender.src.adapters.hairpin_topology_v1 import HairpinTopologyV1Adapter
from dnadesign.baserender.src.adapters.sequence_evidence_map_v1 import (
    SequenceEvidenceMapV1Adapter,
    _style_token_for_owner,
    _style_token_for_tag,
)
from dnadesign.baserender.src.adapters.sequence_windows_v1 import SequenceWindowsV1Adapter
from dnadesign.baserender.src.adapters.yiu_hairpin_topology_v1 import (
    YiuHairpinTopologyV1Adapter,
)
from dnadesign.baserender.src.adapters.yiu_hairpin_topology_v1 import (
    _span as _yiu_hairpin_span,
)
from dnadesign.baserender.src.adapters.yiu_linear_state_v1 import YiuLinearStateV1Adapter
from dnadesign.baserender.src.adapters.yiu_topology_cartoon_v1 import YiuTopologyCartoonV1Adapter
from dnadesign.baserender.src.config import AdapterCfg
from dnadesign.baserender.src.config.adapter_contracts import adapter_descriptor
from dnadesign.baserender.src.core import SchemaError
from dnadesign.baserender.src.render import legend_entries_for_record

from .conftest import write_parquet


def test_required_source_columns_densegen_includes_optional_present_columns() -> None:
    cfg = AdapterCfg(
        kind="densegen_tfbs",
        columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
            "overlay_text": "details",
            "video_subtitle": "subtitle",
        },
        policies={},
    )
    assert required_source_columns(cfg) == ["sequence", "densegen__used_tfbs_detail", "id", "details", "subtitle"]


def test_required_source_columns_generic_features_omits_missing_optional_columns() -> None:
    cfg = AdapterCfg(
        kind="generic_features",
        columns={
            "sequence": "sequence",
            "features": "features",
        },
        policies={},
    )
    assert required_source_columns(cfg) == ["sequence", "features"]


def test_required_source_columns_unknown_kind_is_schema_error() -> None:
    cfg = AdapterCfg(kind="unknown_kind", columns={}, policies={})
    with pytest.raises(SchemaError, match="Unsupported adapter kind"):
        required_source_columns(cfg)


def test_required_source_columns_missing_required_key_is_schema_error() -> None:
    cfg = AdapterCfg(
        kind="densegen_tfbs",
        columns={"annotations": "densegen__used_tfbs_detail"},
        policies={},
    )
    with pytest.raises(SchemaError, match="missing required adapter column key"):
        required_source_columns(cfg)


def test_required_source_columns_densegen_accepts_overlay_text_optional_key() -> None:
    cfg = AdapterCfg(
        kind="densegen_tfbs",
        columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
            "overlay_text": "details",
        },
        policies={},
    )
    assert required_source_columns(cfg) == ["sequence", "densegen__used_tfbs_detail", "id", "details"]


def test_required_source_columns_densegen_accepts_video_subtitle_optional_key() -> None:
    cfg = AdapterCfg(
        kind="densegen_tfbs",
        columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
            "video_subtitle": "subtitle",
        },
        policies={},
    )
    assert required_source_columns(cfg) == ["sequence", "densegen__used_tfbs_detail", "id", "subtitle"]


def test_generic_features_adapter_accepts_display_video_subtitle() -> None:
    cfg = AdapterCfg(
        kind="generic_features",
        columns={
            "id": "id",
            "sequence": "sequence",
            "features": "features",
            "display": "display",
        },
        policies={},
    )
    adapter = build_adapter(cfg, alphabet="DNA")
    record = adapter.apply(
        {
            "id": "row-1",
            "sequence": "ACGT",
            "features": [],
            "display": {
                "overlay_text": None,
                "tag_labels": {"tf:lexA": "lexA"},
                "video_subtitle": "lexA=0.80 cpxR=0.71",
            },
        },
        row_index=0,
    )
    assert record.display.video_subtitle == "lexA=0.80 cpxR=0.71"


def test_sequence_evidence_map_adapter_requires_no_source_columns() -> None:
    cfg = AdapterCfg(kind="sequence_evidence_map_v1", columns={}, policies={})

    assert required_source_columns(cfg) == []

    adapter = build_adapter(cfg, alphabet="DNA")
    assert isinstance(adapter, SequenceEvidenceMapV1Adapter)


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        ("duplex_sequence_v1", DuplexSequenceV1Adapter),
        ("hairpin_topology_v1", HairpinTopologyV1Adapter),
        ("sequence_windows_v1", SequenceWindowsV1Adapter),
        ("yiu_linear_state_v1", YiuLinearStateV1Adapter),
        ("yiu_hairpin_topology_v1", YiuHairpinTopologyV1Adapter),
        ("yiu_topology_cartoon_v1", YiuTopologyCartoonV1Adapter),
    ],
)
def test_build_adapter_constructs_contract_driven_adapter_types(kind: str, expected_type: type[object]) -> None:
    cfg = AdapterCfg(kind=kind, columns={}, policies={})

    adapter = build_adapter(cfg, alphabet="DNA")

    assert isinstance(adapter, expected_type)


def test_adapter_descriptor_policy_normalizers_accept_supported_values() -> None:
    densegen_policies = adapter_descriptor("densegen_tfbs").normalize_policies(
        {
            "ambiguous": "first",
            "offset_mode": "one_based",
            "on_missing_kmer": "skip_entry",
            "on_invalid_row": "skip",
            "min_per_record": "2",
            "require_non_null_cols": ["sequence", 7],
            "zero_as_unspecified": False,
            "require_non_empty": True,
        },
        "input.adapter.policies",
    )
    cruncher_policies = adapter_descriptor("cruncher_best_window").normalize_policies(
        {"on_missing_hit": "skip", "on_missing_pwm": "skip_effect"},
        "input.adapter.policies",
    )

    assert densegen_policies == {
        "ambiguous": "first",
        "offset_mode": "one_based",
        "on_missing_kmer": "skip_entry",
        "on_invalid_row": "skip",
        "min_per_record": 2,
        "require_non_null_cols": ["sequence", "7"],
        "zero_as_unspecified": False,
        "require_non_empty": True,
    }
    assert cruncher_policies == {"on_missing_hit": "skip", "on_missing_pwm": "skip_effect"}


def test_densegen_policy_normalizer_rejects_invalid_non_null_cols_type() -> None:
    with pytest.raises(SchemaError, match="require_non_null_cols must be a list"):
        adapter_descriptor("densegen_tfbs").normalize_policies(
            {"require_non_null_cols": "sequence"},
            "input.adapter.policies",
        )


def test_densegen_policy_normalizer_rejects_non_bool_flags() -> None:
    with pytest.raises(SchemaError, match="zero_as_unspecified must be bool"):
        adapter_descriptor("densegen_tfbs").normalize_policies(
            {"zero_as_unspecified": "yes"},
            "input.adapter.policies",
        )


def test_build_adapter_constructs_cruncher_adapter_from_existing_inputs(tmp_path) -> None:
    hits_path = write_parquet(
        tmp_path / "hits.parquet",
        [
            {
                "elite_id": "elite-1",
                "tf": "lexA",
                "best_start": 1,
                "best_strand": "+",
                "best_window_seq": "TTGACA",
                "best_core_seq": "TTGACA",
            }
        ],
    )
    config_path = tmp_path / "cruncher.yaml"
    config_path.write_text(
        "cruncher:\n  pwms_info:\n    lexA:\n      pwm_matrix:\n        - [0.25, 0.25, 0.25, 0.25]\n"
    )
    cfg = AdapterCfg(
        kind="cruncher_best_window",
        columns={
            "sequence": "sequence",
            "id": "id",
            "hits_path": str(hits_path),
            "config_path": str(config_path),
        },
        policies={},
    )

    adapter = build_adapter(cfg, alphabet="DNA")

    assert isinstance(adapter, CruncherBestWindowAdapter)


@pytest.mark.parametrize(
    ("owner_id", "expected"),
    [
        ("payload_left_half", "segment_payload"),
        ("snapback_adapter", "segment_adapter"),
        ("source_forward_primer_binding_region", "segment_primer"),
        ("retained_region", "segment_retained"),
        ("sacrificial_region_long", "segment_sacrificial"),
        ("misc_region", "segment"),
    ],
)
def test_sequence_evidence_map_owner_style_tokens(owner_id: str, expected: str) -> None:
    assert _style_token_for_owner(owner_id) == expected


@pytest.mark.parametrize(
    ("tag_kind", "expected"),
    [
        ("payload_overhang_left", "site_overhang"),
        ("type_iis_recognition_left", "site_recognition"),
        ("primer_bindable_by_source_forward", "site_primer"),
        ("adapter_binding", "site_adapter"),
        ("ligation_junction_member", "site_boundary"),
        ("custom_effect", "site_effect"),
    ],
)
def test_sequence_evidence_map_tag_style_tokens(tag_kind: str, expected: str) -> None:
    assert _style_token_for_tag(tag_kind) == expected


def test_sequence_evidence_map_adapter_applies_contract_without_complement_sequence() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": "state-1",
            "topology_kind": "linear_dsdna",
            "alphabet": "dna",
            "primary_sequence": "ACGTACGTACGT",
            "owners": [
                {
                    "owner_id": "payload_left_half",
                    "row_id": "primary",
                    "start": 0,
                    "end": 4,
                    "display_label": "Payload",
                    "short_label": "PAY",
                },
                {
                    "owner_id": "retained_region",
                    "row_id": "primary",
                    "start": 4,
                    "end": 6,
                    "display_label": "Retained",
                    "short_label": "RET",
                },
                {
                    "owner_id": "source_forward_primer_binding_region",
                    "row_id": "primary",
                    "start": 6,
                    "end": 8,
                    "display_label": "Primer",
                    "short_label": "PRI",
                },
                {
                    "owner_id": "snapback_adapter",
                    "row_id": "primary",
                    "start": 8,
                    "end": 10,
                    "display_label": "Adapter",
                    "short_label": "ADP",
                },
                {
                    "owner_id": "sacrificial_region_long",
                    "row_id": "primary",
                    "start": 10,
                    "end": 12,
                    "display_label": "Sacrificial",
                    "short_label": "SAC",
                },
            ],
            "effect_tags": [
                {
                    "tag_id": "recognition",
                    "tag_kind": "type_iis_recognition_left",
                    "row_id": "primary",
                    "start": 0,
                    "end": 2,
                    "display_label": "Recognition",
                    "short_label": "REC",
                },
                {
                    "tag_id": "primer",
                    "tag_kind": "primer_bindable_by_source_forward",
                    "row_id": "primary",
                    "start": 2,
                    "end": 4,
                    "display_label": "Primer",
                    "short_label": "PRI",
                },
                {
                    "tag_id": "adapter",
                    "tag_kind": "adapter_binding",
                    "row_id": "primary",
                    "start": 4,
                    "end": 6,
                    "display_label": "Adapter",
                    "short_label": "ADP",
                },
                {
                    "tag_id": "junction",
                    "tag_kind": "ligation_junction_member",
                    "row_id": "primary",
                    "start": 6,
                    "end": 8,
                    "display_label": "Junction",
                    "short_label": "JCT",
                },
                {
                    "tag_id": "custom",
                    "tag_kind": "custom_effect",
                    "row_id": "primary",
                    "start": 8,
                    "end": 10,
                    "display_label": "Custom",
                    "short_label": "CUS",
                },
            ],
            "boundaries": [
                {
                    "boundary_id": "cut-1",
                    "row_id": "primary",
                    "boundary": 6,
                    "boundary_kind": "cut",
                    "display_label": "Cut",
                    "short_label": "CUT",
                }
            ],
            "pairings": [
                {
                    "pairing_id": "pair-1",
                    "primary_start": 0,
                    "primary_end": 2,
                    "complement_start": 10,
                    "complement_end": 12,
                    "display_label": "Pair",
                    "short_label": "PAIR",
                }
            ],
            "display": {"title": "Example"},
            "meta": {"mode": "test"},
        },
        row_index=0,
    )

    assert record.meta["show_reverse_complement"] is False
    assert record.features[0].attrs["style_token"] == "segment_payload"
    assert record.features[1].attrs["style_token"] == "segment_retained"
    assert record.features[2].attrs["style_token"] == "segment_primer"
    assert record.features[3].attrs["style_token"] == "segment_adapter"
    assert record.features[4].attrs["style_token"] == "segment_sacrificial"
    assert record.features[5].attrs["style_token"] == "site_recognition"
    assert record.features[6].attrs["style_token"] == "site_primer"
    assert record.features[7].attrs["style_token"] == "site_adapter"
    assert record.features[8].attrs["style_token"] == "site_boundary"
    assert record.features[9].attrs["style_token"] == "site_effect"
    assert [effect.kind for effect in record.effects] == ["boundary_marker", "span_link"]
    assert record.effects[0].params["semantic"] == "cut"


def test_sequence_evidence_map_adapter_can_exclude_tags_from_legend() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": "assembled_payload",
            "topology_kind": "linear_ssdna",
            "alphabet": "iupac_dna",
            "primary_sequence": "CTCTATATCTGATATAGAG",
            "owners": [
                {
                    "owner_id": "payload_left_half",
                    "row_id": "primary",
                    "start": 0,
                    "end": 9,
                    "display_label": "Left payload half",
                    "short_label": "L",
                },
                {
                    "owner_id": "payload_right_half",
                    "row_id": "primary",
                    "start": 9,
                    "end": 19,
                    "display_label": "Right payload half",
                    "short_label": "R",
                },
            ],
            "effect_tags": [
                {
                    "tag_id": "bulge_2",
                    "tag_kind": "payload_bulge_position",
                    "row_id": "primary",
                    "start": 10,
                    "end": 11,
                    "display_label": "Bulge 2",
                    "short_label": "B2",
                },
            ],
            "boundaries": [],
            "pairings": [],
            "display": {"title": "Assembled payload"},
            "meta": {
                "legend_exclude_tags": [
                    "owner:payload_left_half",
                    "owner:payload_right_half",
                ]
            },
        },
        row_index=0,
    )

    assert record.meta["legend_exclude_tags"] == (
        "owner:payload_left_half",
        "owner:payload_right_half",
    )
    assert legend_entries_for_record(record) == [
        ("effect:payload_bulge_position", "Bulge 2"),
    ]


def test_sequence_evidence_map_adapter_preserves_explicit_complement_and_base_highlights() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": "assembled_payload",
            "topology_kind": "linear_dsdna",
            "alphabet": "iupac_dna",
            "primary_sequence": "CTCTATATCTGATATAGAG",
            "complement_sequence": "GAGATATAGAATATATCTC",
            "owners": [],
            "effect_tags": [],
            "boundaries": [
                {
                    "boundary_id": "left-overhang-boundary",
                    "row_id": "primary",
                    "boundary": 9,
                    "boundary_kind": "ligation_junction",
                    "display_label": "",
                    "short_label": "",
                },
            ],
            "pairings": [],
            "display": {"title": "Assembled payload"},
            "meta": {
                "base_highlight_color": "#B91C1C",
                "base_highlights": {
                    "primary": [10],
                    "complement": [10],
                },
                "dim_base_indices": {
                    "primary": [0, 1, 2, 3, 4, 5, 6],
                    "complement": [0, 1, 2, 3, 4, 5, 6],
                },
                "connector_hidden_indices": [9, 11, 12],
                "connector_cross_indices": [10],
                "connector_overhang_spans": [{"start": 9, "end": 13}],
                "segment_labels": [
                    {"text": "Left", "start": 0, "end": 9},
                    {"text": "Right", "start": 9, "end": 19},
                ],
            },
        },
        row_index=0,
    )

    assert record.meta["show_reverse_complement"] is True
    assert record.meta["complement_sequence"] == "GAGATATAGAATATATCTC"
    assert record.meta["base_highlight_color"] == {"primary": "#B91C1C", "complement": "#B91C1C"}
    assert record.meta["base_highlights"] == {"primary": (10,), "complement": (10,)}
    assert record.meta["dim_base_indices"] == {
        "primary": (0, 1, 2, 3, 4, 5, 6),
        "complement": (0, 1, 2, 3, 4, 5, 6),
    }
    assert record.meta["connector_hidden_indices"] == (9, 11, 12)
    assert record.meta["connector_cross_indices"] == (10,)
    assert record.meta["connector_overhang_spans"] == ({"start": 9, "end": 13},)
    assert record.meta["segment_labels"] == (
        {"text": "Left", "start": 0, "end": 9, "row_id": "primary"},
        {"text": "Right", "start": 9, "end": 19, "row_id": "primary"},
    )
    boundary_effects = [effect for effect in record.effects if effect.kind == "boundary_marker"]
    assert len(boundary_effects) == 1
    assert boundary_effects[0].target == {"boundary": 9, "lane": "primary"}


def test_sequence_evidence_map_adapter_normalizes_span_backdrops() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": "assembled_payload",
            "topology_kind": "linear_dsdna",
            "alphabet": "iupac_dna",
            "primary_sequence": "CTCTATATCTGATATAGAG",
            "complement_sequence": "GAGATATAGAATATATCTC",
            "owners": [],
            "effect_tags": [],
            "boundaries": [],
            "pairings": [],
            "display": {"title": "Assembled payload"},
            "meta": {
                "span_backdrops": [
                    {
                        "start": 9,
                        "end": 13,
                        "coordinate_space": "payload_forward",
                        "fill": "#BFDBFE",
                        "alpha": 0.3,
                        "corner_radius": 8.0,
                        "cover_rows": "both",
                    }
                ]
            },
        },
        row_index=0,
    )

    assert record.meta["span_backdrops"] == (
        {
            "start": 9,
            "end": 13,
            "coordinate_space": "payload_forward",
            "fill": "#BFDBFE",
            "alpha": 0.3,
            "corner_radius": 8.0,
            "cover_rows": "both",
        },
    )


def test_sequence_evidence_map_adapter_rejects_legacy_boundary_marker_style_meta() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    with pytest.raises(SchemaError, match="meta.boundary_marker_style is no longer supported"):
        adapter.apply(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "legacy-boundary-style",
                "topology_kind": "linear_dsdna",
                "alphabet": "iupac_dna",
                "primary_sequence": "AACCGGTT",
                "owners": [],
                "effect_tags": [],
                "boundaries": [],
                "pairings": [],
                "display": {"title": "Legacy"},
                "meta": {"boundary_marker_style": "dashed_uncapped"},
            },
            row_index=0,
        )


def test_sequence_evidence_map_adapter_rejects_connector_indices_outside_overhang_spans() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    with pytest.raises(SchemaError, match="connector_cross_indices must lie within connector_overhang_spans"):
        adapter.apply(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "bad-overhang-cross",
                "topology_kind": "linear_dsdna",
                "alphabet": "iupac_dna",
                "primary_sequence": "AACCGGTT",
                "owners": [],
                "effect_tags": [],
                "boundaries": [
                    {
                        "boundary_id": "left",
                        "row_id": "primary",
                        "boundary": 2,
                        "boundary_kind": "cut",
                        "display_label": "",
                        "short_label": "",
                    },
                    {
                        "boundary_id": "right",
                        "row_id": "complement",
                        "boundary": 6,
                        "boundary_kind": "cut",
                        "display_label": "",
                        "short_label": "",
                    },
                ],
                "pairings": [],
                "display": {"title": "Bad"},
                "meta": {
                    "connector_overhang_spans": [{"start": 2, "end": 6}],
                    "connector_cross_indices": [1],
                },
            },
            row_index=0,
        )


def test_sequence_evidence_map_adapter_allows_connector_spans_without_matching_boundary_positions() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": "assembled-payload-single-seam",
            "topology_kind": "linear_dsdna",
            "alphabet": "iupac_dna",
            "primary_sequence": "AACCGGTT",
            "owners": [],
            "effect_tags": [],
            "boundaries": [
                {
                    "boundary_id": "join",
                    "row_id": "primary",
                    "boundary": 4,
                    "boundary_kind": "ligation_junction",
                    "display_label": "",
                    "short_label": "",
                },
            ],
            "pairings": [],
            "display": {"title": "Assembled payload"},
            "meta": {
                "connector_overhang_spans": [{"start": 2, "end": 6}],
                "connector_hidden_indices": [2, 3, 5],
                "connector_cross_indices": [4],
            },
        },
        row_index=0,
    )

    assert record.meta["connector_overhang_spans"] == ({"start": 2, "end": 6},)
    boundary_effects = [effect for effect in record.effects if effect.kind == "boundary_marker"]
    assert len(boundary_effects) == 1
    assert boundary_effects[0].target == {"boundary": 4, "lane": "primary"}


def test_sequence_evidence_map_adapter_rejects_invalid_contract_payload() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    with pytest.raises(SchemaError, match="Invalid sequence_evidence_map_v1 contract at row 7"):
        adapter.apply(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "bad-state",
                "topology_kind": "linear_dsdna",
                "alphabet": "dna",
                "primary_sequence": "",
            },
            row_index=7,
        )


def test_sequence_evidence_map_adapter_rejects_invalid_record_after_contract_validation() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    with pytest.raises(SchemaError, match="Sequence contains invalid characters for DNA"):
        adapter.apply(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "bad-sequence",
                "topology_kind": "linear_dsdna",
                "alphabet": "dna",
                "primary_sequence": "ACGU",
            },
            row_index=3,
        )


def _linear_duplex_payload(*, sequence: str = "TTTACCTCAGCAAAGCTGAGGTAAA") -> dict:
    return {
        "version": 1,
        "kind": "linear_duplex_v1",
        "view_id": "hit_001.linear_duplex",
        "solution_id": "abc123def456",
        "title": "Hit 1 - Linear duplex",
        "coordinate_semantics": "boundary_inclusive_v2",
        "primary_sequence_5to3": sequence,
        "sequence_span": {"start": 0, "end": len(sequence)},
        "cassette_span": {"start": 0, "end": len(sequence)},
        "row_labels": {
            "primary": "5' -> 3' primary",
            "complement": "3' -> 5' complement",
        },
        "target_strand": "complement",
        "segments": [
            {"id": "stem5p_arm", "start": 0, "end": 10, "semantic": "stem5p_arm", "label": "Stem 5' arm"},
            {"id": "loop", "start": 10, "end": 15, "semantic": "loop", "label": "Loop"},
            {"id": "stem3p_arm", "start": 15, "end": 25, "semantic": "stem3p_arm", "label": "Stem 3' arm"},
        ],
        "site_instances": [
            {
                "id": "left_site",
                "variant_id": "Nb.BbvCI",
                "specificity_id": "BbvCI",
                "start": 2,
                "end": 9,
                "orientation": "forward",
                "intent": "intended_left",
                "label": "Nb.BbvCI",
                "site_target_strand": "complement",
            }
        ],
        "nick_events": [
            {
                "id": "left_nick",
                "boundary": 7,
                "target_strand": "complement",
                "source_site_id": "left_site",
                "intent": "intended_left",
                "label": "Nick",
            }
        ],
        "bounded_segment": {
            "start_boundary": 7,
            "end_boundary": 20,
            "target_strand": "complement",
            "label": "Bounded nicked segment",
        },
        "labels": [{"text": "Target strand: complement", "placement": "header"}],
        "meta": {"rank": 1},
    }


def _hairpin_topology_payload(*, sequence: str = "ACCTCAGCAAAGCTGAGGT") -> dict:
    return {
        "version": 1,
        "kind": "ssdna_hairpin_v1",
        "view_id": "hit_001.ssdna_hairpin",
        "solution_id": "abc123def456",
        "title": "Hit 1 - ssDNA hairpin",
        "primary_sequence_5to3": sequence,
        "topology": {
            "stem5p_span": {"start": 0, "end": 7},
            "loop_span": {"start": 7, "end": 12},
            "stem3p_span": {"start": 12, "end": 19},
        },
        "pair_map": [
            {"left_index": 0, "right_index": 18},
            {"left_index": 1, "right_index": 17},
        ],
        "feature_spans": [
            {
                "id": "left_site_projection",
                "start": 1,
                "end": 7,
                "semantic": "motif_projection",
                "label": "Nb.BbvCI motif",
            }
        ],
        "duplex_derived_annotations": [
            {
                "kind": "informational_note",
                "text": "Nicking is defined in the linear duplex interpretation.",
            }
        ],
        "meta": {"rank": 1},
    }


def _yiu_linear_state_payload(*, sequence: str = "CCTCAGCCCGCTGATCCCTATCAGTGATAGA") -> dict:
    return {
        "contract_kind": "yiu_linear_state_v1",
        "state_id": "hairpin_pcr_linear_insert",
        "topology_kind": "linear_dsdna",
        "alphabet": "iupac_dna",
        "primary_sequence": sequence,
        "complement_sequence": "TCTATCACTGATAGGGATCAGCGGGCTGAGG",
        "segments": [
            {"segment_id": "left_arm", "state_start": 0, "state_end": 5},
            {"segment_id": "skip", "state_start": 5, "state_end": 5},
        ],
        "annotations": [],
        "cuts": [{"site_id": "cut-1", "top_boundary": 4, "bottom_boundary": 8}],
        "junctions": [{"id": "junction-1", "join_index": 12}],
        "fragments": [],
        "display": {"title": "Split-payload insert"},
        "meta": {"evidence_mode": "pattern_compatibility"},
    }


def _yiu_hairpin_topology_payload(*, sequence: str = "CCTCAGCCCGCTGATCAGCGGGCTGAGG") -> dict:
    return {
        "contract_kind": "yiu_hairpin_topology_v1",
        "state_id": "ligated_ssdna_hairpin",
        "topology_kind": "ssdna_hairpin",
        "sequence": sequence,
        "stem_left_span": {"start": 0, "end": 8},
        "stem_right_span": {"start": 20, "end": 28},
        "loop_span": {"start": 8, "end": 20},
        "pair_map": [{"left_index": 0, "right_index": 27}],
        "adapter_branches": [],
        "annotations": [{"note": "structured"}],
        "display": {"title": "Ligation hairpin"},
        "meta": {"evidence_mode": "concrete_realization"},
    }


def _yiu_topology_cartoon_payload(
    *,
    sequence: str = "CCGATGTCCCTATCAGTGATAGAGAGGGGGGGGGGGGCCTCAGCCCGCTGA",
) -> dict:
    return {
        "contract_kind": "yiu_topology_cartoon_v1",
        "state_id": "circularized_payload_candidate",
        "topology_kind": "circular_duplex",
        "sequence": sequence,
        "segments": [
            {"segment_id": "payload", "state_start": 0, "state_end": 10},
            {"segment_id": "skip", "state_start": 10, "state_end": 10},
        ],
        "annotations": [],
        "cuts": [],
        "junctions": [{"id": "junction", "join_index": 15}],
        "fragments": [],
        "display": {"title": "Circularized payload"},
        "meta": {"evidence_mode": "concrete_realization"},
    }


def test_duplex_sequence_adapter_applies_contract_payload() -> None:
    adapter = DuplexSequenceV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(_linear_duplex_payload(), row_index=0)

    assert record.id == "hit_001.linear_duplex"
    assert record.meta["adapter"] == "duplex_sequence_v1"
    assert record.meta["target_strand"] == "complement"
    assert record.display.tag_labels["bounded_segment"] == "Bounded nicked segment"


def test_hairpin_topology_adapter_applies_contract_payload() -> None:
    adapter = HairpinTopologyV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(_hairpin_topology_payload(), row_index=1)

    assert record.id == "hit_001.ssdna_hairpin"
    assert record.meta["adapter"] == "hairpin_topology_v1"
    assert record.meta["solution_id"] == "abc123def456"
    assert record.display.tag_labels["feature_projection"] == "Motif projection"


def test_hairpin_topology_adapter_wraps_invalid_contract_payload() -> None:
    adapter = HairpinTopologyV1Adapter(columns={}, policies={}, alphabet="DNA")

    with pytest.raises(SchemaError, match="Invalid ssdna_hairpin_v1 contract at row 2"):
        adapter.apply({"kind": "ssdna_hairpin_v1"}, row_index=2)


def test_yiu_linear_state_adapter_applies_contract_payload() -> None:
    adapter = YiuLinearStateV1Adapter(columns={}, policies={}, alphabet="IUPAC_DNA")

    record = adapter.apply(_yiu_linear_state_payload(), row_index=2)

    assert record.id == "hairpin_pcr_linear_insert"
    assert [feature.id for feature in record.features] == ["left_arm"]
    assert [effect.kind for effect in record.effects] == [
        "boundary_marker",
        "boundary_marker",
        "boundary_marker",
    ]
    assert record.meta["adapter"] == "yiu_linear_state_v1"


def test_yiu_hairpin_topology_adapter_applies_contract_payload() -> None:
    adapter = YiuHairpinTopologyV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(_yiu_hairpin_topology_payload(), row_index=3)

    assert record.id == "ligated_ssdna_hairpin"
    assert [feature.id for feature in record.features] == ["stem5p_span", "loop_span", "stem3p_span"]
    assert record.meta["adapter"] == "yiu_hairpin_topology_v1"
    assert record.meta["hairpin_notes"] == [{"note": "structured"}]


def test_yiu_topology_cartoon_adapter_applies_contract_payload() -> None:
    adapter = YiuTopologyCartoonV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(_yiu_topology_cartoon_payload(), row_index=4)

    assert record.id == "circularized_payload_candidate"
    assert [feature.id for feature in record.features] == ["payload"]
    assert record.meta["adapter"] == "yiu_topology_cartoon_v1"


@pytest.mark.parametrize(
    ("adapter", "payload"),
    [
        (
            DuplexSequenceV1Adapter(columns={}, policies={}, alphabet="DNA"),
            _linear_duplex_payload(sequence="TTRACCTCAGCAAAGCTGAGGTAAA"),
        ),
        (
            HairpinTopologyV1Adapter(columns={}, policies={}, alphabet="DNA"),
            _hairpin_topology_payload(sequence="ACRTCAGCAAAGCTGAGGT"),
        ),
        (
            YiuLinearStateV1Adapter(columns={}, policies={}, alphabet="DNA"),
            _yiu_linear_state_payload(sequence="CCTRAGCCCGCTGATCCCTATCAGTGATAGA"),
        ),
        (
            YiuHairpinTopologyV1Adapter(columns={}, policies={}, alphabet="DNA"),
            _yiu_hairpin_topology_payload(sequence="CCTRAGCCCGCTGATCAGCGGGCTGAGG"),
        ),
        (
            YiuTopologyCartoonV1Adapter(columns={}, policies={}, alphabet="DNA"),
            _yiu_topology_cartoon_payload(sequence="CCRATGTCCCTATCAGTGATAGAGAGGGGGGGGGGGGCCTCAGCCCGCTGA"),
        ),
    ],
)
def test_contract_driven_adapters_wrap_record_validation_errors(adapter: object, payload: dict) -> None:
    with pytest.raises(SchemaError, match="Sequence contains invalid characters for DNA"):
        adapter.apply(payload, row_index=5)


@pytest.mark.parametrize(
    "adapter",
    [
        YiuLinearStateV1Adapter(columns={}, policies={}, alphabet="DNA"),
        YiuHairpinTopologyV1Adapter(columns={}, policies={}, alphabet="DNA"),
        YiuTopologyCartoonV1Adapter(columns={}, policies={}, alphabet="DNA"),
    ],
)
def test_yiu_adapters_require_mapping_rows(adapter: object) -> None:
    with pytest.raises(SchemaError, match="row 6 must be a mapping"):
        adapter.apply("bad-row", row_index=6)


def test_yiu_topology_cartoon_adapter_wraps_invalid_contract_payload() -> None:
    adapter = YiuTopologyCartoonV1Adapter(columns={}, policies={}, alphabet="DNA")

    with pytest.raises(SchemaError, match="Invalid yiu_topology_cartoon_v1 contract at row 7"):
        adapter.apply({"contract_kind": "yiu_topology_cartoon_v1"}, row_index=7)


def test_yiu_hairpin_span_helper_rejects_invalid_bounds() -> None:
    assert _yiu_hairpin_span([2, 5], ctx="loop_span") == (2, 5)

    with pytest.raises(SchemaError, match="loop_span must be a 2-item list"):
        _yiu_hairpin_span([2], ctx="loop_span")

    with pytest.raises(SchemaError, match="loop_span end must be > start"):
        _yiu_hairpin_span([5, 5], ctx="loop_span")
