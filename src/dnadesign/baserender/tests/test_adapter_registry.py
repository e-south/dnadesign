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
from dnadesign.baserender.src.adapters.yiu_hairpin_topology_v1 import YiuHairpinTopologyV1Adapter
from dnadesign.baserender.src.adapters.yiu_linear_state_v1 import YiuLinearStateV1Adapter
from dnadesign.baserender.src.adapters.yiu_topology_cartoon_v1 import YiuTopologyCartoonV1Adapter
from dnadesign.baserender.src.config import AdapterCfg
from dnadesign.baserender.src.config.adapter_contracts import adapter_descriptor
from dnadesign.baserender.src.core import SchemaError

from .conftest import write_parquet


def test_required_source_columns_densegen_includes_optional_present_columns() -> None:
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
