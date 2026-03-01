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
from dnadesign.baserender.src.config import AdapterCfg
from dnadesign.baserender.src.core import SchemaError


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
