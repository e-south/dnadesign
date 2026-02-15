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

from dnadesign.baserender.src.adapters import required_source_columns
from dnadesign.baserender.src.config import AdapterCfg
from dnadesign.baserender.src.core import SchemaError


def test_required_source_columns_densegen_includes_optional_present_columns() -> None:
    cfg = AdapterCfg(
        kind="densegen_tfbs",
        columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
            "details": "details",
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
