from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping, Sequence

BASERENDER_CONTRACT_SCHEMA_VERSION = "dnadesign.baserender.record_render_contract.v1"
NO_RENDERABLE_RECORDS_LABEL = "(no renderable records)"


def build_notebook_baserender_contract(
    schema_columns: Sequence[str],
    *,
    records_path: str | None = None,
) -> dict[str, Any]:
    """Detect whether records expose a public BaseRender-compatible surface."""

    baserender = import_module("dnadesign.baserender")
    return dict(
        baserender.record_render_contract_for_schema(
            tuple(str(column) for column in schema_columns),
            records_path=records_path,
        )
    )


def build_notebook_baserender_contract_rows(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return compact rows for progressively disclosed BaseRender evidence."""

    adapter_columns = contract.get("adapter_columns")
    adapter_value = _compact_mapping(adapter_columns if isinstance(adapter_columns, Mapping) else {})
    return [
        {"field": "contract", "value": str(contract.get("schema_version") or BASERENDER_CONTRACT_SCHEMA_VERSION)},
        {"field": "available", "value": str(bool(contract.get("available"))).lower()},
        {"field": "route", "value": str(contract.get("render_route") or "not available")},
        {"field": "adapter", "value": str(contract.get("adapter_kind") or "not available")},
        {"field": "renderer", "value": str(contract.get("renderer_name") or "sequence_rows")},
        {"field": "required columns", "value": ", ".join(str(item) for item in contract.get("required_columns") or ())},
        {"field": "adapter columns", "value": adapter_value},
        {"field": "reason", "value": str(contract.get("reason") or "not recorded")},
    ]


def _compact_mapping(mapping: Mapping[str, Any]) -> str:
    return "; ".join(f"{key}={value}" for key, value in mapping.items()) or "not recorded"
