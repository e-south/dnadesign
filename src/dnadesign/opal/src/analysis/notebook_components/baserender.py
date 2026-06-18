"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender.py

Notebook component builders for BaseRender OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping, Sequence

BASERENDER_CONTRACT_SCHEMA_VERSION = "dnadesign.baserender.record_render_contract.v1"
NO_RENDERABLE_RECORDS_LABEL = "(no renderable records)"
GENERIC_BASERENDER_FEATURES_COLUMN = "opal__baserender_features"


def build_notebook_baserender_contract(
    schema_columns: Sequence[str],
    *,
    records_path: str | None = None,
    metadata_records_path: str | None = None,
    metadata_schema_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Detect whether records expose a public BaseRender-compatible surface."""

    baserender = import_module("dnadesign.baserender")
    columns = {str(column) for column in schema_columns}
    metadata_columns = {str(column) for column in metadata_schema_columns or ()}
    if {"id", "sequence", GENERIC_BASERENDER_FEATURES_COLUMN}.issubset(columns):
        return _available_contract(
            adapter_kind="generic_features",
            adapter_columns={
                "id": "id",
                "sequence": "sequence",
                "features": GENERIC_BASERENDER_FEATURES_COLUMN,
            },
            adapter_policies={},
            required_columns=["id", "sequence", GENERIC_BASERENDER_FEATURES_COLUMN],
            render_route="figure",
            renderer_name="sequence_rows",
            records_path=records_path,
            caption="BaseRender generic feature view.",
        )

    densegen_config = baserender.sequence_panel_config_for_adapter("densegen_tfbs")
    densegen_columns = dict(densegen_config.adapter_columns)
    detection_required = [
        str(densegen_columns[key]) for key in ("id", "sequence", "annotations") if key in densegen_columns
    ]
    metadata_required = [str(densegen_columns[key]) for key in ("id", "annotations") if key in densegen_columns]
    row_required = [str(densegen_columns[key]) for key in ("id", "sequence") if key in densegen_columns]
    has_embedded_annotations = set(detection_required).issubset(columns)
    has_metadata_annotations = set(row_required).issubset(columns) and set(metadata_required).issubset(metadata_columns)
    if has_embedded_annotations or has_metadata_annotations:
        return _available_contract(
            adapter_kind=str(densegen_config.adapter_kind),
            adapter_columns=densegen_columns,
            adapter_policies=dict(densegen_config.adapter_policies),
            required_columns=row_required,
            render_route="sequence_panel",
            renderer_name=str(densegen_config.renderer_name),
            records_path=records_path,
            metadata_records_path=metadata_records_path if has_metadata_annotations else None,
            metadata_required_columns=metadata_required if has_metadata_annotations else [],
            caption="BaseRender TFBS metadata view.",
            style_overrides=dict(densegen_config.style_overrides or {}),
            target_width_px=int(densegen_config.target_width_px),
            target_height_px=int(densegen_config.target_height_px),
            vertical_anchor=str(densegen_config.vertical_anchor),
            canvas_top_pad_px=int(densegen_config.canvas_top_pad_px),
        )

    return {
        "schema_version": BASERENDER_CONTRACT_SCHEMA_VERSION,
        "available": False,
        "render_route": None,
        "adapter_kind": None,
        "renderer_name": "sequence_rows",
        "required_columns": [],
        "adapter_columns": {},
        "adapter_policies": {},
        "records_path": records_path,
        "metadata_records_path": metadata_records_path,
        "metadata_required_columns": [],
        "reason": "No public BaseRender-compatible record surface was detected.",
    }


def _available_contract(
    *,
    adapter_kind: str,
    adapter_columns: Mapping[str, Any],
    adapter_policies: Mapping[str, Any],
    required_columns: Sequence[str],
    render_route: str,
    renderer_name: str,
    records_path: str | None,
    metadata_records_path: str | None = None,
    metadata_required_columns: Sequence[str] | None = None,
    caption: str,
    style_overrides: Mapping[str, Any] | None = None,
    target_width_px: int | None = None,
    target_height_px: int | None = None,
    vertical_anchor: str | None = None,
    canvas_top_pad_px: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": BASERENDER_CONTRACT_SCHEMA_VERSION,
        "available": True,
        "render_route": render_route,
        "adapter_kind": adapter_kind,
        "renderer_name": renderer_name,
        "required_columns": list(required_columns),
        "adapter_columns": dict(adapter_columns),
        "adapter_policies": dict(adapter_policies),
        "records_path": records_path,
        "metadata_records_path": metadata_records_path,
        "metadata_required_columns": list(metadata_required_columns or ()),
        "reason": "detected",
        "caption": caption,
        "alt_text_template": "BaseRender sequence diagram for record {record_id}; {feature_count} annotations.",
        "style_overrides": dict(style_overrides or {}),
        "target_width_px": target_width_px,
        "target_height_px": target_height_px,
        "vertical_anchor": vertical_anchor,
        "canvas_top_pad_px": canvas_top_pad_px,
    }


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
