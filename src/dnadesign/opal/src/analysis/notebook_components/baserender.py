from __future__ import annotations

from importlib import import_module
from io import BytesIO
from typing import Any, Mapping, Sequence

BASERENDER_CONTRACT_SCHEMA_VERSION = "opal.notebook_baserender_contract.v1"


def build_notebook_baserender_contract(
    schema_columns: Sequence[str],
    *,
    records_path: str | None = None,
) -> dict[str, Any]:
    """Detect whether records expose a public BaseRender-compatible surface."""

    columns = {str(column) for column in schema_columns}
    for candidate in _contract_candidates(columns):
        missing = [column for column in candidate["required_columns"] if column not in columns]
        if missing:
            continue
        return {
            "schema_version": BASERENDER_CONTRACT_SCHEMA_VERSION,
            "available": True,
            "label": candidate["label"],
            "adapter_kind": candidate["adapter_kind"],
            "adapter_columns": candidate["adapter_columns"],
            "adapter_policies": candidate["adapter_policies"],
            "required_columns": candidate["required_columns"],
            "renderer_name": "sequence_rows",
            "style_preset": "presentation_default",
            "style_overrides": {
                "dpi": 180,
                "legend": True,
                "legend_mode": "bottom",
                "uniform_display_font_size": True,
            },
            "records_path": str(records_path or ""),
            "caption": candidate["caption"],
            "alt_text_template": candidate["alt_text_template"],
            "reason": "records schema satisfies a BaseRender adapter contract",
        }
    return {
        "schema_version": BASERENDER_CONTRACT_SCHEMA_VERSION,
        "available": False,
        "label": "Record render",
        "adapter_kind": None,
        "adapter_columns": {},
        "adapter_policies": {},
        "required_columns": [],
        "renderer_name": "sequence_rows",
        "style_preset": "presentation_default",
        "style_overrides": {},
        "records_path": str(records_path or ""),
        "caption": "",
        "alt_text_template": "",
        "reason": "records schema does not expose a known BaseRender adapter contract",
    }


def build_notebook_baserender_contract_rows(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return compact rows for progressively disclosed BaseRender evidence."""

    adapter_columns = contract.get("adapter_columns")
    adapter_value = _compact_mapping(adapter_columns if isinstance(adapter_columns, Mapping) else {})
    return [
        {"field": "available", "value": str(bool(contract.get("available"))).lower()},
        {"field": "adapter", "value": str(contract.get("adapter_kind") or "not available")},
        {"field": "renderer", "value": str(contract.get("renderer_name") or "sequence_rows")},
        {"field": "required columns", "value": ", ".join(str(item) for item in contract.get("required_columns") or ())},
        {"field": "adapter columns", "value": adapter_value},
        {"field": "reason", "value": str(contract.get("reason") or "not recorded")},
    ]


def render_notebook_baserender_record(record_row: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    """Render a single record through the public BaseRender API."""

    if not bool(contract.get("available")):
        raise ValueError(str(contract.get("reason") or "BaseRender contract is unavailable."))
    record_id = str(record_row.get("id") or "unknown")
    adapter_kind = str(contract.get("adapter_kind") or "")
    if not adapter_kind:
        raise ValueError("BaseRender contract is missing adapter_kind.")

    baserender = import_module("dnadesign.baserender")
    adapt_records = baserender.adapt_records
    render_record_figure = baserender.render_record_figure

    records = adapt_records(
        [dict(record_row)],
        adapter_kind=adapter_kind,
        adapter_columns=dict(contract.get("adapter_columns") or {}),
        adapter_policies=dict(contract.get("adapter_policies") or {}),
        alphabet="DNA",
    )
    if not records:
        raise ValueError(f"BaseRender produced no record for `{record_id}`.")
    figure = render_record_figure(
        records[0],
        renderer_name=str(contract.get("renderer_name") or "sequence_rows"),
        style_preset=contract.get("style_preset") or "presentation_default",
        style_overrides=dict(contract.get("style_overrides") or {}),
    )
    try:
        buffer = BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
        image_bytes = buffer.getvalue()
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close(figure)
        except Exception:
            pass
    if not image_bytes:
        raise ValueError(f"BaseRender image bytes were empty for `{record_id}`.")
    caption = str(contract.get("caption") or "BaseRender record view.")
    return {
        "record_id": record_id,
        "image_bytes": image_bytes,
        "caption": f"{caption} Record `{record_id}`.",
        "alt_text": str(contract.get("alt_text_template") or caption).format(record_id=record_id),
    }


def _contract_candidates(columns: set[str]) -> list[dict[str, Any]]:
    candidates = [
        {
            "label": "Record render",
            "adapter_kind": "densegen_tfbs",
            "adapter_columns": {
                "id": "id",
                "sequence": "sequence",
                "annotations": "densegen__used_tfbs_detail",
            },
            "adapter_policies": {"on_invalid_row": "error"},
            "required_columns": ("id", "sequence", "densegen__used_tfbs_detail"),
            "caption": "BaseRender view of the selected record with annotated sequence features.",
            "alt_text_template": (
                "BaseRender sequence diagram for record {record_id}; annotated sequence features "
                "are drawn over the selected OPAL record."
            ),
        }
    ]
    for feature_column in ("opal__baserender_features", "baserender__features"):
        if feature_column in columns:
            candidates.append(
                {
                    "label": "Record render",
                    "adapter_kind": "generic_features",
                    "adapter_columns": {
                        "id": "id",
                        "sequence": "sequence",
                        "features": feature_column,
                    },
                    "adapter_policies": {"on_invalid_row": "error"},
                    "required_columns": ("id", "sequence", feature_column),
                    "caption": "BaseRender view of the selected record with generic feature annotations.",
                    "alt_text_template": (
                        "BaseRender sequence diagram for record {record_id}; generic feature annotations "
                        "are drawn over the selected OPAL record."
                    ),
                }
            )
    return candidates


def _compact_mapping(mapping: Mapping[str, Any]) -> str:
    return "; ".join(f"{key}={value}" for key, value in mapping.items()) or "not recorded"
