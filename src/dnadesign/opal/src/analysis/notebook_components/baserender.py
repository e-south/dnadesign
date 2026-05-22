from __future__ import annotations

from typing import Any, Mapping, Sequence

BASERENDER_CONTRACT_SCHEMA_VERSION = "opal.notebook_baserender_contract.v1"
NO_RENDERABLE_RECORDS_LABEL = "(no renderable records)"


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


def _contract_candidates(columns: set[str]) -> list[dict[str, Any]]:
    candidates = []
    if "seq_annot__features" in columns:
        candidates.append(
            {
                "label": "Record render",
                "adapter_kind": "usr_genbank_annotations_v1",
                "adapter_columns": {
                    "id": "id",
                    "sequence": "sequence",
                    "annotations": "seq_annot__features",
                    **({"overlay_text": "usr_label__primary"} if "usr_label__primary" in columns else {}),
                    **({"source_file": "seq_annot__source_file"} if "seq_annot__source_file" in columns else {}),
                    **({"product_kind": "derived__product_kind"} if "derived__product_kind" in columns else {}),
                },
                "adapter_policies": {"on_invalid_row": "error"},
                "required_columns": ("id", "sequence", "seq_annot__features"),
                "caption": "BaseRender view of the selected record with sequence annotations.",
                "alt_text_template": (
                    "BaseRender sequence diagram for record {record_id}; sequence annotations "
                    "are drawn over the selected OPAL record."
                ),
            }
        )
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
    candidates.append(
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
    )
    return candidates


def _compact_mapping(mapping: Mapping[str, Any]) -> str:
    return "; ".join(f"{key}={value}" for key, value in mapping.items()) or "not recorded"
