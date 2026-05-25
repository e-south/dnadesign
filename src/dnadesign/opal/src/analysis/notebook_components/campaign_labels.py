from __future__ import annotations

from typing import Any, Mapping

from ._support import mapping


def campaign_dropdown_label(
    campaign: Mapping[str, Any],
    *,
    status: Mapping[str, Any],
    title: str,
    label_context: str,
) -> str:
    metadata = mapping(campaign.get("metadata"))
    compact_parts = [
        _compact_metadata_value(_first_metadata_value(metadata, "target", "probe_target")),
        _compact_metadata_value(_first_metadata_value(metadata, "label_oracle_kind", "probe_oracle_kind")),
        _compact_metadata_value(_first_metadata_value(metadata, "label_split_id", "probe_split_id")),
        _compact_metadata_value(_first_metadata_value(metadata, "label_family_id", "probe_label_family_id")),
        _compact_seed_value(_first_metadata_value(metadata, "seed", "probe_seed")),
    ]
    compact = [part for part in compact_parts if part]
    status_text = str(status.get("progress_status") or "unknown").strip()
    if len(compact) >= 3:
        return " | ".join([*compact, status_text])
    return " | ".join(part for part in (title, label_context, status_text) if part)


def _first_metadata_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _compact_seed_value(value: Any) -> str:
    text = str(value or "").strip()
    return f"s{text}" if text else ""


def _compact_metadata_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    aliases = {
        "cipro": "Cipro",
        "ciprofloxacin": "Cipro",
        "dual": "Dual",
        "ethanol": "Ethanol",
        "leave_sigma35_variant": "sigma35",
        "random_id": "random",
    }
    return aliases.get(text, text)
