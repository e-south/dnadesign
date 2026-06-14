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
    compact_target = _first_metadata_value(metadata, "target_dropdown_label")
    target_label = _first_metadata_value(
        metadata,
        "target_dropdown_label",
        "target_label",
        "target_display_label",
        "target_title",
        "target",
        "probe_target",
    )
    family_label = (
        None if compact_target else _first_metadata_value(metadata, "label_family_id", "probe_label_family_id")
    )
    compact_parts = [
        _compact_metadata_value(target_label),
        _compact_metadata_value(_probe_scope_label(metadata)),
        _compact_metadata_value(_first_metadata_value(metadata, "label_oracle_kind", "probe_oracle_kind")),
        _compact_metadata_value(_first_metadata_value(metadata, "label_split_id", "probe_split_id")),
        _compact_metadata_value(family_label),
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


def _probe_scope_label(metadata: Mapping[str, Any]) -> str:
    """Return a compact probe-scope discriminator when target/role/seed labels collide."""

    if _first_metadata_value(metadata, "candidate_scope_policy_id") == "tfbs_slot_position_target_count_eq_1_v1":
        return "count_fixed"
    if (
        _first_metadata_value(metadata, "null_version")
        == "densegen_tfbs_learnability_slot_geometry_count_matched_null_v1"
    ):
        return "count_preserving"
    return ""


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
        "densegen_plan_logic4": "logic4",
        "leave_sigma35_variant": "sigma35",
        "matched_null": "matched-null",
        "null": "matched-null",
        "positive": "positive",
        "random_id": "random",
        "count_fixed": "count-fixed",
        "count_preserving": "count-preserving",
        "tf_family_count_fraction": "TF family count_fraction",
        "tf_family_presence": "TF family presence",
        "tf_slot_family_presence": "TF slot_family_presence",
    }
    return aliases.get(text, text)
