"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/campaign_set_visuals.py

Notebook component builders for campaign set visuals OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ._support import display_name, mapping, sequence


def build_notebook_collection_set_choices(collection_visuals: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return selectable campaign-set choices from materialized collection visuals."""

    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    matches: dict[str, dict[str, Any]] = {}
    tier_labels: dict[str, str] = {}
    tier_ranks: dict[str, int] = {}
    order: list[str] = []
    for raw in sequence(collection_visuals):
        if not isinstance(raw, Mapping):
            continue
        key = str(raw.get("comparison_set_key") or "").strip()
        if not key:
            continue
        if key not in counts:
            order.append(key)
        counts[key] = counts.get(key, 0) + 1
        labels.setdefault(key, str(raw.get("comparison_set_label") or key))
        if key not in matches:
            matches[key] = dict(mapping(raw.get("comparison_set_match")))
        if key not in tier_labels:
            tier_label = str(raw.get("evidence_tier_label") or "").strip()
            if tier_label:
                tier_labels[key] = tier_label
        if key not in tier_ranks:
            try:
                tier_ranks[key] = int(raw.get("evidence_tier_rank"))
            except (TypeError, ValueError):
                pass
    labels_seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for key in sorted(order, key=lambda item: (tier_ranks.get(item, 10_000), order.index(item))):
        row = {
            "key": key,
            "label": _unique_label(_set_choice_label(labels[key], tier_labels.get(key)), labels_seen),
            "visual_count": counts[key],
            "match": matches[key],
        }
        if tier_labels.get(key):
            row["evidence_tier_label"] = tier_labels[key]
        if key in tier_ranks:
            row["evidence_tier_rank"] = tier_ranks[key]
        rows.append(row)
    return rows


def build_notebook_collection_visual_choices(
    collection_visuals: Iterable[Mapping[str, Any]],
    *,
    comparison_set_key: str | None = None,
) -> list[dict[str, Any]]:
    """Return selectable, manifest-backed collection-level visual choices."""

    choices: list[dict[str, Any]] = []
    labels_seen: set[str] = set()
    for raw in sequence(collection_visuals):
        if not isinstance(raw, Mapping):
            continue
        if comparison_set_key not in (None, "") and str(raw.get("comparison_set_key") or "") != comparison_set_key:
            continue
        choice = dict(raw)
        surface_kind = str(choice.get("surface_kind") or "").strip()
        if not surface_kind:
            raise ValueError("Collection visual choice is missing required surface_kind.")
        choice["surface_kind"] = surface_kind
        choice.setdefault("kind", surface_kind)
        choice["label"] = _unique_label(_choice_label(choice), labels_seen)
        choices.append(choice)
    return choices


def build_notebook_collection_visual_card_rows(choice: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build compact evidence rows for a materialized collection visual."""

    interval = mapping(choice.get("interval"))
    rows = [
        {"field": "visual", "value": choice.get("label") or choice.get("id") or "not recorded"},
        {"field": "campaign set", "value": choice.get("comparison_set_label") or "not recorded"},
        {
            "field": "evidence tier",
            "value": choice.get("evidence_tier_label") or choice.get("evidence_tier") or "not recorded",
        },
        {"field": "surface", "value": choice.get("surface_kind") or "not recorded"},
        {"field": "source plot", "value": choice.get("source_plot_name") or "not recorded"},
        {"field": "relationship", "value": choice.get("relationship_id") or "not recorded"},
        {"field": "grouping", "value": _group_key_text(choice.get("group_key"))},
        {"field": "metric", "value": choice.get("metric") or "not recorded"},
        {"field": "metric label", "value": choice.get("metric_label") or "not recorded"},
        {"field": "metric expression", "value": choice.get("metric_expression") or "not recorded"},
        {"field": "premise", "value": choice.get("premise") or "not recorded"},
        {"field": "math note", "value": choice.get("math_note") or "not recorded"},
        {"field": "design note", "value": choice.get("design_note") or "not recorded"},
        {"field": "claim boundary", "value": choice.get("claim_boundary") or "not recorded"},
        {"field": "axis scale", "value": _axis_scale_text(mapping(choice.get("axis_scale")))},
        {"field": "cohort", "value": choice.get("cohort") or "not recorded"},
        {"field": "summary", "value": choice.get("summary") or "not recorded"},
        {"field": "interval", "value": choice.get("interval_kind") or "not recorded"},
        {"field": "rows", "value": choice.get("row_count") if choice.get("row_count") is not None else "not recorded"},
        {"field": "artifact freshness", "value": mapping(choice.get("freshness")).get("status") or "not recorded"},
        {"field": "provenance file", "value": choice.get("manifest_path") or "not generated"},
        {"field": "tidy csv", "value": choice.get("tidy_csv") or "not generated"},
    ]
    if interval:
        rows.append({"field": "interval unit", "value": interval.get("unit") or "not recorded"})
        rows.append({"field": "is confidence interval", "value": bool(interval.get("is_confidence_interval"))})
    return rows


def build_notebook_collection_visual_description(choice: Mapping[str, Any]) -> str:
    """Build a short reader-facing description for a collection visual accordion."""

    title = str(choice.get("title") or choice.get("label") or "Selected visual").strip()
    caption = str(choice.get("caption") or "").strip()
    metric_label = str(choice.get("metric_label") or "").strip()
    metric_expression = str(choice.get("metric_expression") or "").strip()
    summary = str(choice.get("summary") or "").strip()
    interval_kind = str(choice.get("interval_kind") or "").strip()
    interval = mapping(choice.get("interval"))
    premise = str(choice.get("premise") or "").strip()
    math_note = str(choice.get("math_note") or "").strip()
    design_note = str(choice.get("design_note") or "").strip()
    claim_boundary = str(choice.get("claim_boundary") or "").strip()
    interpretation = str(choice.get("interpretation_note") or "").strip()

    lines = [f"### {title}", ""]
    if premise:
        lines.extend([f"Premise: {premise}", ""])
    if caption:
        lines.extend([caption, ""])
    if metric_label:
        lines.append(f"- Metric: {metric_label}")
    if metric_expression:
        lines.append(f"- Calculation: `{metric_expression}`")
    if math_note:
        lines.append(f"- Math: {math_note}")
    if design_note:
        lines.append(f"- Design: {design_note}")
    if summary:
        lines.append(f"- Summary: {_description_label(summary)}")
    if interval:
        interval_text = _interval_description(interval)
        if interval_text:
            lines.append(f"- Spread: {interval_text}")
    elif interval_kind and interval_kind != "none":
        lines.append(f"- Spread: {_description_label(interval_kind)}")
    else:
        lines.append("- Spread: none for this materialized single-pair review.")
    if claim_boundary:
        lines.extend(["", f"Claim boundary: {claim_boundary}"])
    if interpretation:
        lines.extend(["", interpretation])
    return "\n".join(lines)


def _axis_scale_text(axis_scale: Mapping[str, Any]) -> str:
    if not axis_scale:
        return "not recorded"
    parts = []
    if axis_scale.get("scale_class"):
        parts.append(f"class={axis_scale['scale_class']}")
    limits = axis_scale.get("limits")
    if isinstance(limits, list) and len(limits) == 2:
        parts.append(f"limits={limits}")
    references = sequence(axis_scale.get("reference_lines"))
    if references:
        parts.append(f"reference_lines={len(references)}")
    return "; ".join(parts) if parts else "not recorded"


def _group_key_text(value: Any) -> str:
    key = str(value or "").strip()
    if not key:
        return "not recorded"
    labels = {
        "label_oracle_kind": "label source",
        "probe_oracle_kind": "label source",
        "peer_review_claim_status": "claim status",
        "learning_loop_baseline": "learning-loop comparison",
        "slot_diagnostic_status": "slot diagnostic status",
    }
    return labels.get(key, _description_label(key))


def _interval_description(interval: Mapping[str, Any]) -> str:
    unit = str(interval.get("unit") or "").strip()
    kind = str(interval.get("kind") or "").strip()
    if not kind:
        return ""
    label = _description_label(kind)
    if unit:
        label = f"{label} across {unit}"
    if interval.get("is_confidence_interval") is False:
        label = f"{label}; not a statistical confidence interval"
    return label


def _description_label(value: Any) -> str:
    text = str(value or "").replace("_", " ").strip()
    if text.lower() == "iqr":
        return "IQR"
    return text


def _choice_label(choice: Mapping[str, Any]) -> str:
    return str(choice.get("label") or choice.get("title") or display_name(choice.get("name") or "visual"))


def _unique_label(label: str, seen: set[str]) -> str:
    base = str(label or "Visual").strip() or "Visual"
    candidate = base
    index = 2
    while candidate in seen:
        candidate = f"{base} ({index})"
        index += 1
    seen.add(candidate)
    return candidate


def _set_choice_label(label: str, tier_label: str | None) -> str:
    del tier_label
    base = str(label or "Campaign set").strip() or "Campaign set"
    return base
