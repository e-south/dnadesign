from __future__ import annotations

from typing import Any, Iterable, Mapping

from ._support import display_name, mapping, sequence

CAMPAIGN_SET_COMPARISON_SURFACE_KIND = "campaign_set_metric_comparison"


def build_notebook_collection_set_choices(collection_visuals: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return selectable campaign-set choices from materialized collection visuals."""

    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
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
    return [{"key": key, "label": labels[key], "visual_count": counts[key]} for key in order]


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
        choice.setdefault("surface_kind", CAMPAIGN_SET_COMPARISON_SURFACE_KIND)
        choice.setdefault("kind", CAMPAIGN_SET_COMPARISON_SURFACE_KIND)
        choice["label"] = _unique_label(_choice_label(choice), labels_seen)
        choices.append(choice)
    return choices


def build_notebook_collection_visual_card_rows(choice: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build compact evidence rows for a materialized collection visual."""

    interval = mapping(choice.get("interval"))
    rows = [
        {"field": "visual", "value": choice.get("label") or choice.get("id") or "not recorded"},
        {"field": "campaign set", "value": choice.get("comparison_set_label") or "not recorded"},
        {"field": "surface", "value": choice.get("surface_kind") or "not recorded"},
        {"field": "source plot", "value": choice.get("source_plot_name") or "not recorded"},
        {"field": "relationship", "value": choice.get("relationship_id") or "not recorded"},
        {"field": "group key", "value": choice.get("group_key") or "not recorded"},
        {"field": "metric", "value": choice.get("metric") or "not recorded"},
        {"field": "metric label", "value": choice.get("metric_label") or "not recorded"},
        {"field": "metric expression", "value": choice.get("metric_expression") or "not recorded"},
        {"field": "axis scale", "value": _axis_scale_text(mapping(choice.get("axis_scale")))},
        {"field": "cohort", "value": choice.get("cohort") or "not recorded"},
        {"field": "summary", "value": choice.get("summary") or "not recorded"},
        {"field": "interval", "value": choice.get("interval_kind") or "not recorded"},
        {"field": "rows", "value": choice.get("row_count") if choice.get("row_count") is not None else "not recorded"},
        {"field": "freshness", "value": mapping(choice.get("freshness")).get("status") or "not recorded"},
        {"field": "manifest", "value": choice.get("manifest_path") or "not generated"},
        {"field": "tidy csv", "value": choice.get("tidy_csv") or "not generated"},
    ]
    if interval:
        rows.append({"field": "interval unit", "value": interval.get("unit") or "not recorded"})
        rows.append({"field": "is confidence interval", "value": bool(interval.get("is_confidence_interval"))})
    return rows


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
