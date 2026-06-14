"""Campaign-set visual surface models."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ...core.utils import ExitCodes, OpalError
from ...plots._mpl_utils import pretty_label
from .visual_kinds import collection_visual_surface_kind_for_view_kind

CAMPAIGN_SET_VISUAL_MODEL_SCHEMA_VERSION = "opal.campaign_set_visual_model.v1"
COMPARISON_SET_SCOPE = "comparison_set"
COLLECTION_SCOPE = "collection"


def build_campaign_set_collection_visual_model(
    campaigns: Iterable[Mapping[str, Any]],
    collection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build first-class collection visual choices independent of campaign selection."""

    campaign_list = [campaign for campaign in campaigns if isinstance(campaign, Mapping)]
    collection_map = collection if isinstance(collection, Mapping) else {}
    visuals: list[dict[str, Any]] = []
    comparison_sets_by_key: dict[str, dict[str, Any]] = {}
    for view in _sequence(collection_map.get("comparison_views")):
        if not isinstance(view, Mapping):
            continue
        for visual in _collection_visuals_from_view(view):
            visuals.append(visual)
            set_key = str(visual.get("comparison_set_key") or "").strip()
            if set_key:
                comparison_sets_by_key.setdefault(
                    set_key,
                    {
                        "key": set_key,
                        "label": visual.get("comparison_set_label") or set_key,
                        "match": dict(_mapping(visual.get("comparison_set_match"))),
                    },
                )
    return {
        "schema_version": CAMPAIGN_SET_VISUAL_MODEL_SCHEMA_VERSION,
        "collection_id": collection_map.get("collection_id"),
        "campaign_count": len(campaign_list),
        "visual_count": len(visuals),
        "comparison_set_count": len(comparison_sets_by_key),
        "comparison_sets": sorted(comparison_sets_by_key.values(), key=lambda row: str(row["label"])),
        "visuals": visuals,
    }


def _collection_visuals_from_view(view: Mapping[str, Any]) -> list[dict[str, Any]]:
    relationship = _mapping(view.get("relationship"))
    visual_id = _required_string(view.get("id"), field="comparison_views[].id")
    view_kind = _required_string(view.get("kind"), field=f"comparison view {visual_id!r}.kind")
    collection_visual_surface_kind_for_view_kind(view_kind)
    _required_string(view.get("source_plot_name"), field=f"comparison view {visual_id!r}.source_plot_name")
    _required_string(view.get("source_plot_kind"), field=f"comparison view {visual_id!r}.source_plot_kind")
    _required_string(view.get("interval_kind"), field=f"comparison view {visual_id!r}.interval_kind")
    comparison_scope = _required_string(
        view.get("comparison_scope"),
        field=f"comparison view {visual_id!r}.comparison_scope",
    )
    if comparison_scope not in {COLLECTION_SCOPE, COMPARISON_SET_SCOPE}:
        raise OpalError(
            f"Campaign collection comparison view {visual_id!r} comparison_scope must be one of "
            f"{sorted({COLLECTION_SCOPE, COMPARISON_SET_SCOPE})}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if comparison_scope == COMPARISON_SET_SCOPE:
        match_filters = _mapping(view.get("match_filters"))
        return [
            _collection_visual_from_view(view, relationship=relationship, comparison_set=comparison_set)
            for comparison_set in _comparison_sets(relationship)
            if _matches_filters(_mapping(comparison_set.get("match")), match_filters)
        ]
    return [
        _collection_visual_from_view(
            view,
            relationship=relationship,
            comparison_set={
                "key": "collection",
                "label": "All campaign sets",
                "match": {},
                "pairs": list(_sequence(relationship.get("pairs"))),
                "replicate_count": len(_sequence(relationship.get("pairs"))),
            },
        )
    ]


def _collection_visual_from_view(
    view: Mapping[str, Any],
    *,
    relationship: Mapping[str, Any],
    comparison_set: Mapping[str, Any],
) -> dict[str, Any]:
    visual_id = _required_string(view.get("id"), field="comparison_views[].id")
    view_kind = _required_string(view.get("kind"), field=f"comparison view {visual_id!r}.kind")
    source_plot_name = _required_string(
        view.get("source_plot_name"),
        field=f"comparison view {visual_id!r}.source_plot_name",
    )
    source_plot_kind = _required_string(
        view.get("source_plot_kind"),
        field=f"comparison view {visual_id!r}.source_plot_kind",
    )
    comparison_scope = _required_string(
        view.get("comparison_scope"),
        field=f"comparison view {visual_id!r}.comparison_scope",
    )
    interval_kind = _required_string(
        view.get("interval_kind"),
        field=f"comparison view {visual_id!r}.interval_kind",
    )
    group_key = str(view.get("group_key") or relationship.get("role_dimension") or "")
    label = str(view.get("label") or visual_id or "Campaign-set comparison")
    surface_kind = collection_visual_surface_kind_for_view_kind(view_kind)
    comparison_set_key = str(comparison_set.get("key") or "")
    comparison_set_label = str(comparison_set.get("label") or comparison_set_key)
    return {
        "id": visual_id,
        "visual_id": visual_id,
        "label": label,
        "title": label,
        "kind": surface_kind,
        "view_kind": view_kind,
        "surface_kind": surface_kind,
        "source_plot_name": source_plot_name,
        "source_plot_kind": source_plot_kind,
        "comparison_scope": comparison_scope,
        "comparison_set_key": comparison_set_key,
        "comparison_set_label": comparison_set_label,
        "comparison_set_match": dict(_mapping(comparison_set.get("match"))),
        "match_filters": dict(_mapping(view.get("match_filters"))),
        "comparison_replicate_count": comparison_set.get("replicate_count"),
        "relationship_id": str(view.get("relationship_id") or relationship.get("id") or ""),
        "relationship_kind": str(relationship.get("kind") or ""),
        "role_dimension": str(relationship.get("role_dimension") or group_key),
        "left_role": relationship.get("left_role"),
        "right_role": relationship.get("right_role"),
        "match_on": list(_sequence(relationship.get("match_on"))),
        "replicate_on": list(_sequence(relationship.get("replicate_on"))),
        "pair_count": len(_sequence(comparison_set.get("pairs"))),
        "pairs": list(_sequence(comparison_set.get("pairs"))),
        "comparison_group_options": [group_key] if group_key else [],
        "group_key": group_key,
        "metric": str(view.get("metric") or ""),
        "cohort": str(view.get("cohort") or ""),
        "summary": str(view.get("summary") or ""),
        "interval_kind": interval_kind,
        "confidence_level": view.get("confidence_level"),
        "interpretation_note": view.get("interpretation_note"),
        "manifest_path": view.get("manifest_path"),
        "path": view.get("path"),
        "tidy_csv": view.get("tidy_csv"),
        "caption": view.get("caption"),
        "alt_text": view.get("alt_text"),
        "outputs": list(_sequence(view.get("outputs"))),
        "freshness": view.get("freshness"),
        "row_count": view.get("row_count"),
        "group_count": view.get("group_count"),
        "interval": view.get("interval"),
    }


def _comparison_sets(relationship: Mapping[str, Any]) -> list[dict[str, Any]]:
    replicate_on = {str(item) for item in _sequence(relationship.get("replicate_on")) if str(item).strip()}
    match_order = [str(item) for item in _sequence(relationship.get("match_on")) if str(item).strip()]
    grouped: dict[str, dict[str, Any]] = {}
    for pair in _sequence(relationship.get("pairs")):
        if not isinstance(pair, Mapping):
            continue
        match = dict(_mapping(pair.get("match")))
        set_match = {key: match[key] for key in match_order if key in match and key not in replicate_on}
        if not set_match:
            set_match = dict(match)
        key = _key_text(set_match)
        grouped.setdefault(
            key,
            {
                "key": key,
                "label": _comparison_set_label(set_match, match_order=match_order, replicate_on=replicate_on),
                "match": set_match,
                "pairs": [],
                "replicate_keys": set(),
            },
        )
        grouped[key]["pairs"].append(dict(pair))
        replicate_match = {key: value for key, value in match.items() if key in replicate_on}
        grouped[key]["replicate_keys"].add(_key_text(replicate_match))
    rows: list[dict[str, Any]] = []
    for row in grouped.values():
        replicate_keys = {str(key) for key in row.pop("replicate_keys") if str(key)}
        row["replicate_count"] = len(replicate_keys) or len(row["pairs"])
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["label"]))


def _comparison_set_label(match: Mapping[str, Any], *, match_order: list[str], replicate_on: set[str]) -> str:
    values = [
        _comparison_set_value_label(key, match[key]) for key in match_order if key in match and key not in replicate_on
    ]
    if not values:
        values = [_comparison_set_value_label(key, value) for key, value in sorted(match.items())]
    return " | ".join(values) if values else "Campaign set"


def _comparison_set_value_label(key: str, value: Any) -> str:
    text = str(value)
    if key == "target":
        return text[:1].upper() + text[1:]
    return pretty_label(text)


def _matches_filters(match: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    return all(str(match.get(key)) == str(value) for key, value in filters.items())


def _key_text(values: Mapping[str, Any]) -> str:
    return "|".join(f"{key}={values[key]}" for key in sorted(values))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _required_string(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise OpalError(
            f"Campaign-set collection visual field {field} must be a non-empty string.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return text
