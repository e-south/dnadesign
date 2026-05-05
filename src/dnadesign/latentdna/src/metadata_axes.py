"""Generic metadata axis styling for notebook and plot runtimes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .labels import humanize_label

DEFAULT_NONCANONICAL_CATEGORY = "__latentdna_noncanonical__"
DEFAULT_NONCANONICAL_COLOR = "#9AA5B1"


@dataclass(frozen=True)
class AxisStyle:
    axis_id: str
    column: str
    label: str | None = None
    kind: str | None = None
    category_order: list[str] = field(default_factory=list)
    display_labels: dict[str, str] = field(default_factory=dict)
    compact_display_labels: dict[str, str] = field(default_factory=dict)
    category_colors: dict[str, str] = field(default_factory=dict)
    ordinal_subset: list[str] = field(default_factory=list)
    metric_labels: dict[str, str] = field(default_factory=dict)
    noncanonical_bucket: str | None = None
    noncanonical_label: str | None = None
    include_noncanonical_in_legend: bool = False
    canonical_row_selectors: list[dict[str, object]] = field(default_factory=list)
    canonical_row_match: str = "any"
    canonical_values: list[str] = field(default_factory=list)


def normalize_category_key(value: object) -> str:
    text = str(value or "").strip()
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None and numeric.is_integer():
        text = str(int(numeric))
    return text.lower().replace(" ", "_")


def _model_or_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {
        key: getattr(value, key) for key in dir(value) if not key.startswith("_") and not callable(getattr(value, key))
    }


def _noncanonical_policy_payload(axis: object) -> dict[str, object]:
    payload = _model_or_mapping(axis)
    policy = payload.get("noncanonical_policy")
    return _model_or_mapping(policy) if policy is not None else {}


def _axis_style_from_config(axis_id: str, axis: object) -> AxisStyle:
    payload = _model_or_mapping(axis)
    policy = _noncanonical_policy_payload(axis)
    bucket = policy.get("bucket", payload.get("noncanonical_bucket"))
    label = policy.get("label", payload.get("noncanonical_label"))
    selectors = policy.get("canonical_row_selectors") or payload.get("canonical_row_selectors") or []
    return AxisStyle(
        axis_id=str(axis_id),
        column=str(payload.get("column") or ""),
        label=str(payload.get("label")) if payload.get("label") is not None else None,
        kind=str(payload.get("kind")) if payload.get("kind") is not None else None,
        category_order=[str(value) for value in payload.get("category_order", [])],
        display_labels={str(key): str(value) for key, value in dict(payload.get("display_labels", {}) or {}).items()},
        compact_display_labels={
            str(key): str(value) for key, value in dict(payload.get("compact_display_labels", {}) or {}).items()
        },
        category_colors={str(key): str(value) for key, value in dict(payload.get("category_colors", {}) or {}).items()},
        ordinal_subset=[str(value) for value in payload.get("ordinal_subset", [])],
        metric_labels={str(key): str(value) for key, value in dict(payload.get("metric_labels", {}) or {}).items()},
        noncanonical_bucket=str(bucket) if bucket is not None else None,
        noncanonical_label=str(label) if label is not None else None,
        include_noncanonical_in_legend=bool(
            policy.get("include_in_legend", payload.get("include_noncanonical_in_legend", False))
        ),
        canonical_row_selectors=[
            _model_or_mapping(selector) for selector in selectors if isinstance(_model_or_mapping(selector), dict)
        ],
        canonical_row_match=str(policy.get("canonical_row_match") or payload.get("canonical_row_match") or "any"),
        canonical_values=[str(value) for value in policy.get("canonical_values", payload.get("canonical_values", []))],
    )


def axis_style_map_from_config(config: object) -> dict[str, AxisStyle]:
    metadata = getattr(config, "metadata", None)
    axes = getattr(metadata, "axes", {}) if metadata is not None else {}
    styles: dict[str, AxisStyle] = {}
    for axis_id, axis in dict(axes or {}).items():
        style = _axis_style_from_config(str(axis_id), axis)
        if style.column:
            styles[style.column] = style
    return styles


def axis_style_map_from_payload(payload: Mapping[str, object] | None) -> dict[str, AxisStyle]:
    styles: dict[str, AxisStyle] = {}
    for column, raw_style in dict(payload or {}).items():
        style = _axis_style_from_config(str(_model_or_mapping(raw_style).get("axis_id") or column), raw_style)
        if style.column:
            styles[style.column] = style
    return styles


def axis_styles_payload(styles: Mapping[str, AxisStyle]) -> dict[str, dict[str, object]]:
    return {
        column: {
            "axis_id": style.axis_id,
            "column": style.column,
            "label": style.label,
            "kind": style.kind,
            "category_order": list(style.category_order),
            "display_labels": dict(style.display_labels),
            "compact_display_labels": dict(style.compact_display_labels),
            "category_colors": dict(style.category_colors),
            "ordinal_subset": list(style.ordinal_subset),
            "metric_labels": dict(style.metric_labels),
            "noncanonical_bucket": style.noncanonical_bucket,
            "noncanonical_label": style.noncanonical_label,
            "include_noncanonical_in_legend": style.include_noncanonical_in_legend,
            "canonical_row_selectors": list(style.canonical_row_selectors),
            "canonical_row_match": style.canonical_row_match,
            "canonical_values": list(style.canonical_values),
        }
        for column, style in styles.items()
    }


def _canonical_values(style: AxisStyle) -> set[str]:
    values = [
        *style.category_order,
        *style.ordinal_subset,
        *style.canonical_values,
        *style.display_labels,
        *style.compact_display_labels,
        *style.category_colors,
    ]
    return {normalize_category_key(value) for value in values if str(value).strip()}


def _selector_matches(row: Mapping[str, object], selector: Mapping[str, object]) -> bool:
    column = str(selector.get("column") or "").strip()
    if not column:
        return False
    if column not in row:
        return False
    value = row.get(column)
    if selector.get("equals") is not None:
        return str(value) == str(selector.get("equals"))
    in_values = selector.get("in_values") or []
    if in_values:
        return str(value) in {str(item) for item in in_values}
    if bool(selector.get("non_null", False)):
        return value is not None and str(value).strip() != ""
    return False


def _row_in_canonical_scope(style: AxisStyle, row: Mapping[str, object] | None) -> bool:
    if not style.canonical_row_selectors or row is None:
        return True
    applicable_selectors = [
        selector for selector in style.canonical_row_selectors if str(selector.get("column") or "").strip() in row
    ]
    if not applicable_selectors:
        return True
    matches = [_selector_matches(row, selector) for selector in applicable_selectors]
    if style.canonical_row_match == "all":
        return all(matches)
    return any(matches)


def _canonical_value(style: AxisStyle, value: object) -> str:
    text = str(value or "").strip()
    normalized = normalize_category_key(text)
    for candidate in [
        *style.category_order,
        *style.ordinal_subset,
        *style.canonical_values,
        *style.display_labels,
        *style.compact_display_labels,
        *style.category_colors,
    ]:
        if normalize_category_key(candidate) == normalized:
            return str(candidate)
    return text


def normalize_axis_category(style: AxisStyle | None, value: object, *, row: Mapping[str, object] | None = None) -> str:
    if style is None:
        text = str(value)
        return text if text.strip() else "NA"
    text = str(value or "").strip()
    bucket = style.noncanonical_bucket
    if not text:
        return bucket or "NA"
    if bucket is not None and not _row_in_canonical_scope(style, row):
        return bucket
    canonical = _canonical_value(style, text)
    canonical_values = _canonical_values(style)
    if bucket is not None and canonical_values and normalize_category_key(canonical) not in canonical_values:
        return bucket
    return canonical


def ordered_categories_for_axis(style: AxisStyle | None, values: Sequence[object]) -> list[str]:
    unique = {str(value) for value in values if str(value).strip()}
    if style is None:
        return sorted(unique, key=lambda value: value.casefold())
    order = {normalize_category_key(value): index for index, value in enumerate(style.category_order)}
    bucket = style.noncanonical_bucket
    return sorted(
        unique,
        key=lambda value: (
            order.get(normalize_category_key(value), len(order) + (1 if value != bucket else 2)),
            value.casefold(),
        ),
    )


def legend_categories(style: AxisStyle | None, values: Sequence[object]) -> list[str]:
    categories = ordered_categories_for_axis(style, values)
    if style is None or style.include_noncanonical_in_legend or style.noncanonical_bucket is None:
        return categories
    return [category for category in categories if category != style.noncanonical_bucket]


def axis_display_label(style: AxisStyle | None, fallback: object) -> str:
    if style is not None and style.label:
        return style.label
    return humanize_label(fallback)


def _label_lookup(mapping: Mapping[str, str], value: object) -> str | None:
    text = str(value or "").strip()
    if text in mapping:
        return mapping[text]
    normalized = normalize_category_key(text)
    for key, label in mapping.items():
        if normalize_category_key(key) == normalized:
            return label
    return None


def axis_display_text(style: AxisStyle | None, value: object, *, compact: bool = False) -> str:
    if style is not None:
        if style.noncanonical_bucket is not None and str(value) == style.noncanonical_bucket:
            return style.noncanonical_label or humanize_label(value)
        if compact:
            compact_label = _label_lookup(style.compact_display_labels, value)
            if compact_label is not None:
                return compact_label
        label = _label_lookup(style.display_labels, value)
        if label is not None:
            return label
    return humanize_label(value)


def axis_color_map(
    style: AxisStyle | None,
    categories: Sequence[str],
    *,
    fallback_palette: Sequence[str],
) -> dict[str, str]:
    colors: dict[str, str] = {}
    fallback_index = 0
    for category in categories:
        configured = _label_lookup(style.category_colors, category) if style is not None else None
        if configured is not None:
            colors[str(category)] = configured
            continue
        if style is not None and style.noncanonical_bucket is not None and category == style.noncanonical_bucket:
            colors[str(category)] = DEFAULT_NONCANONICAL_COLOR
            continue
        colors[str(category)] = fallback_palette[fallback_index % len(fallback_palette)]
        fallback_index += 1
    if style is not None and style.noncanonical_bucket is not None:
        colors.setdefault(style.noncanonical_bucket, DEFAULT_NONCANONICAL_COLOR)
    return colors
