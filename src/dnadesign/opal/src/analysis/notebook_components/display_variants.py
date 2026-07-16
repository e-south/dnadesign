"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/display_variants.py

Notebook display-variant contracts for manifest-backed visual choices.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ._support import mapping, sequence


def group_notebook_display_variant_choices(
    choices: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse complete false/true manifest pairs into one notebook choice."""

    rows = [dict(choice) for choice in choices if isinstance(choice, Mapping)]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    positions: dict[tuple[str, str, str, str], int] = {}
    passthrough: dict[int, dict[str, Any]] = {}
    toggle_labels_by_id: dict[str, str] = {}
    for index, choice in enumerate(rows):
        toggle = _variant_toggle(choice)
        if toggle is None:
            passthrough[index] = choice
            continue
        known_label = toggle_labels_by_id.setdefault(toggle["id"], toggle["label"])
        if known_label != toggle["label"]:
            raise ValueError("A notebook display-variant toggle id must use one accessible label.")
        surface_label = _surface_label(choice)
        key = (surface_label, toggle["id"], toggle["label"], _selection_view_id(choice))
        grouped.setdefault(key, []).append(choice)
        positions.setdefault(key, index)

    collapsed: dict[int, dict[str, Any]] = dict(passthrough)
    for key, variants in grouped.items():
        false_variant, true_variant, toggle = _validated_variant_pair(variants)
        false_variant = {
            **false_variant,
            "label": key[0],
            "notebook_toggle": {**toggle, "value": False},
        }
        true_variant = {
            **true_variant,
            "label": key[0],
            "notebook_toggle": {**toggle, "value": True},
        }
        base = dict(false_variant)
        base["display_variants"] = [false_variant, true_variant]
        collapsed[positions[key]] = base
    return _dedupe_choice_labels([collapsed[index] for index in sorted(collapsed)])


def build_notebook_display_variant_toggle(choice: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the validated switch contract for one grouped visual choice."""

    variants = _display_variants(choice)
    if variants is None:
        return None
    _, _, toggle = _validated_variant_pair(variants)
    return {"id": toggle["id"], "label": toggle["label"], "default": False}


def select_notebook_display_variant(
    choice: Mapping[str, Any],
    *,
    enabled: bool | None,
) -> dict[str, Any]:
    """Select one concrete display variant, using the unannotated variant by default."""

    variants = _display_variants(choice)
    if variants is None:
        return dict(choice)
    false_variant, true_variant, _ = _validated_variant_pair(variants)
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("Notebook display-variant selection must be boolean when provided.")
    return dict(true_variant if enabled else false_variant)


def _display_variants(choice: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    if "display_variants" not in choice:
        return None
    return [dict(item) for item in sequence(choice.get("display_variants")) if isinstance(item, Mapping)]


def _validated_variant_pair(
    variants: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = [dict(variant) for variant in variants if isinstance(variant, Mapping)]
    if len(rows) != 2:
        raise ValueError("Notebook display variants require exactly two manifest-backed choices.")

    toggles = [_variant_toggle(row) for row in rows]
    if any(toggle is None for toggle in toggles):
        raise ValueError("Notebook display variants require notebook_toggle metadata on both choices.")
    typed_toggles = [dict(toggle) for toggle in toggles if toggle is not None]
    ids = {toggle["id"] for toggle in typed_toggles}
    labels = {toggle["label"] for toggle in typed_toggles}
    values = [toggle["value"] for toggle in typed_toggles]
    if len(ids) != 1 or len(labels) != 1:
        raise ValueError("Notebook display variants require one matching toggle id and label.")
    if set(values) != {False, True}:
        raise ValueError("Notebook display variants require exactly one false and one true toggle value.")

    if _variant_semantic_signature(rows[0]) != _variant_semantic_signature(rows[1]):
        raise ValueError("Notebook display variants require matching plot kind, selection view, and scope.")
    surface_labels = {_surface_label(row) for row in rows}
    if len(surface_labels) != 1:
        raise ValueError("Notebook display variants require one shared surface label.")

    by_value = {toggle["value"]: row for row, toggle in zip(rows, typed_toggles, strict=True)}
    toggle = typed_toggles[0]
    return by_value[False], by_value[True], {"id": toggle["id"], "label": toggle["label"]}


def _variant_toggle(choice: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates: list[Mapping[str, Any]] = []
    direct = choice.get("notebook_toggle")
    if isinstance(direct, Mapping):
        candidates.append(direct)
    for option in sequence(choice.get("scope_options")) or [choice]:
        if not isinstance(option, Mapping):
            continue
        params = mapping(mapping(option.get("manifest")).get("params"))
        nested = params.get("notebook_toggle")
        if isinstance(nested, Mapping):
            candidates.append(nested)
    if not candidates:
        return None

    normalized = [_normalize_toggle(candidate) for candidate in candidates]
    first = normalized[0]
    if any(candidate != first for candidate in normalized[1:]):
        raise ValueError("All scope options in a notebook display variant require identical notebook_toggle metadata.")
    return first


def _normalize_toggle(toggle: Mapping[str, Any]) -> dict[str, Any]:
    toggle_id = str(toggle.get("id") or "").strip()
    label = str(toggle.get("label") or "").strip()
    value = toggle.get("value")
    if not toggle_id or not label:
        raise ValueError("notebook_toggle requires non-empty id and label values.")
    if not isinstance(value, bool):
        raise ValueError("notebook_toggle.value must be boolean.")
    return {"id": toggle_id, "label": label, "value": value}


def _surface_label(choice: Mapping[str, Any]) -> str:
    params = mapping(mapping(choice.get("manifest")).get("params"))
    label = str(params.get("surface_label") or choice.get("label") or "").strip()
    if not label:
        raise ValueError("Notebook display variants require a non-empty surface label.")
    return label


def _selection_view_id(choice: Mapping[str, Any]) -> str:
    manifest = mapping(choice.get("manifest"))
    return str(choice.get("selection_view_id") or manifest.get("selection_view_id") or "").strip()


def _dedupe_choice_labels(choices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for choice in choices:
        row = dict(choice)
        label = str(row.get("label") or "").strip()
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            view_id = _selection_view_id(row)
            suffix = view_id or str(row.get("filename") or row.get("path_label") or seen[label])
            row["label"] = f"{label} ({suffix})"
        rows.append(row)
    return rows


def _variant_semantic_signature(choice: Mapping[str, Any]) -> tuple[Any, ...]:
    manifest = mapping(choice.get("manifest"))
    kind = str(choice.get("kind") or manifest.get("kind") or "").strip()
    selection_view_id = str(choice.get("selection_view_id") or manifest.get("selection_view_id") or "").strip()
    scope_options = sequence(choice.get("scope_options")) or [choice]
    scopes = []
    for option in scope_options:
        if not isinstance(option, Mapping):
            continue
        option_manifest = mapping(option.get("manifest"))
        option_view = str(option.get("selection_view_id") or option_manifest.get("selection_view_id") or "").strip()
        scopes.append(
            (
                option_view or selection_view_id,
                _freeze(option.get("rounds")),
                str(option.get("run_id") or ""),
            )
        )
    return kind, selection_view_id, tuple(scopes)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


__all__ = [
    "build_notebook_display_variant_toggle",
    "group_notebook_display_variant_choices",
    "select_notebook_display_variant",
]
