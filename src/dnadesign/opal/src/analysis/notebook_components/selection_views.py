"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/selection_views.py

Selection-view controls for generated OPAL notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

from ._support import display_name, mapping, sequence


def build_notebook_selection_view_options(view_model: Mapping[str, Any]) -> dict[str, str]:
    """Build display-label to view-ID options for one campaign."""

    campaign = mapping(view_model.get("campaign"))
    views = sequence(campaign.get("selection_views"))
    if not views:
        raise ValueError("Campaign view model has no selection views.")
    options: dict[str, str] = {}
    for raw_view in views:
        view = mapping(raw_view)
        view_id = str(view.get("id") or "").strip()
        if not view_id:
            raise ValueError("Campaign selection views require non-empty IDs.")
        label = "AND" if view_id.lower() == "and" else display_name(view_id)
        if label in options:
            raise ValueError(f"Selection view display label {label!r} is not unique.")
        options[label] = view_id
    return options


def resolve_notebook_selection_view(view_model: Mapping[str, Any], selection_view_id: str) -> Mapping[str, Any]:
    """Resolve one declared selection view by exact ID."""

    campaign = mapping(view_model.get("campaign"))
    matches = [
        mapping(raw_view)
        for raw_view in sequence(campaign.get("selection_views"))
        if str(mapping(raw_view).get("id") or "") == selection_view_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Selection view {selection_view_id!r} must resolve exactly once; found {len(matches)}.")
    return matches[0]


__all__ = ["build_notebook_selection_view_options", "resolve_notebook_selection_view"]
