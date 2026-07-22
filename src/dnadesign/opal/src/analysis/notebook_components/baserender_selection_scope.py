"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender_selection_scope.py

Selection-view scope for a role-selected BaseRender campaign.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .selection_views import build_notebook_selection_view_options


def build_notebook_baserender_role_control(
    *,
    active_view_mode: str,
    role_choices: list[Mapping[str, Any]],
    mo: Any,
) -> Any | None:
    """Build the campaign-set label-source control when role choices exist."""

    if active_view_mode != "Campaign set" or not role_choices:
        return None
    labels = [str(choice.get("label") or "").strip() for choice in role_choices]
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError("BaseRender label-source choices require unique non-empty labels.")
    return mo.ui.dropdown(labels, value=labels[0], label="Label source")


def resolve_notebook_baserender_campaign_model(
    *,
    active_view_mode: str,
    campaigns: list[Mapping[str, Any]],
    role_choices: list[Mapping[str, Any]],
    role_selector_value: Any | None,
    selected_campaign_model: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    """Resolve the exact campaign selected as the BaseRender evidence source."""

    if active_view_mode != "Campaign set" or role_selector_value is None:
        return selected_campaign_model, None
    selected_label = str(role_selector_value).strip()
    choices = [choice for choice in role_choices if str(choice.get("label") or "").strip() == selected_label]
    if len(choices) != 1:
        raise ValueError(f"BaseRender label source {selected_label!r} must resolve exactly once; found {len(choices)}.")
    campaign_slug = str(choices[0].get("campaign_slug") or "").strip()
    matches = [
        campaign
        for campaign in campaigns
        if str((campaign.get("campaign") or {}).get("slug") or "").strip() == campaign_slug
    ]
    if len(matches) != 1:
        raise ValueError(f"BaseRender campaign {campaign_slug!r} must resolve exactly once; found {len(matches)}.")
    return matches[0], choices[0]


def build_notebook_baserender_selection_view_control(
    *,
    active_view_mode: str,
    campaign_model: Mapping[str, Any],
    mo: Any,
) -> tuple[dict[str, str], Any | None]:
    """Build a campaign-set-only view control from the role-selected campaign."""

    if active_view_mode != "Campaign set":
        return {}, None
    options = build_notebook_selection_view_options(campaign_model)
    if len(options) <= 1:
        return options, None
    labels = list(options)
    return options, mo.ui.dropdown(options, value=labels[0], label="Selection view")


def resolve_notebook_baserender_selection_view_id(
    *,
    active_view_mode: str,
    selection_view_options: Mapping[str, str],
    selector_value: Any | None,
    campaign_selection_view_id: str,
) -> str:
    """Resolve the active BaseRender view without borrowing another campaign's vocabulary."""

    if active_view_mode != "Campaign set":
        return str(campaign_selection_view_id)
    if selector_value is not None:
        resolved = str(selector_value).strip()
    else:
        values = [str(value).strip() for value in selection_view_options.values()]
        if len(values) != 1:
            raise ValueError(f"BaseRender selection view must resolve exactly once; found {len(values)}.")
        resolved = values[0]
    if resolved not in set(selection_view_options.values()):
        raise ValueError(f"Unknown BaseRender selection view {resolved!r} for the role-selected campaign.")
    return resolved


__all__ = [
    "build_notebook_baserender_role_control",
    "build_notebook_baserender_selection_view_control",
    "resolve_notebook_baserender_campaign_model",
    "resolve_notebook_baserender_selection_view_id",
]
