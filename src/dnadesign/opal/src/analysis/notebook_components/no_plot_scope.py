"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/no_plot_scope.py

Notebook component builders for no plot scope OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

from ._support import mapping, sequence
from .overview import _campaign_description, _objective_target_summary
from .plots import build_notebook_visual_surface_model


def build_notebook_no_plot_scope_rows(
    view_model: Mapping[str, Any],
    *,
    selection_view: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build operator-facing scope rows for campaign notebooks with no plot media."""

    campaign = mapping(view_model.get("campaign"))
    if selection_view is None:
        selection_views = sequence(campaign.get("selection_views"))
        if len(selection_views) != 1 or not isinstance(selection_views[0], Mapping):
            raise ValueError("No-plot scope requires an explicit selection view for multi-view campaigns.")
        selection_view = selection_views[0]
    status = mapping(view_model.get("status"))
    visual_surface = build_notebook_visual_surface_model(view_model)
    configured_count = len(sequence(view_model.get("configured_plots")))
    plot_media_count = len(visual_surface["choices"])
    missing_count = len(visual_surface["missing_outputs"])
    stale_count = len(sequence(view_model.get("stale_artifacts")))
    config_path = str(campaign.get("config_path") or "<campaign.yaml>")
    round_selector = str(status.get("round_selector") or "latest")
    rows = [
        {"field": "scope", "value": _campaign_description(campaign)},
        {"field": "campaign", "value": campaign.get("slug") or "unknown"},
        {"field": "status", "value": status.get("progress_status") or "unknown"},
        {"field": "campaign metadata", "value": _campaign_metadata_summary(campaign)},
        {"field": "objective setpoint", "value": _objective_target_summary(selection_view)},
        {
            "field": "label source",
            "value": (
                f"{campaign.get('label_source') or 'unknown'}; "
                f"rounds={status.get('round_count') or 0}; latest_run={status.get('latest_run_id') or 'none'}"
            ),
        },
        {
            "field": "plot state",
            "value": (
                f"configured={configured_count}; media_choices={plot_media_count}; "
                f"missing_outputs={missing_count}; stale_artifacts={stale_count}"
            ),
        },
        {
            "field": "evidence boundary",
            "value": (
                "No plot media are available for the selected round; "
                "do not draw visual or biological conclusions from this notebook state."
            ),
        },
        {
            "field": "next commands",
            "value": (
                f"uv run opal run -c {config_path} --round {round_selector} --resume --json; "
                f"uv run opal plot -c {config_path} --round all; "
                f"uv run opal review -c {config_path} --json"
            ),
        },
    ]
    if visual_surface["missing_outputs"]:
        rows.append({"field": "missing plot outputs", "value": ", ".join(visual_surface["missing_outputs"])})
    return rows


def _campaign_metadata_summary(campaign: Mapping[str, Any]) -> str:
    metadata = mapping(campaign.get("metadata"))
    parts = []
    for key in ("response_axis", "comparison_group", "label_family_id", "label_oracle_kind", "label_split_id"):
        if metadata.get(key):
            parts.append(f"{key}={metadata[key]}")
    for key in sorted(metadata):
        if key.endswith(("_label_family_id", "_oracle_kind", "_split_id")) and metadata.get(key):
            value = f"{key}={metadata[key]}"
            if value not in parts:
                parts.append(value)
    return "; ".join(parts) if parts else "not recorded"
