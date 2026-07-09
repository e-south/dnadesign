"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/review_controls.py

Notebook component builders for OPAL notebook review control surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

_BASERENDER_SURFACE_KINDS = frozenset({"baserender", "campaign_set_baserender"})
_READER_EVIDENCE_SURFACE_KIND = "reader_evidence"
_VIEW_MODES = frozenset({"Campaign", "Campaign set"})


def render_notebook_review_control_surface(
    *,
    active_view_mode: str,
    mo: Any,
    campaign_ui: Any = None,
    view_mode_ui: Any = None,
    collection_set_ui: Any = None,
    visual_group_ui: Any = None,
    plot_ui: Any = None,
    plot_scope_ui: Any = None,
    baserender_role_ui: Any = None,
    baserender_round_ui: Any = None,
    baserender_run_ui: Any = None,
    baserender_record_selector: Any = None,
    reader_evidence_artifact_ui: Any = None,
    reader_evidence_time_ui: Any = None,
    selected_visual_choice: Mapping[str, Any] | None = None,
) -> Any | None:
    """Render the consolidated top control surface for an OPAL review notebook."""

    if active_view_mode not in _VIEW_MODES:
        raise ValueError("active_view_mode must be 'Campaign' or 'Campaign set'.")

    control_groups = [
        _primary_controls(
            active_view_mode=active_view_mode,
            campaign_ui=campaign_ui,
            view_mode_ui=view_mode_ui,
            collection_set_ui=collection_set_ui,
        ),
        _visual_controls(
            baserender_record_selector=baserender_record_selector,
            baserender_role_ui=baserender_role_ui,
            baserender_round_ui=baserender_round_ui,
            baserender_run_ui=baserender_run_ui,
            visual_group_ui=visual_group_ui,
            plot_scope_ui=plot_scope_ui,
            plot_ui=plot_ui,
            reader_evidence_artifact_ui=reader_evidence_artifact_ui,
            reader_evidence_time_ui=reader_evidence_time_ui,
            selected_visual_choice=selected_visual_choice,
        ),
    ]
    rows = [
        mo.hstack(controls, justify="start", align="end", wrap=True, gap=0.35)
        for controls in control_groups
        if controls
    ]
    return mo.vstack(rows, gap=0.35) if rows else None


def _primary_controls(
    *,
    active_view_mode: str,
    campaign_ui: Any,
    view_mode_ui: Any,
    collection_set_ui: Any,
) -> list[Any]:
    if active_view_mode == "Campaign":
        return _present(campaign_ui, view_mode_ui)
    return _present(view_mode_ui, collection_set_ui)


def _visual_controls(
    *,
    baserender_record_selector: Any,
    baserender_role_ui: Any,
    baserender_round_ui: Any,
    baserender_run_ui: Any,
    plot_scope_ui: Any,
    plot_ui: Any,
    reader_evidence_artifact_ui: Any,
    reader_evidence_time_ui: Any,
    visual_group_ui: Any,
    selected_visual_choice: Mapping[str, Any] | None,
) -> list[Any]:
    controls = _present(visual_group_ui, plot_ui)
    if _is_baserender_visual(selected_visual_choice):
        controls.extend(
            _present(
                baserender_role_ui,
                baserender_round_ui,
                baserender_run_ui,
                baserender_record_selector,
            )
        )
    elif _is_reader_evidence_visual(selected_visual_choice):
        controls.extend(_present(reader_evidence_artifact_ui, reader_evidence_time_ui))
    elif plot_scope_ui is not None:
        controls.append(plot_scope_ui)
    return controls


def _is_baserender_visual(selected_visual_choice: Mapping[str, Any] | None) -> bool:
    if selected_visual_choice is None:
        return False
    return selected_visual_choice.get("surface_kind") in _BASERENDER_SURFACE_KINDS


def _is_reader_evidence_visual(selected_visual_choice: Mapping[str, Any] | None) -> bool:
    if selected_visual_choice is None:
        return False
    return selected_visual_choice.get("surface_kind") == _READER_EVIDENCE_SURFACE_KIND


def _present(*items: Any) -> list[Any]:
    return [item for item in items if item is not None]


__all__ = ["render_notebook_review_control_surface"]
