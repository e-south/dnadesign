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
    selection_view_ui: Any = None,
    view_mode_ui: Any = None,
    collection_set_ui: Any = None,
    visual_group_ui: Any = None,
    plot_ui: Any = None,
    plot_scope_ui: Any = None,
    layered_scatter_controls: Mapping[str, Any] | None = None,
    baserender_role_ui: Any = None,
    baserender_selection_view_ui: Any = None,
    baserender_round_ui: Any = None,
    baserender_run_ui: Any = None,
    baserender_record_selector: Any = None,
    reader_evidence_artifact_ui: Any = None,
    selected_visual_choice: Mapping[str, Any] | None = None,
) -> Any | None:
    """Render the consolidated top control surface for an OPAL review notebook."""

    if active_view_mode not in _VIEW_MODES:
        raise ValueError("active_view_mode must be 'Campaign' or 'Campaign set'.")

    context_controls = _primary_controls(
        active_view_mode=active_view_mode,
        campaign_ui=campaign_ui,
        selection_view_ui=selection_view_ui,
        view_mode_ui=view_mode_ui,
        collection_set_ui=collection_set_ui,
        selected_visual_choice=selected_visual_choice,
    )
    review_controls, scope_controls, display_controls = _visual_control_groups(
        baserender_record_selector=baserender_record_selector,
        baserender_role_ui=baserender_role_ui,
        baserender_selection_view_ui=baserender_selection_view_ui,
        baserender_round_ui=baserender_round_ui,
        baserender_run_ui=baserender_run_ui,
        visual_group_ui=visual_group_ui,
        plot_scope_ui=plot_scope_ui,
        plot_ui=plot_ui,
        layered_scatter_controls=layered_scatter_controls,
        reader_evidence_artifact_ui=reader_evidence_artifact_ui,
        selected_visual_choice=selected_visual_choice,
    )
    rows = [
        *(_control_row(mo, controls) for controls in _chunks(context_controls + review_controls, 3)),
        *(_control_row(mo, controls) for controls in _chunks(scope_controls, 1)),
        *(_control_row(mo, controls) for controls in _chunks(display_controls, 3)),
    ]
    return mo.vstack(rows, gap=0.45) if rows else None


def _primary_controls(
    *,
    active_view_mode: str,
    campaign_ui: Any,
    selection_view_ui: Any,
    view_mode_ui: Any,
    collection_set_ui: Any,
    selected_visual_choice: Mapping[str, Any] | None,
) -> list[Any]:
    if active_view_mode == "Campaign":
        scoped_selection_view_ui = (
            None
            if str((selected_visual_choice or {}).get("selection_scope") or "selection_view") == "campaign"
            else selection_view_ui
        )
        return _present(campaign_ui, scoped_selection_view_ui, view_mode_ui)
    return _present(view_mode_ui, collection_set_ui)


def _visual_control_groups(
    *,
    baserender_record_selector: Any,
    baserender_role_ui: Any,
    baserender_selection_view_ui: Any,
    baserender_round_ui: Any,
    baserender_run_ui: Any,
    plot_scope_ui: Any,
    plot_ui: Any,
    layered_scatter_controls: Mapping[str, Any] | None,
    reader_evidence_artifact_ui: Any,
    visual_group_ui: Any,
    selected_visual_choice: Mapping[str, Any] | None,
) -> tuple[list[Any], list[Any], list[Any]]:
    review_controls = _present(visual_group_ui, plot_ui)
    scope_controls = _present(plot_scope_ui)
    display_controls: list[Any] = []
    if _is_baserender_visual(selected_visual_choice):
        display_controls.extend(
            _present(
                baserender_role_ui,
                baserender_selection_view_ui,
                baserender_round_ui,
                baserender_run_ui,
                baserender_record_selector,
            )
        )
    elif _is_reader_evidence_visual(selected_visual_choice):
        display_controls.extend(_present(reader_evidence_artifact_ui))
    else:
        scatter = dict(layered_scatter_controls or {})
        figure = scatter.get("figure")
        figure_mode = str(figure.value) if figure is not None else "publication_2d"
        selected = scatter.get("selected")
        show_selected = bool(getattr(selected, "value", True)) if selected is not None else False
        display_controls.extend(
            _present(
                figure,
                scatter.get("prediction_pool"),
                selected,
                scatter.get("selection_rounds") if show_selected else None,
                scatter.get("observed_batches"),
                scatter.get("labels") if figure_mode == "publication_2d" else None,
            )
        )
    return review_controls, scope_controls, display_controls


def _control_row(mo: Any, controls: list[Any]) -> Any:
    return mo.hstack(
        controls,
        justify="start",
        align="end",
        wrap=True,
        gap=0.5,
        widths="equal",
    )


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


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
