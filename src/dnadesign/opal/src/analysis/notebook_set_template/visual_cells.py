"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/visual_cells.py

Notebook-set template builders for visual cells OPAL analysis notebook set template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ._support import block
from .visual_panel_cells import render_visual_panel_cell


def render_visual_cells() -> str:
    """Render campaign and campaign-set visual controls."""

    return "\n".join(
        (
            _campaign_visual_model_cell(),
            _visual_memory_cell(),
            _visual_choices_cell(),
            _visual_group_selector_cell(),
            _filtered_visual_choices_cell(),
            _visual_selector_cell(),
            _selected_visual_cell(),
            _visual_scope_cell(),
            render_visual_panel_cell(),
        )
    )


def _campaign_visual_model_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            build_notebook_visual_surface_model,
            selected_campaign_model,
        ):
            campaign_visual_surface_model = build_notebook_visual_surface_model(selected_campaign_model)
            campaign_plot_choices = campaign_visual_surface_model["choices"]
            return campaign_plot_choices
        """
    )


def _visual_memory_cell() -> str:
    return block(
        """
        @app.cell
        def _(mo):
            visual_group_label_memory, set_visual_group_label_memory = mo.state(None)
            visual_label_memory, set_visual_label_memory = mo.state(None)
            return (
                set_visual_group_label_memory,
                set_visual_label_memory,
                visual_group_label_memory,
                visual_label_memory,
            )
        """
    )


def _visual_choices_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            CAMPAIGN_SET_BASERENDER_SURFACE_KIND,
            active_view_mode,
            annotate_notebook_visual_choices,
            baserender_role_ui,
            build_notebook_collection_visual_choices,
            build_notebook_reader_evidence_visual_choices,
            campaign_plot_choices,
            collection_visuals,
            reader_evidence_surface,
            selected_baserender_ids,
            selected_campaign_baserender_contract,
            selected_collection_set_choice,
        ):
            if active_view_mode == "Campaign set":
                _set_key = (
                    selected_collection_set_choice.get("key")
                    if selected_collection_set_choice is not None
                    else None
                )
                visual_choices = build_notebook_collection_visual_choices(
                    collection_visuals,
                    comparison_set_key=_set_key,
                )
                if (
                    baserender_role_ui is not None
                    and selected_campaign_baserender_contract.get("available")
                    and selected_baserender_ids
                ):
                    visual_choices.append(
                        {
                            "label": "Selected sequence render",
                            "surface_kind": CAMPAIGN_SET_BASERENDER_SURFACE_KIND,
                            "title": "Selected sequence render",
                        }
                    )
            else:
                visual_choices = []
                if selected_campaign_baserender_contract.get("available") and selected_baserender_ids:
                    visual_choices.append(
                        {
                            "label": "Selected sequence render",
                            "surface_kind": "baserender",
                            "title": "Selected sequence render",
                        }
                    )
                visual_choices.extend(campaign_plot_choices)
                visual_choices.extend(build_notebook_reader_evidence_visual_choices(reader_evidence_surface))
            visual_choices = annotate_notebook_visual_choices(visual_choices)
            return visual_choices
        """
    )


def _visual_group_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            build_notebook_visual_group_options,
            mo,
            set_visual_group_label_memory,
            visual_choices,
            visual_group_label_memory,
        ):
            visual_group_options = build_notebook_visual_group_options(visual_choices)
            if visual_group_options:
                _preferred_group_label = visual_group_label_memory()
                _preferred_group_label = (
                    _preferred_group_label
                    if _preferred_group_label in visual_group_options
                    else visual_group_options[0]
                )
                visual_group_ui = mo.ui.dropdown(
                    visual_group_options,
                    value=_preferred_group_label,
                    label="Review section",
                    on_change=set_visual_group_label_memory,
                )
            else:
                visual_group_ui = None
            return visual_group_options, visual_group_ui
        """
    )


def _filtered_visual_choices_cell() -> str:
    return block(
        """
        @app.cell
        def _(filter_notebook_visual_choices_by_group, visual_choices, visual_group_ui):
            selected_visual_group_label = (
                str(visual_group_ui.value) if visual_group_ui is not None else None
            )
            visual_choices_in_group = filter_notebook_visual_choices_by_group(
                visual_choices,
                selected_visual_group_label,
            )
            return selected_visual_group_label, visual_choices_in_group
        """
    )


def _visual_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(mo, set_visual_label_memory, visual_choices_in_group, visual_label_memory):
            if visual_choices_in_group:
                _labels = [choice["label"] for choice in visual_choices_in_group]
                _preferred_visual_label = visual_label_memory()
                _preferred_visual_label = _preferred_visual_label if _preferred_visual_label in _labels else _labels[0]
                plot_ui = mo.ui.dropdown(
                    _labels,
                    value=_preferred_visual_label,
                    label="Deliverable",
                    on_change=set_visual_label_memory,
                )
            else:
                plot_ui = None
            return plot_ui
        """
    )


def _selected_visual_cell() -> str:
    return block(
        """
        @app.cell
        def _(plot_ui, visual_choices_in_group):
            if plot_ui is None:
                selected_visual_choice = None
            else:
                _selected = str(plot_ui.value)
                selected_visual_choice = next(
                    choice for choice in visual_choices_in_group if choice["label"] == _selected
                )
            return selected_visual_choice
        """
    )


def _visual_scope_cell() -> str:
    return block(
        """
        @app.cell
        def _(active_view_mode, build_notebook_plot_scope_options, mo, selected_visual_choice):
            if (
                active_view_mode == "Campaign set"
                or selected_visual_choice is None
                or selected_visual_choice.get("surface_kind")
                in {"baserender", "campaign_set_baserender", "reader_evidence"}
            ):
                plot_scope_options = []
                plot_scope_ui = None
            else:
                plot_scope_options = build_notebook_plot_scope_options(selected_visual_choice)
                if len(plot_scope_options) > 1:
                    _scope_labels = [option["label"] for option in plot_scope_options]
                    _scope_control_label = str(plot_scope_options[0].get("control_label") or "Plot scope")
                    plot_scope_ui = mo.ui.dropdown(
                        _scope_labels,
                        value=_scope_labels[0],
                        label=_scope_control_label,
                    )
                else:
                    plot_scope_ui = None
            return plot_scope_options, plot_scope_ui
        """
    )


__all__ = ["render_visual_cells"]
