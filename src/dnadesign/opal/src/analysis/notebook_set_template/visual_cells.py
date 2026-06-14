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
            build_notebook_plot_inventory_rows,
            build_notebook_visual_surface_model,
            selected_campaign_model,
        ):
            campaign_visual_surface_model = build_notebook_visual_surface_model(selected_campaign_model)
            campaign_plot_choices = campaign_visual_surface_model["choices"]
            plot_inventory_rows = build_notebook_plot_inventory_rows(campaign_visual_surface_model)
            plot_inventory_counts = campaign_visual_surface_model["inventory_status_counts"]
            return campaign_plot_choices, plot_inventory_counts, plot_inventory_rows
        """
    )


def _visual_memory_cell() -> str:
    return block(
        """
        @app.cell
        def _(mo):
            visual_label_memory, set_visual_label_memory = mo.state(None)
            return set_visual_label_memory, visual_label_memory
        """
    )


def _visual_choices_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            CAMPAIGN_SET_BASERENDER_SURFACE_KIND,
            active_view_mode,
            baserender_role_ui,
            build_notebook_collection_visual_choices,
            campaign_plot_choices,
            collection_visuals,
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
                if baserender_role_ui is not None and selected_campaign_baserender_contract.get("available"):
                    visual_choices.append(
                        {
                            "label": "Selected DenseGen sequence render",
                            "surface_kind": CAMPAIGN_SET_BASERENDER_SURFACE_KIND,
                            "title": "Selected DenseGen sequence render",
                        }
                    )
            else:
                visual_choices = []
                if selected_campaign_baserender_contract.get("available"):
                    visual_choices.append(
                        {
                            "label": "Selected sequence render",
                            "surface_kind": "baserender",
                            "title": "Selected sequence render",
                        }
                    )
                visual_choices.extend(campaign_plot_choices)
            return visual_choices
        """
    )


def _visual_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(active_view_mode, mo, set_visual_label_memory, visual_choices, visual_label_memory):
            if visual_choices:
                _labels = [choice["label"] for choice in visual_choices]
                _preferred_visual_label = visual_label_memory()
                _preferred_visual_label = _preferred_visual_label if _preferred_visual_label in _labels else _labels[0]
                if active_view_mode == "Campaign set":
                    plot_ui = mo.ui.dropdown(
                        _labels,
                        value=_preferred_visual_label,
                        label="Collection visual",
                        on_change=set_visual_label_memory,
                    )
                else:
                    plot_ui = mo.ui.dropdown(
                        _labels,
                        value=_preferred_visual_label,
                        label="Visual surface",
                        on_change=set_visual_label_memory,
                    )
            else: plot_ui = None
            return plot_ui
        """
    )


def _selected_visual_cell() -> str:
    return block(
        """
        @app.cell
        def _(plot_ui, visual_choices):
            if plot_ui is None:
                selected_visual_choice = None
            else:
                _selected = str(plot_ui.value)
                selected_visual_choice = next(choice for choice in visual_choices if choice["label"] == _selected)
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
                or selected_visual_choice.get("surface_kind") in {"baserender", "campaign_set_baserender"}
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
