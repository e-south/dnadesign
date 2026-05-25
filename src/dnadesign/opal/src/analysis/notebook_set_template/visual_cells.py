from __future__ import annotations

from ._support import block
from .visual_panel_cells import render_visual_panel_cell


def render_visual_cells() -> str:
    """Render campaign-set visual selector and manifest-backed plot panel cells."""

    return "\n".join(
        (
            _visual_model_cell(),
            _visual_memory_cell(),
            _visual_selector_cell(),
            _selected_visual_cell(),
            _visual_scope_cell(),
            _visual_comparison_cell(),
            render_visual_panel_cell(),
        )
    )


def _visual_model_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            build_notebook_campaign_set_visual_choices,
            build_notebook_plot_inventory_rows,
            build_notebook_visual_surface_model,
            campaigns,
            collection,
            selected_campaign_model,
        ):
            visual_surface_model = build_notebook_visual_surface_model(selected_campaign_model)
            plot_choices = visual_surface_model["choices"]
            visual_choices = build_notebook_campaign_set_visual_choices(plot_choices, campaigns, collection)
            plot_inventory_rows = build_notebook_plot_inventory_rows(visual_surface_model)
            plot_inventory_counts = visual_surface_model["inventory_status_counts"]
            return plot_choices, plot_inventory_rows, plot_inventory_counts, visual_choices
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


def _visual_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(mo, set_visual_label_memory, visual_choices, visual_label_memory):
            if visual_choices:
                _labels = [choice["label"] for choice in visual_choices]
                _preferred_visual_label = visual_label_memory()
                _preferred_visual_label = _preferred_visual_label if _preferred_visual_label in _labels else _labels[0]
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
        def _(build_notebook_plot_scope_options, mo, selected_visual_choice):
            if (
                selected_visual_choice is None
                or selected_visual_choice.get("surface_kind") == "campaign_set_metric_comparison"
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


def _visual_comparison_cell() -> str:
    return block(
        """
        @app.cell
        def _(mo, selected_visual_choice):
            if (
                selected_visual_choice is not None
                and selected_visual_choice.get("surface_kind") == "campaign_set_metric_comparison"
            ):
                comparison_group_options = list(selected_visual_choice.get("comparison_group_options") or [])
            else:
                comparison_group_options = []
            if comparison_group_options:
                comparison_group_key = str(comparison_group_options[0])
                if len(comparison_group_options) > 1:
                    comparison_group_ui = mo.ui.dropdown(
                        comparison_group_options,
                        value=comparison_group_key,
                        label="Compare by",
                    )
                else:
                    comparison_group_ui = None
            else:
                comparison_group_key = None
                comparison_group_ui = None
            return comparison_group_key, comparison_group_options, comparison_group_ui
        """
    )


__all__ = ["render_visual_cells"]
