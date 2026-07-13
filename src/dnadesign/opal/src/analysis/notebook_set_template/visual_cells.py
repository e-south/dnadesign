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
from .visual_selector_cells import render_visual_selector_cells


def render_visual_cells() -> str:
    """Render campaign and campaign-set visual controls."""

    return "\n".join(
        (
            _campaign_visual_model_cell(),
            _visual_choices_cell(),
            render_visual_selector_cells(),
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
            selected_selection_view_id,
        ):
            campaign_visual_surface_model = build_notebook_visual_surface_model(
                selected_campaign_model,
                selection_view_id=selected_selection_view_id,
            )
            campaign_plot_choices = campaign_visual_surface_model["choices"]
            return campaign_plot_choices
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
            build_notebook_selection_batch_choice,
            campaign_plot_choices,
            collection_visuals,
            reader_evidence_surface,
            selected_baserender_ids,
            selected_campaign_baserender_contract, selected_campaign_model,
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
                _selection_batch_choice = build_notebook_selection_batch_choice(
                    selected_campaign_model.get("selection_batch")
                )
                if _selection_batch_choice is not None:
                    visual_choices.append(_selection_batch_choice)
            visual_choices = annotate_notebook_visual_choices(visual_choices)
            return visual_choices
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
                in {"baserender", "campaign_set_baserender", "reader_evidence", "selection_batch"}
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
