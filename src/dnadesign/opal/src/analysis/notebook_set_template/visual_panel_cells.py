"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/visual_panel_cells.py

Notebook-set template builders for visual panel cells OPAL analysis notebook set.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ._support import block


def render_visual_panel_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            CAMPAIGN_SET_BASERENDER_SURFACE_KIND, active_view_mode, baserender_campaign_model,
            baserender_record_id, baserender_record_row, baserender_record_selector, baserender_role_ui,
            baserender_round_ui, baserender_run_ui, build_notebook_baserender_contract_rows,
            build_notebook_baserender_label_rows, build_notebook_collection_visual_card_rows,
            collection_visual_description, build_notebook_plot_card_rows,
            build_notebook_plot_method_sections, mo, opal_table, pl, plot_scope_ui, plot_ui,
            render_notebook_baserender_record, render_notebook_plot_choice_image,
            render_notebook_visual_panel,
            selected_baserender_round, selected_baserender_status_rows, selected_campaign_baserender_contract,
            selected_campaign_labels_df, selected_visual_choice, select_notebook_plot_scope,
        ):
            plot_panel = render_notebook_visual_panel(
                active_view_mode=active_view_mode,
                baserender_campaign_model=baserender_campaign_model,
                baserender_record_id=baserender_record_id,
                baserender_record_row=baserender_record_row,
                baserender_record_selector=baserender_record_selector,
                baserender_role_ui=baserender_role_ui,
                baserender_round_ui=baserender_round_ui,
                baserender_run_ui=baserender_run_ui,
                build_notebook_baserender_contract_rows=build_notebook_baserender_contract_rows,
                build_notebook_baserender_label_rows=build_notebook_baserender_label_rows,
                build_notebook_collection_visual_card_rows=build_notebook_collection_visual_card_rows,
                build_notebook_plot_card_rows=build_notebook_plot_card_rows,
                build_notebook_plot_method_sections=build_notebook_plot_method_sections,
                campaign_set_baserender_surface_kind=CAMPAIGN_SET_BASERENDER_SURFACE_KIND,
                collection_visual_description=collection_visual_description,
                mo=mo,
                opal_table=opal_table,
                pl=pl,
                plot_scope_ui=plot_scope_ui,
                plot_ui=plot_ui,
                render_notebook_baserender_record=render_notebook_baserender_record,
                render_notebook_plot_choice_image=render_notebook_plot_choice_image,
                selected_baserender_round=selected_baserender_round,
                selected_baserender_status_rows=selected_baserender_status_rows,
                selected_campaign_baserender_contract=selected_campaign_baserender_contract,
                selected_campaign_labels_df=selected_campaign_labels_df,
                selected_visual_choice=selected_visual_choice,
                select_notebook_plot_scope=select_notebook_plot_scope,
            )
            return plot_panel
        """
    )


__all__ = ["render_visual_panel_cell"]
