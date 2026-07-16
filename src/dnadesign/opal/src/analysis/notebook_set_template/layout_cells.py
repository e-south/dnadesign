"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/layout_cells.py

Notebook-set template builders for final layout cells.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# ruff: noqa: E501

from __future__ import annotations

from ._support import block


def render_layout_cells() -> str:
    """Render the final notebook layout and app entrypoint cells."""

    return "\n\n".join((_layout_cell(), _main_cell()))


def _layout_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            active_view_mode,
            artifact_garden_panel,
            baserender_diagnostic_panel,
            baserender_record_selector,
            baserender_role_ui,
            baserender_selection_view_ui,
            baserender_round_ui,
            baserender_run_ui,
            campaign_summary_df,
            collection_set_ui,
            campaign_ui,
            changes_panel,
            collection_visuals,
            evidence_panel,
            label_staging_panel,
            metric_definitions_panel,
            mo,
            opal_table,
            plot_panel,
            layered_scatter_controls,
            plot_scope_ui,
            plot_ui,
            reader_evidence_artifact_ui,
            reader_evidence_panel,
            reader_evidence_time_ui,
            render_notebook_review_control_surface,
            selected_campaign_context_panel,
            selected_campaign_title_md,
            selected_collection_set_title_md,
            selected_visual_choice,
            selected_overview_panel,
            selection_view_ui,
            selected_validity_md,
            visual_group_ui,
            view_mode_ui,
        ):
            _items = [
                selected_collection_set_title_md
                if active_view_mode == "Campaign set"
                else selected_campaign_title_md
            ]
            review_control_surface = render_notebook_review_control_surface(
                active_view_mode=active_view_mode,
                baserender_record_selector=baserender_record_selector,
                baserender_role_ui=baserender_role_ui,
                baserender_selection_view_ui=baserender_selection_view_ui,
                baserender_round_ui=baserender_round_ui,
                baserender_run_ui=baserender_run_ui,
                campaign_ui=campaign_ui,
                layered_scatter_controls=layered_scatter_controls,
                selection_view_ui=selection_view_ui,
                collection_set_ui=collection_set_ui,
                mo=mo,
                plot_scope_ui=plot_scope_ui,
                plot_ui=plot_ui,
                reader_evidence_artifact_ui=reader_evidence_artifact_ui,
                reader_evidence_time_ui=reader_evidence_time_ui,
                selected_visual_choice=selected_visual_choice,
                visual_group_ui=visual_group_ui,
                view_mode_ui=view_mode_ui,
            )
            if review_control_surface is not None:
                _items.append(review_control_surface)
            if active_view_mode != "Campaign set":
                _items.append(selected_campaign_context_panel)
            _items.append(plot_panel)
            if baserender_diagnostic_panel is not None:
                _items.append(baserender_diagnostic_panel)
            _campaign_inventory_label = "Raw campaign inventory" if collection_visuals else "Campaigns at a glance"
            _accordion_items = {
                _campaign_inventory_label: opal_table(campaign_summary_df, page_size=12),
            }
            if active_view_mode != "Campaign set":
                _status_panel = mo.vstack(
                    [selected_overview_panel, selected_validity_md, changes_panel, evidence_panel],
                    gap=0.35,
                )
                _data_panel = mo.vstack(
                    [reader_evidence_panel, label_staging_panel, metric_definitions_panel, artifact_garden_panel],
                    gap=0.35,
                )
                _accordion_items.update(
                    {
                        "Campaign status": _status_panel,
                        "Data and evidence records": _data_panel,
                    }
                )
            _items.append(mo.accordion(_accordion_items, multiple=True))
            mo.vstack(_items, gap=0.35)
        """
    )


def _main_cell() -> str:
    return block(
        """
        if __name__ == "__main__":
            app.run()
        """
    )


__all__ = ["render_layout_cells"]
