from __future__ import annotations

from ._support import block


def render_setup_cells() -> str:
    """Render campaign-set imports and view-model setup cells."""

    return "\n\n".join((_preamble(), _import_cell(), _view_model_cell()))


def _preamble() -> str:
    return block(
        """
        import marimo

        __generated_with = "__GENERATED_WITH__"

        app = marimo.App(width="full")
        """
    )


def _import_cell() -> str:
    return block(
        """
        @app.cell
        def _():
            __opal_notebook_template_schema__ = "__OPAL_NOTEBOOK_TEMPLATE_SCHEMA__"
            generated_with = "__GENERATED_WITH__"
            from pathlib import Path

            import marimo as mo
            import polars as pl

            def opal_table(data, *, page_size):
                return mo.ui.table(data, page_size=page_size, show_column_summaries=False)

            from dnadesign.opal.notebooks.api.generated import (
                build_campaign_set_notebook_view_model,
                build_notebook_artifact_garden_rows,
                build_notebook_artifact_garden_summary_rows,
                build_notebook_at_a_glance_rows,
                build_notebook_campaign_header_lines,
                build_notebook_campaign_summary_row,
                build_notebook_change_rows,
                build_notebook_change_summary_rows,
                build_notebook_collection_set_choices,
                build_notebook_collection_visual_card_rows,
                build_notebook_collection_visual_choices,
                build_notebook_evidence_rows,
                build_notebook_metric_definition_rows,
                build_notebook_no_plot_scope_rows,
                build_notebook_plot_card_rows,
                build_notebook_plot_inventory_rows,
                build_notebook_plot_method_sections,
                build_notebook_plot_scope_options,
                build_notebook_visual_surface_model,
                build_notebook_validity_rows,
                select_notebook_plot_scope,
            )
            return (
                Path,
                build_campaign_set_notebook_view_model,
                build_notebook_artifact_garden_rows,
                build_notebook_artifact_garden_summary_rows,
                build_notebook_at_a_glance_rows,
                build_notebook_campaign_header_lines,
                build_notebook_campaign_summary_row,
                build_notebook_change_rows,
                build_notebook_change_summary_rows,
                build_notebook_collection_set_choices,
                build_notebook_collection_visual_card_rows,
                build_notebook_collection_visual_choices,
                build_notebook_evidence_rows,
                build_notebook_metric_definition_rows,
                build_notebook_no_plot_scope_rows,
                build_notebook_plot_card_rows,
                build_notebook_plot_inventory_rows,
                build_notebook_plot_method_sections,
                build_notebook_plot_scope_options,
                build_notebook_visual_surface_model,
                build_notebook_validity_rows,
                generated_with,
                mo,
                opal_table,
                pl,
                select_notebook_plot_scope,
            )
        """
    )


def _view_model_cell() -> str:
    return block(
        """
        @app.cell
        def _(Path):
            config_paths = [Path(path) for path in __CONFIG_PATHS__]
            collection_manifest_path = __COLLECTION_MANIFEST_PATH__
            collection_visual_index_path = __COLLECTION_VISUAL_INDEX_PATH__
            return collection_manifest_path, collection_visual_index_path, config_paths


        @app.cell
        def _(
            build_campaign_set_notebook_view_model,
            collection_manifest_path,
            collection_visual_index_path,
            config_paths,
        ):
            selected_round_selector = __DEFAULT_ROUND__
            campaign_set_view_model = build_campaign_set_notebook_view_model(
                config_paths,
                round_selector=selected_round_selector,
                run_id=__DEFAULT_RUN_ID__,
                collection_manifest_path=collection_manifest_path,
                collection_visual_index_path=collection_visual_index_path,
            )
            campaigns = campaign_set_view_model["campaigns"]
            collection = campaign_set_view_model.get("collection")
            collection_visuals = campaign_set_view_model.get("collection_visuals") or []
            return campaign_set_view_model, campaigns, collection, collection_visuals, selected_round_selector
        """
    )


__all__ = ["render_setup_cells"]
