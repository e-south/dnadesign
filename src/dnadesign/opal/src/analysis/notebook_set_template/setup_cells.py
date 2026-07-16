"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/setup_cells.py

Notebook-set template builders for setup cells OPAL analysis notebook set template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ._support import block


def render_setup_cells() -> str:
    """Render campaign-set imports and view-model setup cells."""

    return "\n".join((_preamble(), _import_cell(), _view_model_cell()))


def _preamble() -> str:
    return block(
        """
        import marimo

        __generated_with = "__GENERATED_WITH__"

        # fmt: off
        # ruff: noqa

        app = marimo.App(width="medium")
        """
    )


def _import_cell() -> str:
    return block(
        """
        @app.cell
        def _():
            __opal_notebook_template_schema__ = "__OPAL_NOTEBOOK_TEMPLATE_SCHEMA__"  # noqa: F841
            generated_with = "__GENERATED_WITH__"
            from pathlib import Path

            import marimo as mo
            import polars as pl

            def opal_table(data, *, page_size):
                return mo.ui.table(data, page_size=page_size, show_column_summaries=False)

            from dnadesign.opal.notebooks.api.generated import (
                CAMPAIGN_SET_BASERENDER_SURFACE_KIND, CAMPAIGN_SET_SELECTION_OVERLAP_SURFACE_KIND,
                CampaignAnalysis, annotate_notebook_visual_choices, available_rounds,
                build_campaign_set_notebook_view_model, build_notebook_artifact_garden_rows,
                build_notebook_artifact_garden_summary_rows, build_notebook_at_a_glance_rows,
                build_notebook_baserender_contract, build_notebook_baserender_contract_rows,
                build_notebook_baserender_label_rows, build_notebook_baserender_record_annotation_counts,
                build_notebook_baserender_record_choices, build_notebook_baserender_record_choices_with_counts,
                build_notebook_baserender_record_options, build_notebook_campaign_header_lines,
                build_notebook_campaign_summary_row, build_notebook_change_rows, build_notebook_change_summary_rows,
                build_notebook_collection_baserender_role_choices, build_notebook_collection_set_choices,
                build_notebook_campaign_set_selection_overlap_card_rows, build_notebook_collection_visual_card_rows,
                build_notebook_collection_visual_choices, build_notebook_evidence_rows,
                build_notebook_label_staging_rows, build_notebook_metric_definition_rows, build_notebook_plot_card_rows,
                build_notebook_layered_scatter_contract,
                build_notebook_layered_scatter_controls,
                build_notebook_plot_method_sections, build_notebook_plot_scope_options,
                build_notebook_reader_evidence_visual_choices, build_notebook_run_options,
                build_notebook_selection_batch_choice,
                build_notebook_selection_batch_rows, build_notebook_selection_batch_summary_rows,
                build_notebook_selection_view_options,
                build_notebook_selected_baserender_record_ids, build_notebook_validity_rows,
                build_notebook_visual_group_options, build_notebook_visual_surface_model,
                filter_notebook_visual_choices_by_group, latest_round, latest_run_id,
                load_notebook_baserender_record_row, render_notebook_baserender_record,
                render_notebook_campaign_set_selection_overlap_image,
                render_notebook_plot_choice_image, render_notebook_reader_evidence_artifact_control,
                render_notebook_reader_evidence_artifact_visual, render_notebook_reader_evidence_panel,
                render_notebook_reader_evidence_time_control, render_notebook_review_control_surface,
                render_notebook_visual_panel, read_notebook_layered_scatter_state,
                resolve_notebook_round_default, resolve_notebook_selection_view,
                select_notebook_baserender_default_record_id,
                select_notebook_plot_scope,
            )
            from dnadesign.opal.notebooks.api.generated import (
                build_notebook_collection_visual_description as collection_visual_description,
            )
            return (
                Path, CAMPAIGN_SET_BASERENDER_SURFACE_KIND, CAMPAIGN_SET_SELECTION_OVERLAP_SURFACE_KIND,
                CampaignAnalysis, annotate_notebook_visual_choices, available_rounds,
                build_campaign_set_notebook_view_model, build_notebook_artifact_garden_rows,
                build_notebook_artifact_garden_summary_rows, build_notebook_at_a_glance_rows,
                build_notebook_baserender_contract, build_notebook_baserender_contract_rows,
                build_notebook_baserender_label_rows, build_notebook_baserender_record_annotation_counts,
                build_notebook_baserender_record_choices, build_notebook_baserender_record_choices_with_counts,
                build_notebook_baserender_record_options, build_notebook_campaign_header_lines,
                build_notebook_campaign_summary_row, build_notebook_change_rows, build_notebook_change_summary_rows,
                build_notebook_collection_baserender_role_choices, build_notebook_collection_set_choices,
                build_notebook_campaign_set_selection_overlap_card_rows, build_notebook_collection_visual_card_rows,
                build_notebook_collection_visual_choices, collection_visual_description, build_notebook_evidence_rows,
                build_notebook_label_staging_rows, build_notebook_metric_definition_rows, build_notebook_plot_card_rows,
                build_notebook_layered_scatter_contract,
                build_notebook_layered_scatter_controls,
                build_notebook_plot_method_sections, build_notebook_plot_scope_options,
                build_notebook_reader_evidence_visual_choices, build_notebook_run_options,
                build_notebook_selection_batch_choice,
                build_notebook_selection_batch_rows, build_notebook_selection_batch_summary_rows,
                build_notebook_selection_view_options,
                build_notebook_selected_baserender_record_ids, build_notebook_validity_rows,
                build_notebook_visual_group_options, build_notebook_visual_surface_model,
                filter_notebook_visual_choices_by_group, generated_with, latest_round, latest_run_id,
                load_notebook_baserender_record_row, mo, opal_table, pl, render_notebook_baserender_record,
                render_notebook_campaign_set_selection_overlap_image, render_notebook_plot_choice_image,
                render_notebook_reader_evidence_artifact_control, render_notebook_reader_evidence_artifact_visual,
                render_notebook_reader_evidence_panel, render_notebook_reader_evidence_time_control,
                render_notebook_review_control_surface, render_notebook_visual_panel,
                read_notebook_layered_scatter_state, resolve_notebook_round_default, resolve_notebook_selection_view,
                select_notebook_baserender_default_record_id,
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
