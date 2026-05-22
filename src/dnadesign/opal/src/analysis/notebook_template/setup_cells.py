from __future__ import annotations

from textwrap import dedent

SETUP_CELLS = dedent(
    """
    import marimo

    __generated_with = "__GENERATED_WITH__"

    app = marimo.App(width="medium")


    @app.cell
    def _():
        __opal_notebook_template_schema__ = "__OPAL_NOTEBOOK_TEMPLATE_SCHEMA__"
        import marimo as mo
        import polars as pl
        from pathlib import Path
        from dnadesign.opal import (
            CampaignAnalysis,
            assess_records_contract_for_schema,
            available_rounds,
            build_notebook_artifact_garden_lines,
            build_notebook_artifact_garden_rows,
            build_ledger_status_table,
            build_notebook_at_a_glance_lines,
            build_notebook_change_lines,
            build_notebook_change_rows,
            build_notebook_distrust_lines,
            build_notebook_evidence_rows,
            build_notebook_metric_definition_rows,
            build_notebook_no_run_lines,
            build_notebook_plot_card_lines,
            build_notebook_plot_gallery_model,
            build_notebook_run_options,
            build_notebook_run_summary_lines,
            build_notebook_validity_lines,
            build_notebook_view_model,
            build_records_preview,
            cli_handoff_lines,
            latest_round,
            latest_run_id,
            load_plot_config,
            parse_enabled,
            parse_tags,
            read_optional_table,
            records_status_lines,
            resolve_notebook_round_default,
            require_columns,
            table_status_lines,
            unavailable_table,
            x_provenance_status_lines,
        )
        return (
            mo,
            pl,
            Path,
            assess_records_contract_for_schema,
            build_notebook_artifact_garden_lines,
            build_notebook_artifact_garden_rows,
            build_ledger_status_table,
            build_notebook_at_a_glance_lines,
            build_notebook_change_lines,
            build_notebook_change_rows,
            build_notebook_distrust_lines,
            build_notebook_evidence_rows,
            build_notebook_metric_definition_rows,
            build_notebook_no_run_lines,
            build_notebook_plot_card_lines,
            build_notebook_plot_gallery_model,
            build_notebook_run_options,
            build_notebook_run_summary_lines,
            build_notebook_validity_lines,
            build_notebook_view_model,
            build_records_preview,
            cli_handoff_lines,
            read_optional_table,
            records_status_lines,
            resolve_notebook_round_default,
            table_status_lines,
            unavailable_table,
            CampaignAnalysis,
            available_rounds,
            latest_round,
            latest_run_id,
            require_columns,
            load_plot_config,
            parse_enabled,
            parse_tags,
            x_provenance_status_lines,
        )


    @app.cell
    def _(Path):
        config_path = Path(__CONFIG_PATH__)
        default_round = __DEFAULT_ROUND__
        return config_path, default_round


    @app.cell
    def _(CampaignAnalysis, build_notebook_view_model, config_path, default_round):
        campaign = CampaignAnalysis.from_config_path(config_path, allow_dir=True)
        notebook_view_model = build_notebook_view_model(config_path, round_selector=default_round)
        return campaign, notebook_view_model
    """
).strip("\n")
