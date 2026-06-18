"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_template/summary_cells.py

Notebook template builders for summary cells OPAL analysis notebook template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from textwrap import dedent

SUMMARY_CELLS = dedent(
    """
    @app.cell
    def _(
        build_notebook_at_a_glance_rows,
        build_notebook_distrust_rows,
        build_notebook_status_line,
        build_notebook_trust_rows,
        build_notebook_validity_rows,
        mo,
        notebook_view_model,
        pl,
    ):
        at_a_glance_rows = build_notebook_at_a_glance_rows(notebook_view_model)
        status_line_md = mo.md(build_notebook_status_line(notebook_view_model))
        at_a_glance_md = mo.accordion(
            {
                "Status": mo.ui.table(pl.DataFrame(at_a_glance_rows), page_size=14),
                "Trust checks": mo.ui.table(
                    pl.DataFrame(build_notebook_trust_rows(notebook_view_model)),
                    page_size=8,
                ),
            },
            lazy=True,
        )
        distrust_md = mo.ui.table(
            pl.DataFrame(build_notebook_distrust_rows(notebook_view_model)),
            page_size=8,
        )
        validity_md = mo.ui.table(
            pl.DataFrame(build_notebook_validity_rows(notebook_view_model)),
            page_size=14,
        )
        return at_a_glance_md, distrust_md, status_line_md, validity_md


    @app.cell
    def _(
        build_notebook_change_rows,
        build_notebook_change_summary_rows,
        build_notebook_evidence_rows,
        mo,
        notebook_view_model,
        pl,
    ):
        evidence_rows = build_notebook_evidence_rows(notebook_view_model)
        if evidence_rows:
            evidence_panel = mo.ui.table(pl.DataFrame(evidence_rows), page_size=10)
        else:
            evidence_panel = mo.md("No warnings or stale artifacts reported for this campaign.")
        change_rows = build_notebook_change_rows(notebook_view_model)
        if change_rows:
            changes_table = mo.ui.table(pl.DataFrame(change_rows), page_size=10)
        else:
            changes_table = mo.md("No round changes are available yet.")
        changes_panel = mo.vstack(
            [
                mo.ui.table(
                    pl.DataFrame(build_notebook_change_summary_rows(notebook_view_model)),
                    page_size=8,
                ),
                changes_table,
            ]
        )
        return changes_panel, evidence_panel


    @app.cell
    def _(
        build_notebook_artifact_garden_rows,
        build_notebook_artifact_garden_summary_rows,
        build_notebook_metric_definition_rows,
        mo,
        notebook_view_model,
        pl,
    ):
        metric_definition_rows = build_notebook_metric_definition_rows(notebook_view_model)
        if metric_definition_rows:
            metric_definitions_panel = mo.ui.table(pl.DataFrame(metric_definition_rows), page_size=10)
        else:
            metric_definitions_panel = mo.md("No plot metric definitions are available.")

        artifact_garden_rows = build_notebook_artifact_garden_rows(notebook_view_model)
        artifact_summary_rows = build_notebook_artifact_garden_summary_rows(notebook_view_model)
        artifact_rows_panel = (
            mo.ui.table(pl.DataFrame(artifact_garden_rows), page_size=10)
            if artifact_garden_rows
            else mo.md("No artifact garden rows are available.")
        )
        artifact_garden_panel = mo.vstack(
            [
                mo.ui.table(pl.DataFrame(artifact_summary_rows), page_size=10),
                artifact_rows_panel,
            ]
        )
        return artifact_garden_panel, metric_definitions_panel


    @app.cell
    def _(build_notebook_campaign_header_lines, campaign, mo, notebook_view_model):
        cfg = campaign.config
        ws = campaign.workspace
        store = campaign.records_store()
        header_md = mo.md("\\n".join(build_notebook_campaign_header_lines(notebook_view_model)))
        return cfg, header_md, store, ws


    @app.cell
    def _(pl, store):
        records_schema_columns = store.schema_columns()
        records_row_count = store.row_count()
        records_loaded_columns = [
            column
            for column in (
                "id",
                "bio_type",
                "sequence",
                "alphabet",
                store.y_col,
                store.label_hist_col(),
            )
            if column in records_schema_columns
        ]
        records_df = pl.from_pandas(store.load_columns(records_loaded_columns))
        return records_df, records_loaded_columns, records_row_count, records_schema_columns


    @app.cell
    def _(
        assess_records_contract_for_schema,
        cfg,
        records_loaded_columns,
        records_row_count,
        records_schema_columns,
    ):
        records_report = assess_records_contract_for_schema(
            row_count=records_row_count,
            schema_columns=records_schema_columns,
            campaign_slug=cfg.campaign.slug,
            x_column=cfg.data.x_column_name,
            loaded_columns=records_loaded_columns,
        )
        return records_report


    @app.cell
    def _(build_records_preview, records_df, records_report):
        records_preview_df = build_records_preview(records_df, records_report)
        return records_preview_df


    @app.cell
    def _(build_ledger_status_table, ws):
        ledger_status_df = build_ledger_status_table(ws.workdir)
        return ledger_status_df


    @app.cell
    def _(campaign, read_optional_table):
        runs_read = read_optional_table(
            "runs",
            campaign.workspace.ledger_runs_path,
            campaign.read_runs,
        )
        runs_df = runs_read.df
        return runs_df, runs_read
    """
).strip("\n")
