from __future__ import annotations

from textwrap import dedent

LAYOUT_CELLS = dedent(
    """
    @app.cell
    def _(
        active_record_md,
        active_record_table_df,
        artifact_garden_panel,
        at_a_glance_md,
        campaign_contract_md,
        changes_panel,
        cli_handoff_md,
        data_source_ui,
        data_status_md,
        data_table,
        distrust_md,
        evidence_panel,
        header_md,
        ledger_status_df,
        metric_definitions_panel,
        mo,
        plot_panel,
        record_selector,
        records_preview_df,
        round_run_controls,
        run_summary_md,
        validity_md,
        x_provenance_md,
    ):
        round_run_panel = mo.vstack([round_run_controls, run_summary_md])
        ledger_panel = mo.vstack(
            [
                mo.ui.table(ledger_status_df, page_size=8),
                cli_handoff_md,
            ]
        )
        records_panel = mo.vstack(
            [
                mo.ui.table(records_preview_df, page_size=10),
                record_selector,
                active_record_md,
                mo.ui.table(active_record_table_df, page_size=1),
            ]
        )
        data_panel = mo.vstack(
            [
                data_status_md,
                data_source_ui,
                data_table,
            ]
        )
        distrust_panel = mo.vstack([distrust_md, evidence_panel])
        mo.vstack(
            [
                header_md,
                at_a_glance_md,
                plot_panel,
                mo.accordion(
                    {
                        "Campaign contract": campaign_contract_md,
                        "Round and run": round_run_panel,
                        "Validity": validity_md,
                        "Changes": changes_panel,
                        "Ledger readiness": ledger_panel,
                        "Records and active record": records_panel,
                        "Labels and predictions": data_panel,
                        "Metric definitions": metric_definitions_panel,
                        "Artifacts": artifact_garden_panel,
                        "X provenance and limitations": x_provenance_md,
                        "Distrust and limitations": distrust_panel,
                    },
                    multiple=True,
                    lazy=True,
                ),
            ]
        )
        return


    if __name__ == "__main__":
        app.run()
    """
).strip("\n")
