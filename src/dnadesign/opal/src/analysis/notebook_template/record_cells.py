from __future__ import annotations

from textwrap import dedent

RECORD_CELLS = dedent(
    """
    @app.cell
    def _(config_path, mo, cfg, records_status_lines, records_report, store, ws):
        contract_lines = [
            "### Campaign contract",
            f"- Campaign: `{cfg.campaign.name}`",
            f"- Slug: `{cfg.campaign.slug}`",
            f"- Config: `{config_path}`",
            f"- Workdir: `{ws.workdir}`",
            f"- Records: `{store.records_path}`",
            f"- X column: `{cfg.data.x_column_name}`",
            f"- Y column: `{cfg.data.y_column_name}`",
            f"- Y expected length: `{cfg.data.y_expected_length}`",
            f"- Model: `{cfg.model.name}`",
            f"- Selection: `{cfg.selection.selection.name}`",
        ]
        if cfg.objectives.objectives:
            contract_lines.append(f"- Objective: `{cfg.objectives.objectives[0].name}`")
        contract_lines.extend(records_status_lines(records_report))
        campaign_contract_md = mo.md("\\n".join(contract_lines))
        return campaign_contract_md


    @app.cell
    def _(cli_handoff_lines, config_path, mo):
        cli_handoff_md = mo.md("\\n".join(cli_handoff_lines(config_path)))
        return cli_handoff_md


    @app.cell
    def _(x_provenance_status_lines, records_report, mo):
        x_provenance_md = mo.md(
            "\\n".join(
                [
                    "### X provenance and limitations",
                    "",
                    *x_provenance_status_lines(records_report),
                    "",
                    "- OPAL review surfaces show campaign contracts, ledgers, selections, "
                    "labels, predictions, and OPAL plot artifacts.",
                    "- Producer-specific representation browsers and study benchmark reports "
                    "stay outside canonical OPAL notebooks.",
                ]
            )
        )
        return x_provenance_md


    @app.cell
    def _(mo, pl, records_df):
        record_options = ["(no records)"]
        if "id" in records_df.columns and not records_df.is_empty():
            record_options = (
                records_df.select("id")
                .drop_nulls()
                .head(500)
                .get_column("id")
                .cast(pl.Utf8)
                .to_list()
            )
            if not record_options:
                record_options = ["(no records)"]
        record_selector = mo.ui.dropdown(
            options=record_options,
            value=record_options[0],
            label="Record id",
            searchable=True,
            full_width=True,
        )
        return record_selector


    @app.cell
    def _(build_records_preview, mo, pl, record_selector, records_df, records_report):
        active_record_df = records_df.head(0)
        selected_id = record_selector.value
        if selected_id != "(no records)" and "id" in records_df.columns:
            active_record_df = records_df.filter(pl.col("id").cast(pl.Utf8) == str(selected_id)).head(1)
        active_record_table_df = build_records_preview(active_record_df, records_report, limit=1)

        if active_record_df.is_empty():
            active_record_md = mo.md("No active record selected.")
        else:
            row = active_record_df.to_dicts()[0]
            sequence = str(row.get("sequence") or "")
            _lines = [
                f"- id: `{row.get('id')}`",
                f"- sequence length: `{len(sequence)}`",
            ]
            if records_report.x_column:
                if records_report.x_column in active_record_df.columns:
                    _lines.append(f"- X present: `{row.get(records_report.x_column) is not None}`")
                else:
                    _lines.append("- X preview: `not loaded`")
            if records_report.label_hist_column and records_report.label_hist_column in active_record_df.columns:
                _lines.append(f"- label history present: `{row.get(records_report.label_hist_column) is not None}`")
            if sequence:
                _lines.append(f"- sequence preview: `{sequence[:120]}`")
            active_record_md = mo.md("\\n".join(_lines))
        return active_record_md, active_record_table_df
    """
).strip("\n")
