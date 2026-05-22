from __future__ import annotations

from textwrap import dedent

RECORD_CELLS = dedent(
    """
    @app.cell
    def _(compact_notebook_path, config_path, mo, cfg, pl, records_report, store, ws):
        contract_rows = [
            {"field": "campaign", "value": cfg.campaign.name},
            {"field": "slug", "value": cfg.campaign.slug},
            {"field": "config", "value": compact_notebook_path(config_path, base=ws.workdir)},
            {"field": "workspace", "value": compact_notebook_path(ws.workdir, max_parts=1)},
            {"field": "records", "value": compact_notebook_path(store.records_path, base=ws.workdir)},
            {"field": "X column", "value": cfg.data.x_column_name},
            {"field": "Y column", "value": cfg.data.y_column_name},
            {"field": "Y expected length", "value": cfg.data.y_expected_length},
            {"field": "model", "value": cfg.model.name},
            {"field": "selection", "value": cfg.selection.selection.name},
            {"field": "rows", "value": records_report.row_count},
            {"field": "columns", "value": records_report.column_count},
            {"field": "records contract", "value": "ready" if records_report.ready else "missing required columns"},
            {"field": "X values loaded", "value": "yes" if records_report.x_values_loaded else "no"},
        ]
        if cfg.objectives.objectives:
            contract_rows.append({"field": "objective", "value": cfg.objectives.objectives[0].name})
        campaign_contract_md = mo.ui.table(pl.DataFrame(contract_rows), page_size=16)
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
            _rows = [
                {"field": "id", "value": row.get("id")},
                {"field": "sequence length", "value": len(sequence)},
            ]
            if records_report.x_column:
                if records_report.x_column in active_record_df.columns:
                    _rows.append({"field": "X present", "value": row.get(records_report.x_column) is not None})
                else:
                    _rows.append({"field": "X preview", "value": "not loaded"})
            if records_report.label_hist_column and records_report.label_hist_column in active_record_df.columns:
                _rows.append(
                    {"field": "label history present", "value": row.get(records_report.label_hist_column) is not None}
                )
            if sequence:
                _rows.append({"field": "sequence preview", "value": sequence[:120]})
            active_record_md = mo.ui.table(pl.DataFrame(_rows), page_size=8)
        return active_record_md, active_record_table_df


    @app.cell
    def _(build_notebook_baserender_contract, records_schema_columns, store):
        notebook_baserender_contract = build_notebook_baserender_contract(
            records_schema_columns,
            records_path=str(store.records_path),
        )
        return notebook_baserender_contract


    @app.cell
    def _(notebook_baserender_contract, pl, record_selector, store):
        baserender_record_row = None
        if (
            notebook_baserender_contract.get("available")
            and record_selector.value != "(no records)"
        ):
            required_columns = [
                str(column)
                for column in notebook_baserender_contract.get("required_columns", [])
            ]
            row_df = (
                pl.scan_parquet(str(store.records_path))
                .filter(pl.col("id").cast(pl.Utf8) == str(record_selector.value))
                .select(required_columns)
                .limit(1)
                .collect()
            )
            if not row_df.is_empty():
                baserender_record_row = row_df.to_dicts()[0]
        return baserender_record_row
    """
).strip("\n")
