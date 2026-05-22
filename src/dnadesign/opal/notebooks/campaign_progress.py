import marimo

__generated_with = "0.19.4"

app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from dnadesign.opal import (
        assess_records_contract,
        build_ledger_status_table,
        build_records_preview,
        cli_handoff_lines,
        records_status_lines,
        x_provenance_status_lines,
    )
    from dnadesign.opal.notebooks.api import (
        campaign_label_from_path,
        diagnostics_to_lines,
        find_repo_root,
        list_campaign_paths,
        load_campaign_selection,
        load_parquet_cached,
    )

    return (
        Path,
        assess_records_contract,
        build_ledger_status_table,
        build_records_preview,
        campaign_label_from_path,
        cli_handoff_lines,
        diagnostics_to_lines,
        find_repo_root,
        list_campaign_paths,
        load_campaign_selection,
        load_parquet_cached,
        mo,
        pl,
        records_status_lines,
        x_provenance_status_lines,
    )


@app.cell
def _(Path, find_repo_root):
    notebook_path = Path(__file__).resolve()
    repo_root = find_repo_root(notebook_path)
    return (repo_root,)


@app.cell
def _(campaign_label_from_path, list_campaign_paths, repo_root):
    campaign_paths = list_campaign_paths(repo_root)
    campaign_labels = [campaign_label_from_path(path, repo_root) for path in campaign_paths]
    campaign_path_map = dict(zip(campaign_labels, campaign_paths))

    default_campaign_label = None
    for label in campaign_labels:
        if label.replace("\\", "/").endswith("src/dnadesign/opal/campaigns/demo"):
            default_campaign_label = label
            break
    if default_campaign_label is None and campaign_labels:
        default_campaign_label = campaign_labels[0]
    if not campaign_labels:
        campaign_labels = ["(no campaigns found)"]
        default_campaign_label = campaign_labels[0]
    return campaign_labels, campaign_path_map, default_campaign_label


@app.cell
def _(campaign_labels, default_campaign_label, mo):
    campaign_dropdown = mo.ui.dropdown(
        options=campaign_labels,
        value=default_campaign_label,
        label="Campaign config",
        full_width=True,
    )
    return (campaign_dropdown,)


@app.cell
def _(campaign_dropdown, campaign_path_map, diagnostics_to_lines, load_campaign_selection, mo, repo_root):
    campaign_label = campaign_dropdown.value
    campaign_path = campaign_path_map.get(campaign_label)
    campaign_selection = load_campaign_selection(campaign_path=campaign_path, repo_root=repo_root)
    campaign_diag_lines = diagnostics_to_lines(campaign_selection.diagnostics)
    campaign_notice = mo.md("\n".join(campaign_diag_lines)) if campaign_diag_lines else mo.md("")
    return campaign_notice, campaign_selection


@app.cell
def _(campaign_selection, load_parquet_cached, mo, pl):
    records_df = pl.DataFrame()
    records_error = ""
    records_path = campaign_selection.records_path
    if records_path is not None and records_path.exists():
        try:
            records_df = load_parquet_cached(records_path)
        except Exception as exc:
            records_error = f"Failed to read records.parquet: {exc}"
    elif records_path is None:
        records_error = "No records.parquet path resolved from campaign config."
    else:
        records_error = f"records.parquet not found: {records_path}"
    records_error_md = mo.md(records_error) if records_error else mo.md("")
    return records_df, records_error_md, records_path


@app.cell
def _(assess_records_contract, campaign_selection, records_df):
    records_report = assess_records_contract(records_df, campaign_selection.info)
    return (records_report,)


@app.cell
def _(build_records_preview, records_df, records_report):
    records_preview_df = build_records_preview(records_df, records_report)
    return (records_preview_df,)


@app.cell
def _(build_ledger_status_table, campaign_selection):
    ledger_status_df = build_ledger_status_table(campaign_selection.workdir)
    return (ledger_status_df,)


@app.cell
def _(campaign_selection, mo, records_path, records_report, records_status_lines):
    info = campaign_selection.info
    _lines = ["### Campaign contract"]
    if info is not None:
        _lines.extend(
            [
                f"- Campaign: `{info.slug}`",
                f"- X column: `{info.x_column}`",
                f"- Y column: `{info.y_column}`",
                f"- Y expected length: `{info.y_expected_length}`",
                f"- Model: `{info.model_name}`",
                f"- Objective: `{info.objective_name}`",
                f"- Selection: `{info.selection_name}`",
            ]
        )
    if campaign_selection.path is not None:
        _lines.append(f"- Config: `{campaign_selection.path}`")
    if records_path is not None:
        _lines.append(f"- Records: `{records_path}`")
    _lines.extend(records_status_lines(records_report))
    campaign_contract_md = mo.md("\n".join(_lines))
    return (campaign_contract_md,)


@app.cell
def _(mo, pl, records_df):
    record_options = ["(no records)"]
    if "id" in records_df.columns and not records_df.is_empty():
        record_options = records_df.select("id").drop_nulls().head(500).get_column("id").cast(pl.Utf8).to_list()
        if not record_options:
            record_options = ["(no records)"]
    record_selector = mo.ui.dropdown(
        options=record_options,
        value=record_options[0],
        label="Record id",
        searchable=True,
        full_width=True,
    )
    return (record_selector,)


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
            _lines.append(f"- X present: `{row.get(records_report.x_column) is not None}`")
        if records_report.label_hist_column and records_report.label_hist_column in active_record_df.columns:
            _lines.append(f"- label history present: `{row.get(records_report.label_hist_column) is not None}`")
        if sequence:
            _lines.append(f"- sequence preview: `{sequence[:120]}`")
        active_record_md = mo.md("\n".join(_lines))
    return active_record_md, active_record_table_df


@app.cell
def _(mo, records_report, x_provenance_status_lines):
    x_provenance_md = mo.md("\n".join(x_provenance_status_lines(records_report)))
    return (x_provenance_md,)


@app.cell
def _(campaign_selection, cli_handoff_lines, mo):
    config_text = str(campaign_selection.path or "<campaign.yaml>")
    cli_md = mo.md("\n".join(cli_handoff_lines(config_text)))
    return (cli_md,)


@app.cell
def _(mo):
    visual_boundary_md = mo.md(
        "\n".join(
            [
                "### Sequence visualization boundary",
                "",
                "OPAL treats sequence rendering as a producer-owned visualization contract.",
                "Producer notebooks should call their rendering systems through public APIs "
                "and own their visual contracts outside canonical OPAL campaign review.",
                "This notebook keeps OPAL focused on campaign progress, ledgers, selections, and record inspection.",
            ]
        )
    )
    return (visual_boundary_md,)


@app.cell
def _(
    active_record_md,
    active_record_table_df,
    campaign_contract_md,
    campaign_dropdown,
    campaign_notice,
    cli_md,
    ledger_status_df,
    mo,
    record_selector,
    records_error_md,
    records_preview_df,
    visual_boundary_md,
    x_provenance_md,
):
    header = mo.md(
        "\n".join(
            [
                "# OPAL Campaign Progress",
                "",
                "Read-only progress view for campaign readiness, ledgers, selected records, and handoff commands.",
            ]
        )
    )
    campaign_panel = mo.vstack([campaign_dropdown, campaign_notice, records_error_md, campaign_contract_md])
    records_panel = mo.vstack(
        [
            mo.md("### Records preview"),
            mo.ui.table(records_preview_df, page_size=10),
            record_selector,
            active_record_md,
            mo.ui.table(active_record_table_df, page_size=1),
        ]
    )
    ledger_panel = mo.vstack([mo.md("### Ledger readiness"), mo.ui.table(ledger_status_df, page_size=8), cli_md])
    context_panel = mo.vstack([x_provenance_md, visual_boundary_md])
    mo.vstack(
        [
            header,
            mo.accordion(
                {
                    "Campaign contract": campaign_panel,
                    "Records and active record": records_panel,
                    "Ledger and CLI handoff": ledger_panel,
                    "X provenance and limitations": context_panel,
                },
                multiple=True,
                lazy=True,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
