from __future__ import annotations

from textwrap import dedent

DATA_CELLS = dedent(
    """
    @app.cell
    def _(campaign):
        pred_core_columns = [
            "id",
            "sequence",
            "as_of_round",
            "run_id",
            "pred__score_selected",
            "sel__rank_competition",
            "sel__is_selected",
        ]
        try:
            prediction_schema_columns = list(campaign.scan_predictions().collect_schema().keys())
        except Exception:
            prediction_schema_columns = []
        pred_extra_columns = [
            column
            for column in prediction_schema_columns
            if column.startswith(("obj__", "pred__", "sel__")) and column not in pred_core_columns
        ][:12]
        pred_columns = [*pred_core_columns, *pred_extra_columns]
        return pred_columns


    @app.cell
    def _():
        pred_required = [
            "id",
            "run_id",
            "as_of_round",
            "pred__score_selected",
            "sel__rank_competition",
            "sel__is_selected",
        ]
        return pred_required


    @app.cell
    def _(
        campaign,
        pred_columns,
        pred_required,
        read_optional_table,
        require_columns,
        run_id,
        selected_round,
        unavailable_table,
    ):
        labels_read = read_optional_table(
            "labels",
            campaign.workspace.ledger_labels_path,
            campaign.read_labels,
        )
        labels_df = labels_read.df

        if selected_round is None or run_id is None:
            pred_read = unavailable_table(
                "predictions",
                campaign.workspace.ledger_predictions_dir,
                "No run selected; predictions are available after `opal run`.",
            )
        else:
            def _load_predictions():
                df = campaign.read_predictions(
                    columns=pred_columns,
                    round_selector=[selected_round],
                    run_id=run_id,
                    allow_missing=True,
                )
                require_columns(
                    df,
                    pred_required,
                    ctx="predictions",
                )
                if df.is_empty():
                    raise ValueError("No predictions found for selected run/round.")
                return df

            pred_read = read_optional_table(
                "predictions",
                campaign.workspace.ledger_predictions_dir,
                _load_predictions,
            )
        pred_df = pred_read.df
        return labels_df, labels_read, pred_df, pred_read


    @app.cell
    def _(labels_read, mo, pred_read, selected_round):
        data_source_options = ["records"]
        default_source = "records"
        if labels_read.available:
            if selected_round is not None:
                data_source_options.append("labels (selected round)")
            data_source_options.append("labels (all rounds)")
        if pred_read.available:
            data_source_options = [
                "predictions (selected run)",
                "predictions (all rounds)",
                *data_source_options,
            ]
            default_source = "predictions (selected run)"
        data_source_ui = mo.ui.dropdown(
            options=data_source_options,
            value=default_source,
            label="Data source",
        )
        return data_source_ui, default_source


    @app.cell
    def _(
        campaign,
        data_source_ui,
        labels_df,
        labels_read,
        mo,
        pl,
        pred_columns,
        pred_df,
        pred_read,
        pred_required,
        read_optional_table,
        records_df,
        require_columns,
        selected_round,
        table_status_lines,
    ):
        source = str(data_source_ui.value)
        if source == "records":
            data_df = records_df
        elif source == "labels (all rounds)":
            data_df = labels_df
        elif source == "labels (selected round)":
            if "as_of_round" not in labels_df.columns:
                raise ValueError("Labels do not include as_of_round for round filtering.")
            data_df = labels_df.filter(pl.col("as_of_round") == selected_round)
        elif source == "predictions (selected run)":
            data_df = pred_df
        elif source == "predictions (all rounds)":
            all_pred_read = read_optional_table(
                "predictions_all_rounds",
                campaign.workspace.ledger_predictions_dir,
                lambda: campaign.read_predictions(
                    columns=pred_columns,
                    round_selector="all",
                    allow_missing=True,
                    require_run_id=False,
                ),
            )
            if all_pred_read.available:
                data_df = all_pred_read.df
                require_columns(data_df, pred_required, ctx="predictions")
            else:
                data_df = records_df
        else:
            raise ValueError(f"Unknown data source: {source}")
        data_status_lines = [
            f"- Selected data source: `{source}`",
            "",
            *table_status_lines(labels_read),
            "",
            *table_status_lines(pred_read),
        ]
        if data_df.is_empty():
            data_status_lines.append(
                f"- Selected data source `{source}` returned no rows."
            )
        data_table = mo.ui.table(data_df, page_size=10)
        data_status_md = mo.md("\\n".join(data_status_lines))
        return data_df, data_status_md, data_table
    """
).strip("\n")
