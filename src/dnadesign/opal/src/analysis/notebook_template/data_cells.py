"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_template/data_cells.py

Notebook template builders for data cells OPAL analysis notebook template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

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
    def _(cfg, labels_df, labels_read, mo, pred_read, selected_round):
        label_round_column = str(cfg.labels.round_column or "observed_round")
        data_source_options = ["records"]
        default_source = "records"
        if labels_read.available:
            if selected_round is not None and label_round_column in labels_df.columns:
                data_source_options.append("labels (selected round)")
            data_source_options.append("labels (all rounds)")
        if pred_read.available:
            data_source_options = [
                "predictions (selected run)",
                *data_source_options,
            ]
            default_source = "predictions (selected run)"
        data_source_ui = mo.ui.dropdown(
            options=data_source_options,
            value=default_source,
            label="Data source",
        )
        return data_source_ui, default_source, label_round_column


    @app.cell
    def _(
        campaign,
        data_source_ui,
        labels_df,
        labels_read,
        label_round_column,
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
    ):
        source = str(data_source_ui.value)
        if source == "records":
            data_df = records_df
        elif source == "labels (all rounds)":
            data_df = labels_df
        elif source == "labels (selected round)":
            if label_round_column not in labels_df.columns:
                raise ValueError(f"Labels do not include {label_round_column} for round filtering.")
            data_df = labels_df.filter(pl.col(label_round_column) == selected_round)
        elif source == "predictions (selected run)":
            data_df = pred_df
        else:
            raise ValueError(f"Unknown data source: {source}")
        data_status_rows = [
            {"field": "selected data source", "value": source},
            {"field": "selected rows", "value": data_df.height},
            {"field": "labels status", "value": labels_read.status},
            {"field": "labels rows", "value": labels_read.df.height},
            {"field": "labels path", "value": str(labels_read.path or "")},
            {"field": "predictions status", "value": pred_read.status},
            {"field": "predictions rows", "value": pred_read.df.height},
            {"field": "predictions path", "value": str(pred_read.path or "")},
        ]
        if data_df.is_empty():
            data_status_rows.append(
                {"field": "empty source", "value": f"{source} returned no rows"}
            )
        data_table = mo.ui.table(data_df, page_size=10)
        data_status_md = mo.ui.table(pl.DataFrame(data_status_rows), page_size=10)
        return data_df, data_status_md, data_table
    """
).strip("\n")
