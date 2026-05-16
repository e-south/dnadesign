"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/notebook_template.py

Renders marimo notebook templates for OPAL campaigns. Generates scaffolded
notebooks with campaign context and data previews.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def render_campaign_notebook(config_path: Path, *, round_selector: str) -> str:
    """
    Render a marimo notebook template tied to a campaign.
    """
    try:
        import marimo as _marimo
    except Exception:
        _marimo = None
    if _marimo is None:
        marimo_version = "unknown"
    else:
        marimo_version = getattr(_marimo, "__version__", "unknown")

    template = dedent(
        """
        import marimo

        __generated_with = "__GENERATED_WITH__"

        app = marimo.App(width="medium")


        @app.cell
        def _():
            import marimo as mo
            import polars as pl
            from pathlib import Path
            from dnadesign.opal.src.analysis.campaign_progress import (
                assess_records_contract_for_values,
                build_ledger_status_table,
                build_records_preview,
                cli_handoff_lines,
                projection_status_lines,
                read_optional_table,
                records_status_lines,
                table_status_lines,
                unavailable_table,
            )
            from dnadesign.opal.src.analysis.facade import (
                CampaignAnalysis,
                available_rounds,
                latest_round,
                latest_run_id,
                require_columns,
            )
            from dnadesign.opal.src.plots.config import load_plot_config, parse_enabled, parse_tags
            return (
                mo,
                pl,
                Path,
                assess_records_contract_for_values,
                build_ledger_status_table,
                build_records_preview,
                cli_handoff_lines,
                projection_status_lines,
                read_optional_table,
                records_status_lines,
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
            )


        @app.cell
        def _(Path):
            config_path = Path(__CONFIG_PATH__)
            default_round = __DEFAULT_ROUND__
            return config_path, default_round


        @app.cell
        def _(CampaignAnalysis, config_path):
            campaign = CampaignAnalysis.from_config_path(config_path, allow_dir=True)
            return campaign


        @app.cell
        def _(campaign, config_path, mo):
            cfg = campaign.config
            ws = campaign.workspace
            store = campaign.records_store()
            summary_lines = [
                "# OPAL Campaign Notebook",
                "",
                "Campaign-specific artifact viewer for records, ledgers, selected records, and plot deliverables.",
            ]
            summary = "\\n".join(summary_lines)
            header_md = mo.md(summary)
            return cfg, header_md, store, ws


        @app.cell
        def _(pl, store):
            records_df = pl.from_pandas(store.load())
            return records_df


        @app.cell
        def _(assess_records_contract_for_values, cfg, records_df):
            records_report = assess_records_contract_for_values(
                records_df,
                campaign_slug=cfg.campaign.slug,
                x_column=cfg.data.x_column_name,
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


        @app.cell
        def _(
            available_rounds,
            cfg,
            default_round,
            latest_round,
            latest_run_id,
            mo,
            pl,
            runs_df,
            runs_read,
            table_status_lines,
        ):
            rounds = available_rounds(runs_df)
            selected_round = None
            run_id = None
            run_meta = {}
            runs_for_round = runs_df.head(0)
            objective_name = (
                str(cfg.objectives.objectives[0].name)
                if cfg.objectives.objectives
                else ""
            )
            if rounds:
                if str(default_round).strip().lower() in ("latest", ""):
                    round_default = latest_round(runs_df)
                else:
                    round_default = int(default_round)
                    if round_default not in rounds:
                        raise ValueError(
                            f"default round {round_default} not in available rounds: {rounds}"
                        )
                round_ui = mo.ui.dropdown(rounds, value=round_default, label="Round")
                selected_round = int(round_ui.value)
                runs_for_round = runs_df.filter(pl.col("as_of_round") == selected_round)
                if runs_for_round.is_empty():
                    raise ValueError(f"No runs found for round {selected_round}.")
                run_default = latest_run_id(runs_for_round)
                run_options = (
                    runs_for_round.select("run_id")
                    .unique()
                    .sort("run_id")["run_id"]
                    .to_list()
                )
                run_ui = mo.ui.dropdown(run_options, value=run_default, label="Run ID")
                run_id = str(run_ui.value)
                run_row = runs_for_round.filter(pl.col("run_id") == run_id)
                if run_row.is_empty():
                    raise ValueError(f"Run id not found: {run_id}")
                run_meta = run_row.to_dicts()[0]
                objective_name = str(run_meta.get("objective__name") or objective_name)
                run_summary_lines = [
                    "## Run Summary",
                    "",
                    (
                        f"Run `{run_id}` (round {run_meta.get('as_of_round', -1)}) uses "
                        f"objective `{objective_name}` and selection `{run_meta.get('selection__name')}`."
                    ),
                    f"Model: `{run_meta.get('model__name')}`",
                    f"Train size: {run_meta.get('stats__n_train')} | Scored: {run_meta.get('stats__n_scored')}",
                ]
                round_run_controls = mo.vstack([round_ui, run_ui])
                run_summary_md = mo.md("\\n".join(run_summary_lines))
            else:
                no_run_lines = [
                    "### Round and run",
                    "",
                    "No runs available yet.",
                    "The campaign contract and records remain inspectable before the first OPAL run.",
                    "Expected runs ledger: `outputs/ledger/runs.parquet`.",
                    "",
                    *table_status_lines(runs_read),
                ]
                round_run_controls = mo.md("\\n".join(no_run_lines))
                run_summary_md = mo.md("")
            return (
                objective_name,
                round_run_controls,
                run_id,
                run_meta,
                run_summary_md,
                runs_for_round,
                selected_round,
            )


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
        def _(projection_status_lines, records_report, mo):
            optional_context_md = mo.md(
                "\\n".join(
                    [
                        "### Optional context boundaries",
                        "",
                        *projection_status_lines(records_report),
                        "",
                        "- Sequence rendering stays with producer-owned visualization contracts "
                        "and BaseRender public APIs.",
                        "- This notebook does not select X by UMAP position or centroid-nearest behavior.",
                    ]
                )
            )
            return optional_context_md


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
                    _lines.append(f"- X present: `{row.get(records_report.x_column) is not None}`")
                if records_report.label_hist_column and records_report.label_hist_column in active_record_df.columns:
                    _lines.append(f"- label history present: `{row.get(records_report.label_hist_column) is not None}`")
                if sequence:
                    _lines.append(f"- sequence preview: `{sequence[:120]}`")
                active_record_md = mo.md("\\n".join(_lines))
            return active_record_md, active_record_table_df


        @app.cell
        def _(campaign, config_path, load_plot_config):
            plot_cfg = None
            plot_cfg_error = None
            try:
                plot_cfg = load_plot_config(
                    campaign_cfg=campaign.read_config_dict(),
                    campaign_yaml=config_path,
                    campaign_dir=campaign.workspace.workdir,
                    plot_config_opt=None,
                )
            except Exception as exc:
                plot_cfg_error = str(exc)
            return plot_cfg, plot_cfg_error


        @app.cell
        def _(objective_name, parse_enabled, parse_tags, plot_cfg):
            plot_entries = []
            if plot_cfg is not None:
                for plot_entry_item in plot_cfg.plots:
                    if not isinstance(plot_entry_item, dict):
                        raise ValueError(
                            "Plot entry must be a mapping (got "
                            f"{type(plot_entry_item).__name__})."
                        )
                    name = plot_entry_item.get("name")
                    if not name:
                        raise ValueError("Plot entry missing name.")
                    preset_name = plot_entry_item.get("preset")
                    preset = plot_cfg.plot_presets.get(preset_name) if preset_name else {}
                    kind = plot_entry_item.get("kind") or preset.get("kind")
                    if not kind:
                        raise ValueError(f"Plot '{name}' missing kind.")
                    enabled = parse_enabled(
                        plot_entry_item.get("enabled")
                        if "enabled" in plot_entry_item
                        else preset.get("enabled"),
                        ctx=name,
                    )
                    if not enabled:
                        continue
                    _plot_tags_list = []
                    if preset_name:
                        _plot_tags_list += parse_tags(
                            preset.get("tags"),
                            ctx=f"plot_presets.{preset_name}",
                        )
                    _plot_tags_list += parse_tags(
                        plot_entry_item.get("tags"),
                        ctx=f"plot {name}",
                    )
                    plot_entries.append(
                        {"name": name, "kind": kind, "tags": _plot_tags_list}
                    )
            objective_is_sfxi = str(objective_name).lower().startswith("sfxi")

            def _is_sfxi_kind(kind: str) -> bool:
                return str(kind).lower().startswith("sfxi_")

            if objective_is_sfxi:
                plot_entries_filtered = [
                    plot_entry_filter
                    for plot_entry_filter in plot_entries
                    if _is_sfxi_kind(plot_entry_filter["kind"])
                ]
            else:
                plot_entries_filtered = [
                    plot_entry_filter
                    for plot_entry_filter in plot_entries
                    if not _is_sfxi_kind(plot_entry_filter["kind"])
                ]
            return objective_is_sfxi, plot_entries_filtered


        @app.cell
        def _(campaign, plot_entries_filtered):
            plots_dir = campaign.workspace.workdir / "outputs" / "plots"
            plot_files = []
            if plots_dir.exists():
                plot_files = sorted(plots_dir.glob("*.png"))

            def _latest_match(name: str):
                candidates = [path for path in plot_files if path.name.startswith(name)]
                if not candidates:
                    return None
                return max(candidates, key=lambda p: p.stat().st_mtime)

            plot_choices = []
            missing_outputs = []
            for plot_entry_choice in plot_entries_filtered:
                path = _latest_match(plot_entry_choice["name"])
                if path is None:
                    missing_outputs.append(plot_entry_choice["name"])
                    continue
                label = f"{plot_entry_choice['name']} ({path.name})"
                plot_choices.append(
                    {"label": label, "path": path, "entry": plot_entry_choice}
                )
            return plots_dir, plot_choices, missing_outputs


        @app.cell
        def _(mo, objective_is_sfxi, plot_cfg_error, plot_choices, plots_dir, missing_outputs):
            plot_ui = None
            filter_note = "SFXI plots only." if objective_is_sfxi else "Non-SFXI plots only."
            if plot_cfg_error:
                plot_gallery_note = (
                    "### Plot gallery (outputs/plots)\\n\\n"
                    f"Plot config unavailable: `{plot_cfg_error}`"
                )
            elif not plot_choices:
                _lines = [
                    "### Plot gallery (outputs/plots)",
                    "",
                    f"No plot outputs found in `{plots_dir}`.",
                    "Run `uv run opal plot -c <campaign.yaml>` to generate plots.",
                ]
                _lines.append(filter_note)
                if missing_outputs:
                    _lines.append(
                        f"Configured plots without outputs: {', '.join(missing_outputs)}"
                    )
                plot_gallery_note = "\\n".join(_lines)
            else:
                labels = [plot_choice["label"] for plot_choice in plot_choices]
                plot_ui = mo.ui.dropdown(labels, value=labels[0], label="Plot")
                plot_gallery_note = "### Plot gallery (outputs/plots)\\n\\n" + filter_note
            return plot_ui, plot_gallery_note


        @app.cell
        def _(mo, plot_choices, plot_gallery_note, plot_ui):
            if plot_ui is None:
                plot_panel = mo.md(plot_gallery_note)
            else:
                selected = str(plot_ui.value)
                choice = next(
                    (
                        plot_choice
                        for plot_choice in plot_choices
                        if plot_choice["label"] == selected
                    ),
                    None,
                )
                if choice is None:
                    raise ValueError(f"Plot selection not found: {selected}")
                plot_entry_selected = choice["entry"]
                _plot_tags_str = (
                    ", ".join(plot_entry_selected["tags"])
                    if plot_entry_selected["tags"]
                    else "none"
                )
                details = [
                    plot_gallery_note,
                    "",
                    f"**Plot**: `{plot_entry_selected['name']}`",
                    f"Kind: `{plot_entry_selected['kind']}`",
                    f"Tags: `{_plot_tags_str}`",
                    f"File: `{choice['path']}`",
                ]
                plot_panel = mo.vstack(
                    [
                        mo.md("\\n".join(details)),
                        plot_ui,
                        mo.image(choice["path"].read_bytes()),
                    ]
                )
            return plot_panel


        @app.cell
        def _():
            pred_columns = [
                "id",
                "sequence",
                "as_of_round",
                "run_id",
                "pred__score_selected",
                "sel__rank_competition",
                "sel__is_selected",
                "obj__logic_fidelity",
                "obj__effect_raw",
                "obj__effect_scaled",
            ]
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
                "### Labels and predictions",
                "",
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


        @app.cell
        def _(
            active_record_md,
            active_record_table_df,
            campaign_contract_md,
            cli_handoff_md,
            data_source_ui,
            data_status_md,
            data_table,
            header_md,
            ledger_status_df,
            mo,
            optional_context_md,
            plot_panel,
            record_selector,
            records_preview_df,
            round_run_controls,
            run_summary_md,
        ):
            round_run_panel = mo.vstack([round_run_controls, run_summary_md])
            ledger_panel = mo.vstack(
                [
                    mo.md("### Ledger readiness"),
                    mo.ui.table(ledger_status_df, page_size=8),
                    cli_handoff_md,
                ]
            )
            records_panel = mo.vstack(
                [
                    mo.md("### Records preview"),
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
            mo.vstack(
                [
                    header_md,
                    mo.accordion(
                        {
                            "Campaign contract": campaign_contract_md,
                            "Round and run": round_run_panel,
                            "Ledger readiness": ledger_panel,
                            "Records and active record": records_panel,
                            "Labels and predictions": data_panel,
                            "Plot deliverables": plot_panel,
                            "Optional context boundaries": optional_context_md,
                        },
                        multiple=True,
                    ),
                ]
            )
            return


        if __name__ == "__main__":
            app.run()
        """
    ).strip("\n")

    return (
        template.replace("__CONFIG_PATH__", repr(str(config_path)))
        .replace("__DEFAULT_ROUND__", repr(str(round_selector)))
        .replace("__GENERATED_WITH__", str(marimo_version))
        + "\n"
    )
