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
        # ABOUTME: Marimo notebook for OPAL campaign analysis workflows.
        # ABOUTME: Loads records, labels, and predictions for interactive inspection.
        import marimo

        __generated_with = "__GENERATED_WITH__"

        app = marimo.App(width="medium")


        @app.cell
        def _():
            import marimo as mo
            import polars as pl
            from pathlib import Path
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
                f"Campaign `{cfg.campaign.name}` (slug `{cfg.campaign.slug}`) uses config `{config_path}`.",
                f"Workdir: `{ws.workdir}`",
                f"Records: `{store.records_path}`",
                f"X column: `{cfg.data.x_column_name}`; Y column: `{cfg.data.y_column_name}`",
            ]
            summary = "\\n".join(summary_lines)
            mo.md(summary)
            return cfg, store


        @app.cell
        def _(pl, store):
            records_df = pl.from_pandas(store.load())
            return records_df


        @app.cell
        def _(campaign):
            runs_df = campaign.read_runs()
            return runs_df


        @app.cell
        def _(mo, available_rounds, latest_round, runs_df, default_round):
            rounds = available_rounds(runs_df)
            mo.stop(len(rounds) == 0, mo.md("No runs available. Run `opal run ...` first."))
            if str(default_round).strip().lower() in ("latest", ""):
                round_default = latest_round(runs_df)
            else:
                round_default = int(default_round)
                if round_default not in rounds:
                    raise ValueError(
                        f"default round {round_default} not in available rounds: {rounds}"
                    )
            round_ui = mo.ui.dropdown(rounds, value=round_default, label="Round")
            return rounds, round_default, round_ui


        @app.cell
        def _(mo, pl, latest_run_id, runs_df, round_ui):
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
            return selected_round, runs_for_round, run_options, run_default, run_ui


        @app.cell
        def _(mo, round_ui, run_ui):
            mo.vstack([round_ui, run_ui])
            return


        @app.cell
        def _(pl, runs_for_round, run_ui):
            run_id = str(run_ui.value)
            run_row = runs_for_round.filter(pl.col("run_id") == run_id)
            if run_row.is_empty():
                raise ValueError(f"Run id not found: {run_id}")
            run_meta = run_row.to_dicts()[0]
            objective_name = str(run_meta.get("objective__name") or "")
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
            run_summary = "\\n".join(run_summary_lines)
            return run_id, run_meta, objective_name, run_summary


        @app.cell
        def _(mo, run_summary):
            mo.md(run_summary)
            return


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
                lines = [
                    "### Plot gallery (outputs/plots)",
                    "",
                    f"No plot outputs found in `{plots_dir}`.",
                    "Run `uv run opal plot -c <campaign.yaml>` to generate plots.",
                ]
                lines.append(filter_note)
                if missing_outputs:
                    lines.append(
                        f"Configured plots without outputs: {', '.join(missing_outputs)}"
                    )
                plot_gallery_note = "\\n".join(lines)
            else:
                labels = [plot_choice["label"] for plot_choice in plot_choices]
                plot_ui = mo.ui.dropdown(labels, value=labels[0], label="Plot")
                plot_gallery_note = "### Plot gallery (outputs/plots)\\n\\n" + filter_note
            return plot_ui, plot_gallery_note


        @app.cell
        def _(mo, plot_choices, plot_gallery_note, plot_ui):
            if plot_ui is None:
                panel = mo.md(plot_gallery_note)
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
                panel = mo.vstack(
                    [
                        mo.md("\\n".join(details)),
                        plot_ui,
                        mo.image(choice["path"].read_bytes()),
                    ]
                )
            panel
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
        def _(campaign, pred_columns, pred_required, require_columns, run_id, selected_round):
            labels_df = campaign.read_labels()
            pred_df = campaign.read_predictions(
                columns=pred_columns,
                round_selector=[selected_round],
                run_id=run_id,
                allow_missing=True,
            )
            require_columns(
                pred_df,
                pred_required,
                ctx="predictions",
            )
            if pred_df.is_empty():
                raise ValueError("No predictions found for selected run/round.")
            return labels_df, pred_df


        @app.cell
        def _(mo):
            data_source_ui = mo.ui.dropdown(
                options=[
                    "predictions (selected run)",
                    "predictions (all rounds)",
                    "labels (selected round)",
                    "labels (all rounds)",
                    "records",
                ],
                value="predictions (selected run)",
                label="Data source",
            )
            return data_source_ui


        @app.cell
        def _(
            campaign,
            data_source_ui,
            labels_df,
            mo,
            pl,
            pred_columns,
            pred_df,
            pred_required,
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
                if "as_of_round" not in labels_df.columns:
                    raise ValueError("Labels do not include as_of_round for round filtering.")
                data_df = labels_df.filter(pl.col("as_of_round") == selected_round)
            elif source == "predictions (selected run)":
                data_df = pred_df
            elif source == "predictions (all rounds)":
                data_df = campaign.read_predictions(
                    columns=pred_columns,
                    round_selector="all",
                    allow_missing=True,
                    require_run_id=False,
                )
                require_columns(data_df, pred_required, ctx="predictions")
            else:
                raise ValueError(f"Unknown data source: {source}")
            if data_df.is_empty():
                raise ValueError(f"Selected data source '{source}' returned no rows.")
            data_table = mo.ui.table(data_df, page_size=10)
            return data_df, data_table


        @app.cell
        def _(data_source_ui, data_table, mo):
            mo.vstack([data_source_ui, data_table])
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
