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
            import altair as alt
            from pathlib import Path
            from dnadesign.opal.src.analysis.dashboard.theme import setup_altair_theme
            from dnadesign.opal.src.analysis.facade import (
                CampaignAnalysis,
                available_rounds,
                latest_round,
                latest_run_id,
                require_columns,
            )
            setup_altair_theme()
            return (
                mo,
                pl,
                alt,
                Path,
                setup_altair_theme,
                CampaignAnalysis,
                available_rounds,
                latest_round,
                latest_run_id,
                require_columns,
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
        def _():
            pred_columns = [
                "id",
                "sequence",
                "as_of_round",
                "run_id",
                "pred__y_obj_scalar",
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
                "pred__y_obj_scalar",
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
        def _(mo, pred_df):
            plot_max = max(200, int(pred_df.height))
            plot_rows_ui = mo.ui.slider(
                200,
                plot_max,
                value=min(plot_max, 5000),
                step=200,
                label="Plot rows",
            )
            return plot_rows_ui


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


        @app.cell
        def _(mo):
            show_selected_ui = mo.ui.checkbox(label="Selected only", value=False)
            return show_selected_ui


        @app.cell
        def _(mo, pred_df):
            color_candidates = []
            for col in pred_df.columns:
                if col in ("id", "sequence"):
                    continue
                dtype = pred_df.schema.get(col)
                if dtype is None:
                    continue
                if dtype.is_numeric() or str(dtype) in ("Boolean", "String", "Categorical"):
                    color_candidates.append(col)
            default_color = (
                "sel__is_selected"
                if "sel__is_selected" in color_candidates
                else (color_candidates[0] if color_candidates else None)
            )
            mo.stop(default_color is None, mo.md("No columns available for color."))
            color_by_ui = mo.ui.dropdown(
                color_candidates, value=default_color, label="Color by"
            )
            return color_candidates, default_color, color_by_ui


        @app.cell
        def _(pl, pred_df, show_selected_ui):
            filtered_df = pred_df
            if show_selected_ui.value and "sel__is_selected" in filtered_df.columns:
                filtered_df = filtered_df.filter(pl.col("sel__is_selected") == True)
            return filtered_df


        @app.cell
        def _(mo, filtered_df):
            mo.stop(filtered_df.is_empty(), mo.md("No rows matched the current filters."))
            return


        @app.cell
        def _(mo, pred_df):
            schema = pred_df.schema
            candidates = [
                c
                for c in pred_df.columns
                if (c.startswith("pred__") or c.startswith("obj__"))
                and (schema.get(c) is not None)
                and schema.get(c).is_numeric()
            ]
            default_field = "pred__y_obj_scalar" if "pred__y_obj_scalar" in candidates else None
            if default_field is None:
                default_field = candidates[0] if candidates else None
            mo.stop(default_field is None, mo.md("No score fields available."))
            score_field_ui = mo.ui.dropdown(
                candidates, value=default_field, label="Score field"
            )
            return candidates, default_field, score_field_ui


        @app.cell
        def _(color_by_ui, mo, plot_rows_ui, score_field_ui, show_selected_ui):
            mo.vstack([score_field_ui, color_by_ui, show_selected_ui, plot_rows_ui])
            return


        @app.cell
        def _(pl, filtered_df, score_field_ui):
            score_field = str(score_field_ui.value)
            try:
                scores = filtered_df.get_column(score_field).cast(pl.Float64, strict=False)
            except Exception as exc:
                raise ValueError(f"Score field '{score_field}' could not be cast to float.") from exc
            finite_mask = scores.is_finite().fill_null(False)
            scores = scores.filter(finite_mask)
            if len(scores) == 0:
                raise ValueError(f"Score field '{score_field}' has no finite values after filtering.")
            return score_field, scores


        @app.cell
        def _(mo, objective_name):
            is_sfxi = objective_name.lower().startswith("sfxi")
            mo.md("SFXI diagnostics enabled." if is_sfxi else "SFXI diagnostics not applicable.")
            return is_sfxi


        @app.cell
        def _(alt, color_by_ui, filtered_df, is_sfxi, mo, pl, plot_rows_ui, score_field):
            plot_limit = int(plot_rows_ui.value)
            plot_total_rows = filtered_df.height
            plot_limit = min(plot_limit, plot_total_rows)
            if plot_total_rows > plot_limit:
                plot_df = filtered_df.sample(n=plot_limit, seed=0, shuffle=True)
                plot_note = (
                    f"Plotting a random sample of {plot_limit} of {plot_total_rows} rows. "
                    "Increase Plot rows to include more data. Large charts may exceed Marimo output limits."
                )
            else:
                plot_df = filtered_df
                plot_note = (
                    f"Plotting all {plot_total_rows} rows. "
                    "Large charts may exceed Marimo output limits; lower Plot rows if needed."
                )
            mo.md(plot_note)
            color_field = str(color_by_ui.value)
            if color_field not in plot_df.columns:
                raise ValueError(f"Color field '{color_field}' not found in data.")
            color_dtype = plot_df.schema.get(color_field)
            color_type = "Q" if color_dtype is not None and color_dtype.is_numeric() else "N"
            color_enc = alt.Color(
                f"{color_field}:{color_type}",
                legend=alt.Legend(title=color_field),
            )

            select_cols = ["id", "sel__rank_competition", score_field, color_field]
            select_cols = list(dict.fromkeys(select_cols))
            base = plot_df.select(select_cols).with_columns(
                pl.col("sel__rank_competition").cast(pl.Int64),
                pl.col(score_field).cast(pl.Float64, strict=False),
            )
            base = base.filter(
                pl.col("sel__rank_competition").is_not_null()
                & pl.col(score_field).is_finite()
            )
            if base.is_empty():
                raise ValueError("No finite rank/score pairs after filtering.")
            base_data = base.to_pandas()

            hist = (
                alt.Chart(base_data, title="Score distribution")
                .mark_bar(opacity=0.75)
                .encode(
                    x=alt.X(
                        f"{score_field}:Q",
                        bin=alt.Bin(maxbins=40),
                        title=score_field,
                    ),
                    y=alt.Y("count():Q", title="Count"),
                    color=color_enc,
                    tooltip=[alt.Tooltip("count():Q", title="Count")],
                )
                .properties(width=260, height=240)
            )

            scatter = (
                alt.Chart(base_data, title="Score vs rank")
                .mark_circle(size=60, opacity=0.7)
                .encode(
                    x=alt.X("sel__rank_competition:Q", title="Rank (competition)"),
                    y=alt.Y(f"{score_field}:Q", title=score_field),
                    color=color_enc,
                    tooltip=[
                        alt.Tooltip("id:N", title="id"),
                        alt.Tooltip("sel__rank_competition:Q", title="rank"),
                        alt.Tooltip(f"{score_field}:Q", title=score_field),
                        alt.Tooltip(f"{color_field}:{color_type}", title=color_field),
                    ],
                )
                .properties(width=260, height=240)
            )

            if not is_sfxi:
                sfxi_chart = (
                    alt.Chart([{"note": "SFXI diagnostics not applicable."}])
                    .mark_text(color="black")
                    .encode(text="note:N")
                    .properties(width=260, height=240)
                )
            else:
                needed = ["obj__logic_fidelity", "obj__effect_scaled"]
                missing = [c for c in needed if c not in filtered_df.columns]
                if missing:
                    sfxi_chart = (
                        alt.Chart([{"note": f"Missing columns: {missing}"}])
                        .mark_text(color="black")
                        .encode(text="note:N")
                        .properties(width=260, height=240)
                    )
                else:
                    sfxi_cols = [
                        "id",
                        "obj__logic_fidelity",
                        "obj__effect_scaled",
                        color_field,
                    ]
                    sfxi_cols = list(dict.fromkeys(sfxi_cols))
                    sfxi_df = plot_df.select(sfxi_cols).with_columns(
                        pl.col("obj__logic_fidelity").cast(pl.Float64, strict=False),
                        pl.col("obj__effect_scaled").cast(pl.Float64, strict=False),
                    )
                    sfxi_df = sfxi_df.filter(
                        pl.col("obj__logic_fidelity").is_finite()
                        & pl.col("obj__effect_scaled").is_finite()
                    )
                    sfxi_data = sfxi_df.to_pandas()
                    sfxi_chart = (
                        alt.Chart(sfxi_data, title="SFXI: effect_scaled vs logic_fidelity")
                        .mark_circle(size=60, opacity=0.7)
                        .encode(
                            x=alt.X("obj__logic_fidelity:Q", title="logic_fidelity"),
                            y=alt.Y("obj__effect_scaled:Q", title="effect_scaled"),
                            color=color_enc,
                            tooltip=[
                                alt.Tooltip("id:N", title="id"),
                                alt.Tooltip("obj__logic_fidelity:Q", title="logic_fidelity"),
                                alt.Tooltip("obj__effect_scaled:Q", title="effect_scaled"),
                                alt.Tooltip(f"{color_field}:{color_type}", title=color_field),
                            ],
                        )
                        .properties(width=260, height=240)
                    )

            charts = alt.hconcat(hist, scatter, sfxi_chart).properties(
                background="white"
            ).configure_view(
                fill="white",
                stroke="black",
            ).configure_axis(
                labelColor="black",
                titleColor="black",
                tickColor="black",
                domainColor="black",
                grid=False,
            ).configure_legend(
                labelColor="black",
                titleColor="black",
            ).configure_title(
                color="black",
            )
            mo.ui.altair_chart(charts)
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
