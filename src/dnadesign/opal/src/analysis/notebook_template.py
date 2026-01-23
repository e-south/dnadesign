# ABOUTME: Renders marimo notebook templates for OPAL campaigns.
# ABOUTME: Generates scaffolded notebooks with campaign context and data previews.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/notebook_template.py

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

        app = marimo.App(width="full")


        @app.cell
        def _():
            import marimo as mo
            import polars as pl
            from pathlib import Path
            from dnadesign.opal.src.plots._mpl_utils import ensure_mpl_config_dir
            ensure_mpl_config_dir(workdir=Path(__CONFIG_PATH__).parent)
            import matplotlib.pyplot as plt
            from dnadesign.opal.src.analysis.facade import (
                CampaignAnalysis,
                available_rounds,
                latest_round,
                latest_run_id,
                require_columns,
            )
            return (
                mo,
                pl,
                plt,
                Path,
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
            info_lines = [
                "# OPAL Campaign Notebook",
                "",
                f"- Config: `{config_path}`",
                f"- Name: `{cfg.campaign.name}`",
                f"- Slug: `{cfg.campaign.slug}`",
                f"- Workdir: `{ws.workdir}`",
                f"- Records: `{store.records_path}`",
                f"- X column: `{cfg.data.x_column_name}`",
                f"- Y column: `{cfg.data.y_column_name}`",
            ]
            mo.md("\\n".join(info_lines))
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
            summary_df = pl.DataFrame(
                {
                    "field": [
                        "run_id",
                        "round",
                        "model",
                        "objective",
                        "selection",
                        "n_train",
                        "n_scored",
                    ],
                    "value": [
                        run_id,
                        int(run_meta.get("as_of_round", -1)),
                        run_meta.get("model__name"),
                        objective_name,
                        run_meta.get("selection__name"),
                        run_meta.get("stats__n_train"),
                        run_meta.get("stats__n_scored"),
                    ],
                }
            )
            return run_id, run_meta, objective_name, summary_df


        @app.cell
        def _(mo, summary_df):
            mo.ui.dataframe(summary_df)
            return


        @app.cell
        def _(campaign, require_columns, run_id, selected_round):
            labels_df = campaign.read_labels()
            pred_df = campaign.read_predictions(
                columns=[
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
                ],
                round_selector=[selected_round],
                run_id=run_id,
                allow_missing=True,
            )
            require_columns(
                pred_df,
                [
                    "id",
                    "run_id",
                    "as_of_round",
                    "pred__y_obj_scalar",
                    "sel__rank_competition",
                    "sel__is_selected",
                ],
                ctx="predictions",
            )
            if pred_df.is_empty():
                raise ValueError("No predictions found for selected run/round.")
            return labels_df, pred_df


        @app.cell
        def _(mo):
            data_source_ui = mo.ui.dropdown(
                options=["records", "labels", "predictions"],
                value="records",
                label="Data source",
            )
            return data_source_ui


        @app.cell
        def _(data_source_ui, labels_df, mo, pred_df, records_df):
            source = str(data_source_ui.value)
            if source == "labels":
                data_df = labels_df
            elif source == "predictions":
                data_df = pred_df
            else:
                data_df = records_df
            data_table = mo.ui.table(data_df, page_size=10)
            return data_df, data_table


        @app.cell
        def _(data_source_ui, data_table, mo):
            mo.vstack([data_source_ui, data_table])
            return


        @app.cell
        def _(mo, pred_df):
            ranks = pred_df.get_column("sel__rank_competition").drop_nulls()
            if len(ranks) == 0:
                raise ValueError("sel__rank_competition has no non-null values.")
            max_rank = int(ranks.max())
            if max_rank < 1:
                raise ValueError("sel__rank_competition must be >= 1.")
            default_max = min(max_rank, 2000)
            show_selected_ui = mo.ui.checkbox(label="Selected only", value=False)
            max_rank_ui = mo.ui.slider(1, max_rank, value=default_max, label="Max rank")
            return max_rank_ui, show_selected_ui


        @app.cell
        def _(pl, pred_df, show_selected_ui, max_rank_ui):
            df = pred_df
            if show_selected_ui.value and "sel__is_selected" in df.columns:
                df = df.filter(pl.col("sel__is_selected") == True)
            max_rank_val = int(max_rank_ui.value)
            df = df.filter(pl.col("sel__rank_competition") <= max_rank_val)
            return df


        @app.cell
        def _(mo, filtered_df):
            mo.stop(filtered_df.is_empty(), mo.md("No rows matched the current filters."))
            return


        @app.cell
        def _(mo, pred_df):
            candidates = [
                c
                for c in pred_df.columns
                if c.startswith("pred__") or c.startswith("obj__")
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
        def _(mo, max_rank_ui, score_field_ui, show_selected_ui):
            mo.vstack([score_field_ui, show_selected_ui, max_rank_ui])
            return


        @app.cell
        def _(pl, filtered_df, score_field_ui):
            score_field = str(score_field_ui.value)
            try:
                scores = filtered_df.get_column(score_field).cast(pl.Float64)
            except Exception as exc:
                raise ValueError(f"Score field '{score_field}' could not be cast to float.") from exc
            finite_mask = scores.is_finite().fill_null(False)
            scores = scores.filter(finite_mask)
            if len(scores) == 0:
                raise ValueError(f"Score field '{score_field}' has no finite values after filtering.")
            return score_field, scores


        @app.cell
        def _(plt, scores, score_field):
            def _make_hist():
                fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
                vals = scores.drop_nulls().to_list()
                ax.hist(vals, bins=40, alpha=0.75)
                ax.set_title(f"Score distribution: {score_field}")
                ax.set_xlabel(score_field)
                ax.set_ylabel("Count")
                return fig

            score_hist_fig = _make_hist()
            score_hist_fig
            return score_hist_fig


        @app.cell
        def _(plt, pl, filtered_df, score_field):
            def _make_scatter():
                fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
                df = filtered_df.select(
                    ["sel__rank_competition", score_field, "sel__is_selected"]
                ).with_columns(
                    pl.col("sel__rank_competition").cast(pl.Int64),
                    pl.col(score_field).cast(pl.Float64),
                )
                df = df.filter(
                    pl.col("sel__rank_competition").is_not_null()
                    & pl.col(score_field).is_finite()
                )
                if df.is_empty():
                    raise ValueError("No finite rank/score pairs after filtering.")
                x = df["sel__rank_competition"].to_list()
                y = df[score_field].to_list()
                ax.scatter(x, y, s=18, alpha=0.5)
                ax.set_title("Score vs rank")
                ax.set_xlabel("Rank (competition)")
                ax.set_ylabel(score_field)
                ax.set_xlim(max(x), 1)
                return fig

            score_scatter_fig = _make_scatter()
            score_scatter_fig
            return score_scatter_fig


        @app.cell
        def _(mo, objective_name):
            is_sfxi = objective_name.lower().startswith("sfxi")
            mo.md("SFXI diagnostics enabled." if is_sfxi else "SFXI diagnostics not applicable.")
            return is_sfxi


        @app.cell
        def _(mo, plt, pl, filtered_df, is_sfxi):
            mo.stop(not is_sfxi, mo.md("SFXI diagnostics not applicable."))
            needed = ["obj__logic_fidelity", "obj__effect_scaled"]
            missing = [c for c in needed if c not in filtered_df.columns]
            mo.stop(bool(missing), mo.md(f"Missing SFXI diagnostic columns: {missing}"))

            def _make_sfxi():
                fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
                x = filtered_df["obj__logic_fidelity"].cast(pl.Float64, strict=False).to_list()
                y = filtered_df["obj__effect_scaled"].cast(pl.Float64, strict=False).to_list()
                ax.scatter(x, y, s=18, alpha=0.5)
                ax.set_title("SFXI: effect_scaled vs logic_fidelity")
                ax.set_xlabel("logic_fidelity")
                ax.set_ylabel("effect_scaled")
                return fig

            sfxi_diag_fig = _make_sfxi()
            sfxi_diag_fig
            return sfxi_diag_fig


        @app.cell
        def _(mo, filtered_df):
            mo.ui.data_explorer(filtered_df)
            return


        @app.cell
        def _(mo, labels_df):
            mo.ui.dataframe(labels_df)
            return


        if __name__ == "__main__":
            app.run()
        """
    ).strip()

    return (
        template.replace("__CONFIG_PATH__", repr(str(config_path)))
        .replace("__DEFAULT_ROUND__", repr(str(round_selector)))
        .replace("__GENERATED_WITH__", str(marimo_version))
        + "\n"
    )
