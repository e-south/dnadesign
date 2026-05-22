from __future__ import annotations

from textwrap import dedent

RUN_CELLS = dedent(
    """
    @app.cell
    def _(
        available_rounds,
        build_notebook_no_run_lines,
        cfg,
        default_round,
        latest_round,
        mo,
        runs_df,
        runs_read,
        resolve_notebook_round_default,
        table_status_lines,
    ):
        rounds = available_rounds(runs_df)
        objective_name_default = (
            str(cfg.objectives.objectives[0].name)
            if cfg.objectives.objectives
            else ""
        )
        if rounds:
            round_default = resolve_notebook_round_default(default_round, rounds, latest_round(runs_df))
            round_ui = mo.ui.dropdown(rounds, value=round_default, label="Round")
            round_run_intro_md = mo.md("### Round and run")
        else:
            round_ui = None
            round_run_intro_md = mo.md(
                "\\n".join(
                    build_notebook_no_run_lines(
                        table_status_lines(runs_read),
                        expected_runs_ledger="outputs/ledger/runs.parquet",
                        no_run_message="No runs available yet.",
                    )
                )
            )
        return objective_name_default, round_run_intro_md, round_ui, rounds


    @app.cell
    def _(pl, round_ui, runs_df):
        selected_round = None
        runs_for_round = runs_df.head(0)
        if round_ui is not None:
            selected_round = int(round_ui.value)
            runs_for_round = runs_df.filter(pl.col("as_of_round") == selected_round)
            if runs_for_round.is_empty():
                raise ValueError(f"No runs found for round {selected_round}.")
        return runs_for_round, selected_round


    @app.cell
    def _(build_notebook_run_options, latest_run_id, mo, runs_for_round, selected_round):
        run_ui = None
        if selected_round is not None:
            run_default = latest_run_id(runs_for_round)
            run_options = build_notebook_run_options(runs_for_round)
            run_ui = mo.ui.dropdown(run_options, value=run_default, label="Run ID")
        return run_ui


    @app.cell
    def _(
        build_notebook_run_summary_lines,
        mo,
        objective_name_default,
        pl,
        round_run_intro_md,
        round_ui,
        run_ui,
        runs_for_round,
        selected_round,
    ):
        objective_name = objective_name_default
        run_id = None
        run_meta = {}
        run_summary_md = mo.md("")
        round_run_controls = round_run_intro_md
        if run_ui is not None:
            run_id = str(run_ui.value)
            run_row = runs_for_round.filter(pl.col("run_id") == run_id)
            if run_row.is_empty():
                raise ValueError(f"Run id not found: {run_id}")
            run_meta = run_row.to_dicts()[0]
            objective_name = str(run_meta.get("objective__name") or objective_name)
            run_summary_lines = build_notebook_run_summary_lines(run_id, run_meta, objective_name)
            round_run_controls = mo.vstack([round_ui, run_ui])
            run_summary_md = mo.md("\\n".join(run_summary_lines))
        return (
            objective_name,
            round_run_controls,
            run_id,
            run_meta,
            run_summary_md,
            runs_for_round,
            selected_round,
        )
    """
).strip("\n")
