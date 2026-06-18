"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/baserender_scope_cells.py

Notebook-set template builders for BaseRender scope cells OPAL analysis notebook set.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ._support import block
from .baserender_campaign_scope_cells import render_baserender_campaign_scope_cells


def render_baserender_scope_cells() -> str:
    return "\n\n".join(
        (
            render_baserender_campaign_scope_cells(),
            _selection_round_cell(),
            _selection_run_cell(),
            _selected_round_value_cell(),
        )
    )


def _selection_round_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            available_rounds,
            latest_round,
            mo,
            resolve_notebook_round_default,
            selected_campaign_runs_df,
            selected_round_selector,
        ):
            baserender_rounds = available_rounds(selected_campaign_runs_df)
            if baserender_rounds:
                _round_selector = (
                    "latest"
                    if str(selected_round_selector).strip().lower() == "all"
                    else selected_round_selector
                )
                _default_round = resolve_notebook_round_default(
                    _round_selector,
                    baserender_rounds,
                    latest_round(selected_campaign_runs_df),
                )
                baserender_round_ui = mo.ui.dropdown(
                    baserender_rounds,
                    value=_default_round,
                    label="Selection round",
                )
            else:
                baserender_round_ui = None
            return baserender_round_ui, baserender_rounds
        """
    )


def _selection_run_cell() -> str:
    return block(
        """
        @app.cell
        def _(baserender_round_ui, pl, selected_campaign_runs_df):
            selected_baserender_round = None
            baserender_runs_for_round = selected_campaign_runs_df.head(0)
            if baserender_round_ui is not None:
                selected_baserender_round = int(baserender_round_ui.value)
                baserender_runs_for_round = selected_campaign_runs_df.filter(
                    pl.col("as_of_round") == selected_baserender_round
                )
            return baserender_runs_for_round, selected_baserender_round
        """
    )


def _selected_round_value_cell() -> str:
    return block(
        """
        @app.cell
        def _(baserender_runs_for_round, build_notebook_run_options, latest_run_id, mo):
            baserender_run_options = []
            baserender_run_ui = None
            if not baserender_runs_for_round.is_empty():
                baserender_run_options = build_notebook_run_options(baserender_runs_for_round)
                _default_run = latest_run_id(baserender_runs_for_round)
                baserender_run_ui = mo.ui.dropdown(
                    baserender_run_options,
                    value=_default_run,
                    label="Selection run",
                )
            return baserender_run_options, baserender_run_ui
        """
    )


__all__ = ["render_baserender_scope_cells"]
