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
            _selection_batch_scope_cell(),
            _selected_run_labels_cell(),
        )
    )


def _selection_batch_scope_cell() -> str:
    return block(
        """
        @app.cell
        def _(baserender_campaign_model, resolve_notebook_baserender_selection_batch_scope):
            selected_baserender_round, selected_baserender_run_id = (
                resolve_notebook_baserender_selection_batch_scope(
                    baserender_campaign_model.get("selection_batch")
                )
            )
            baserender_round_ui = None
            baserender_run_ui = None
            return (
                baserender_round_ui,
                baserender_run_ui,
                selected_baserender_round,
                selected_baserender_run_id,
            )
        """
    )


def _selected_run_labels_cell() -> str:
    return block(
        """
        @app.cell
        def _(pl, selected_baserender_round, selected_baserender_run_id, selected_campaign_analysis):
            if selected_baserender_run_id is None or selected_baserender_round is None:
                selected_campaign_labels_df = pl.DataFrame()
            else:
                selected_campaign_labels_df = selected_campaign_analysis.read_run_labels_used(
                    round_selector=selected_baserender_round,
                    run_id=selected_baserender_run_id,
                )
            return selected_campaign_labels_df,
        """
    )


__all__ = ["render_baserender_scope_cells"]
