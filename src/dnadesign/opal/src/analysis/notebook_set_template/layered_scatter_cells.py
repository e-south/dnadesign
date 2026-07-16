"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/layered_scatter_cells.py

Generate marimo cells for manifest-declared layered-scatter controls.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ._support import block


def render_layered_scatter_cells() -> str:
    """Render persistent controls and view state for generic layered scatters."""

    return "\n".join((_memory_cell(), _control_cell(), _state_cell()))


def _memory_cell() -> str:
    return block(
        """
        @app.cell
        def _(mo):
            layered_scatter_memory, set_layered_scatter_memory = mo.state({})
            return layered_scatter_memory, set_layered_scatter_memory
        """
    )


def _control_cell() -> str:
    return block(
        """
        @app.cell
        def _(build_notebook_layered_scatter_contract, build_notebook_layered_scatter_controls,
              layered_scatter_memory, mo, plot_scope_ui, selected_visual_choice,
              select_notebook_plot_scope, set_layered_scatter_memory):
            _scope_label = str(plot_scope_ui.value) if plot_scope_ui is not None else None
            _s = (
                select_notebook_plot_scope(selected_visual_choice, _scope_label)
                if selected_visual_choice is not None else None
            )
            layered_scatter_contract = build_notebook_layered_scatter_contract(_s) if _s is not None else None
            layered_scatter_controls = build_notebook_layered_scatter_controls(
                layered_scatter_contract, memory=layered_scatter_memory,
                set_memory=set_layered_scatter_memory, mo=mo,
            )
            scatter_prediction_pool_ui = layered_scatter_controls["prediction_pool"]
            scatter_selected_ui = layered_scatter_controls["selected"]
            scatter_observed_batches_ui = layered_scatter_controls["observed_batches"]
            scatter_labels_ui = layered_scatter_controls["labels"]
            return (layered_scatter_contract, layered_scatter_controls, scatter_labels_ui,
                    scatter_observed_batches_ui, scatter_prediction_pool_ui, scatter_selected_ui)
        """
    )


def _state_cell() -> str:
    return block(
        """
        @app.cell
        def _(scatter_labels_ui, scatter_observed_batches_ui,
              scatter_prediction_pool_ui, scatter_selected_ui,
              read_notebook_layered_scatter_state):
            plot_view_state = read_notebook_layered_scatter_state({
                "prediction_pool": scatter_prediction_pool_ui,
                "selected": scatter_selected_ui,
                "observed_batches": scatter_observed_batches_ui,
                "labels": scatter_labels_ui,
            })
            return plot_view_state
        """
    )


__all__ = ["render_layered_scatter_cells"]
