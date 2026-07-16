"""Generated marimo cells for manifest-declared layered-scatter controls."""

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
            return layered_scatter_contract, layered_scatter_controls
        """
    )


def _state_cell() -> str:
    return block(
        """
        @app.cell
        def _(layered_scatter_controls, read_notebook_layered_scatter_state):
            plot_view_state = read_notebook_layered_scatter_state(layered_scatter_controls)
            return plot_view_state
        """
    )


__all__ = ["render_layered_scatter_cells"]
