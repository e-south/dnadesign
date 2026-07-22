"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/visual_selector_cells.py

Generated Marimo cells for visual group and deliverable selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ._support import block


def render_visual_selector_cells() -> str:
    return "\n".join(
        (
            _visual_memory_cell(),
            _visual_group_selector_cell(),
            _filtered_visual_choices_cell(),
            _visual_selector_cell(),
            _selected_visual_cell(),
        )
    )


def _visual_memory_cell() -> str:
    return block(
        """
        @app.cell
        def _(mo):
            plot_scope_label_memory, set_plot_scope_label_memory = mo.state({})
            visual_group_label_memory, set_visual_group_label_memory = mo.state(None)
            visual_label_memory, set_visual_label_memory = mo.state({})
            return (
                plot_scope_label_memory,
                set_plot_scope_label_memory,
                set_visual_group_label_memory,
                set_visual_label_memory,
                visual_group_label_memory,
                visual_label_memory,
            )
        """
    )


def _visual_group_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            build_notebook_visual_group_options,
            mo,
            set_visual_group_label_memory,
            visual_choices,
            visual_group_label_memory,
        ):
            visual_group_options = build_notebook_visual_group_options(visual_choices)
            if visual_group_options:
                _preferred = visual_group_label_memory()
                _preferred = _preferred if _preferred in visual_group_options else visual_group_options[0]
                visual_group_ui = mo.ui.dropdown(
                    visual_group_options,
                    value=_preferred,
                    label="Review section",
                    on_change=set_visual_group_label_memory,
                )
            else:
                visual_group_ui = None
            return visual_group_options, visual_group_ui
        """
    )


def _filtered_visual_choices_cell() -> str:
    return block(
        """
        @app.cell
        def _(filter_notebook_visual_choices_by_group, visual_choices, visual_group_ui):
            selected_visual_group_label = str(visual_group_ui.value) if visual_group_ui is not None else None
            visual_choices_in_group = filter_notebook_visual_choices_by_group(
                visual_choices, selected_visual_group_label
            )
            return selected_visual_group_label, visual_choices_in_group
        """
    )


def _visual_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(mo, selected_visual_group_label, set_visual_label_memory,
              visual_choices_in_group, visual_label_memory):
            if visual_choices_in_group:
                _labels = [choice["label"] for choice in visual_choices_in_group]
                _memory_key = str(selected_visual_group_label or "ungrouped")
                _memory = dict(visual_label_memory())
                _preferred = _memory.get(_memory_key) if _memory.get(_memory_key) in _labels else _labels[0]
                def _remember_visual(value):
                    set_visual_label_memory({**_memory, _memory_key: str(value)})
                plot_ui = mo.ui.dropdown(
                    _labels, value=_preferred, label="Deliverable", on_change=_remember_visual
                )
            else:
                plot_ui = None
            return plot_ui
        """
    )


def _selected_visual_cell() -> str:
    return block(
        """
        @app.cell
        def _(plot_ui, visual_choices_in_group):
            selected_visual_choice = (
                None
                if plot_ui is None
                else next(choice for choice in visual_choices_in_group if choice["label"] == str(plot_ui.value))
            )
            return selected_visual_choice
        """
    )


__all__ = ["render_visual_selector_cells"]
