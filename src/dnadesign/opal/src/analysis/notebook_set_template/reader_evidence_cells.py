"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/reader_evidence_cells.py

Notebook-set template builders for Reader evidence selector cells.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ._support import block


def render_reader_evidence_cells() -> str:
    """Render Reader evidence selector and visual cells."""

    return "\n\n".join(
        (
            _reader_evidence_plot_type_cell(),
            _reader_evidence_artifact_cell(),
            _reader_evidence_time_cell(),
            _reader_evidence_visual_cell(),
        )
    )


def _reader_evidence_plot_type_cell() -> str:
    return block(
        """
        @app.cell
        def _(selected_visual_choice):
            reader_evidence_plot_type_ui = None
            selected_reader_evidence_plot_type_label = None
            if (
                selected_visual_choice is not None
                and selected_visual_choice.get("surface_kind") == "reader_evidence"
            ):
                selected_reader_evidence_plot_type_label = str(
                    selected_visual_choice.get("reader_plot_type_label") or ""
                ).strip()
                if not selected_reader_evidence_plot_type_label:
                    raise ValueError("Reader deliverable choice is missing reader_plot_type_label.")
            return reader_evidence_plot_type_ui, selected_reader_evidence_plot_type_label
        """
    )


def _reader_evidence_artifact_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            mo,
            reader_evidence_surface,
            render_notebook_reader_evidence_artifact_control,
            selected_reader_evidence_plot_type_label,
        ):
            reader_evidence_artifact_ui = render_notebook_reader_evidence_artifact_control(
                reader_evidence_surface,
                selected_plot_type_label=selected_reader_evidence_plot_type_label,
                mo=mo,
            )
            return reader_evidence_artifact_ui, selected_reader_evidence_plot_type_label
        """
    )


def _reader_evidence_time_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            mo,
            reader_evidence_artifact_ui,
            reader_evidence_surface,
            render_notebook_reader_evidence_time_control,
            selected_reader_evidence_plot_type_label,
        ):
            _selected_artifact_label = (
                None if reader_evidence_artifact_ui is None else str(reader_evidence_artifact_ui.value)
            )
            reader_evidence_time_ui = render_notebook_reader_evidence_time_control(
                reader_evidence_surface,
                selected_plot_type_label=selected_reader_evidence_plot_type_label,
                selected_artifact_label=_selected_artifact_label,
                mo=mo,
            )
            return reader_evidence_time_ui
        """
    )


def _reader_evidence_visual_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            mo,
            reader_evidence_artifact_ui,
            reader_evidence_surface,
            reader_evidence_time_ui,
            render_notebook_reader_evidence_artifact_visual,
            selected_reader_evidence_plot_type_label,
        ):
            if selected_reader_evidence_plot_type_label is None:
                reader_evidence_visual = None
            else:
                _selected_artifact_label = (
                    None if reader_evidence_artifact_ui is None else str(reader_evidence_artifact_ui.value)
                )
                _selected_time_h = None
                if reader_evidence_time_ui is not None and hasattr(reader_evidence_time_ui, "value"):
                    _selected_time_h = float(reader_evidence_time_ui.value)
                reader_evidence_visual = render_notebook_reader_evidence_artifact_visual(
                    reader_evidence_surface,
                    selected_plot_type_label=selected_reader_evidence_plot_type_label,
                    selected_artifact_label=_selected_artifact_label,
                    mo=mo,
                    selected_time_h=_selected_time_h,
                )
            return reader_evidence_visual
        """
    )


__all__ = ["render_reader_evidence_cells"]
