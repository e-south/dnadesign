"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/collection_cells.py

Notebook-set template builders for collection cells OPAL analysis notebook set template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ._support import block


def render_collection_cells() -> str:
    """Render campaign-set mode and set-selection controls."""

    return "\n".join(
        (
            _collection_visual_model_cell(),
            _view_mode_cell(),
            _active_view_mode_cell(),
            _collection_set_selector_cell(),
            _selected_collection_set_cell(),
            _selected_collection_heading_cell(),
        )
    )


def _collection_visual_model_cell() -> str:
    return block(
        """
        @app.cell
        def _(build_notebook_collection_set_choices, campaigns, collection_visuals):
            collection_set_choices = [] if len(campaigns) == 1 else build_notebook_collection_set_choices(
                collection_visuals
            )
            return collection_set_choices
        """
    )


def _view_mode_cell() -> str:
    return block(
        """
        @app.cell
        def _(collection_set_choices, mo):
            if collection_set_choices:
                view_mode_ui = mo.ui.dropdown(
                    {
                        "Campaign": "Campaign",
                        "Cross-campaign comparison": "Campaign set",
                    },
                    value="Campaign",
                    label="Review scope",
                )
            else:
                view_mode_ui = None
            return view_mode_ui
        """
    )


def _active_view_mode_cell() -> str:
    return block(
        """
        @app.cell
        def _(view_mode_ui):
            active_view_mode = str(view_mode_ui.value) if view_mode_ui is not None else "Campaign"
            return active_view_mode
        """
    )


def _collection_set_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(active_view_mode, collection_set_choices, mo):
            if active_view_mode == "Campaign set" and collection_set_choices:
                collection_set_ui = mo.ui.dropdown(
                    [choice["label"] for choice in collection_set_choices],
                    value=collection_set_choices[0]["label"],
                    label="Comparison group",
                )
            else:
                collection_set_ui = None
            return collection_set_ui
        """
    )


def _selected_collection_set_cell() -> str:
    return block(
        """
        @app.cell
        def _(collection_set_choices, collection_set_ui):
            if collection_set_ui is None:
                selected_collection_set_choice = collection_set_choices[0] if collection_set_choices else None
            else:
                _selected = str(collection_set_ui.value)
                selected_collection_set_choice = next(
                    choice for choice in collection_set_choices if choice["label"] == _selected
                )
            return selected_collection_set_choice
        """
    )


def _selected_collection_heading_cell() -> str:
    return block(
        """
        @app.cell
        def _(mo, selected_collection_set_choice):
            _label = (
                str(selected_collection_set_choice.get("label") or "").strip()
                if selected_collection_set_choice is not None else ""
            )
            _title = "Cross-campaign comparison" + (f" · {_label}" if _label else "")
            selected_collection_set_title_md = mo.md(f"# {_title}")
            return selected_collection_set_title_md
        """
    )


__all__ = ["render_collection_cells"]
