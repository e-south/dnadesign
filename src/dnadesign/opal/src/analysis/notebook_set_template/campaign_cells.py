"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/campaign_cells.py

Notebook-set template builders for campaign cells OPAL analysis notebook set template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# ruff: noqa: E501

from __future__ import annotations

from ._support import block


def render_campaign_cells() -> str:
    """Render campaign selection and selected-campaign overview cells."""

    return "\n".join(
        (
            _campaign_selector_cell(),
            _selected_campaign_cell(),
            _selection_view_selector_cell(),
            _selected_selection_view_cell(),
            _selected_overview_cell(),
        )
    )


def _campaign_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            build_notebook_campaign_summary_row,
            campaigns,
            mo,
            pl,
        ):
            _rows = [build_notebook_campaign_summary_row(campaign_model) for campaign_model in campaigns]
            campaign_labels = [f"{index + 1}. {row['label']}" for index, row in enumerate(_rows)]
            campaign_ui = mo.ui.dropdown(campaign_labels, value=campaign_labels[0], label="Campaign")
            campaign_summary_df = pl.DataFrame(_rows)
            header_md = mo.md("# OPAL Review Notebook")
            return campaign_labels, campaign_summary_df, campaign_ui, header_md
        """
    )


def _selected_campaign_cell() -> str:
    return block(
        """
        @app.cell
        def _(campaign_labels, campaign_ui, campaigns):
            selected_label = str(campaign_ui.value)
            selected_index = campaign_labels.index(selected_label)
            selected_campaign_model = campaigns[selected_index]
            return selected_campaign_model, selected_label
        """
    )


def _selection_view_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(build_notebook_selection_view_options, mo, selected_campaign_model):
            _labels = build_notebook_selection_view_options(selected_campaign_model)
            selection_view_ui = mo.ui.dropdown(options=_labels, value=next(iter(_labels)), label="Selection view")
            return selection_view_ui,
        """
    )


def _selected_selection_view_cell() -> str:
    return block(
        """
        @app.cell
        def _(resolve_notebook_selection_view, selected_campaign_model, selection_view_ui):
            selected_selection_view_id = str(selection_view_ui.value)
            selected_selection_view = resolve_notebook_selection_view(selected_campaign_model, selected_selection_view_id)
            return selected_selection_view, selected_selection_view_id
        """
    )


def _selected_overview_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            build_notebook_at_a_glance_rows,
            build_notebook_campaign_header_lines,
            build_notebook_validity_rows,
            mo,
            opal_table,
            pl,
            selected_campaign_model,
            selected_selection_view,
        ):
            _header_lines = build_notebook_campaign_header_lines(
                selected_campaign_model, selection_view=selected_selection_view, heading_level=2
            )
            selected_campaign_brief_md = mo.md(_header_lines[2] if len(_header_lines) > 2 else "")
            _overview_rows = build_notebook_at_a_glance_rows(
                selected_campaign_model, selection_view=selected_selection_view
            )
            selected_overview_panel = opal_table(pl.DataFrame(_overview_rows), page_size=14)
            selected_validity_md = opal_table(
                pl.DataFrame(build_notebook_validity_rows(selected_campaign_model)),
                page_size=14,
            )
            return selected_campaign_brief_md, selected_overview_panel, selected_validity_md
        """
    )


__all__ = ["render_campaign_cells"]
