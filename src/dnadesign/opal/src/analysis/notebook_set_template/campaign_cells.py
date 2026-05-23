from __future__ import annotations

from ._support import block


def render_campaign_cells() -> str:
    """Render campaign selection and selected-campaign overview cells."""

    return "\n\n".join((_campaign_selector_cell(), _selected_campaign_cell(), _selected_overview_cell()))


def _campaign_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(build_notebook_campaign_summary_row, campaign_set_view_model, campaigns, mo, pl, selected_round_selector):
            _rows = [build_notebook_campaign_summary_row(campaign_model) for campaign_model in campaigns]
            campaign_labels = [f"{index + 1}. {row['label']}" for index, row in enumerate(_rows)]
            campaign_ui = mo.ui.dropdown(campaign_labels, value=campaign_labels[0], label="Campaign")
            campaign_summary_df = pl.DataFrame(_rows)
            header_md = mo.md(
                "# Campaigns\\n\\n"
                f"`{campaign_set_view_model['campaign_count']}` campaigns. "
                f"Round selector: `{selected_round_selector}`."
            )
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


def _selected_overview_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            build_notebook_at_a_glance_rows,
            build_notebook_campaign_header_lines,
            build_notebook_validity_rows,
            mo,
            pl,
            selected_campaign_model,
        ):
            selected_campaign_header_md = mo.md(
                "\\n".join(build_notebook_campaign_header_lines(selected_campaign_model, heading_level=2))
            )
            selected_overview_panel = mo.ui.table(
                pl.DataFrame(build_notebook_at_a_glance_rows(selected_campaign_model)),
                page_size=14,
            )
            selected_validity_md = mo.ui.table(
                pl.DataFrame(build_notebook_validity_rows(selected_campaign_model)),
                page_size=14,
            )
            return selected_campaign_header_md, selected_overview_panel, selected_validity_md
        """
    )


__all__ = ["render_campaign_cells"]
