"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/layout_cells.py

Notebook-set template builders for final layout cells.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# ruff: noqa: E501

from __future__ import annotations

from ._support import block


def render_layout_cells() -> str:
    """Render the final notebook layout and app entrypoint cells."""

    return "\n\n".join((_layout_cell(), _main_cell()))


def _layout_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            active_view_mode,
            artifact_garden_panel,
            campaign_summary_df,
            collection_set_ui,
            campaign_ui,
            changes_panel,
            collection_visuals,
            evidence_panel,
            header_md,
            label_staging_panel,
            metric_definitions_panel,
            mo,
            opal_table,
            plot_panel,
            reader_evidence_artifact_ui,
            reader_evidence_panel,
            reader_evidence_plot_type_ui,
            reader_evidence_time_ui,
            reader_evidence_visual,
            selected_campaign_brief_md,
            selected_visual_choice,
            selected_overview_panel,
            selected_validity_md,
            view_mode_ui,
        ):
            _items = [header_md]
            if active_view_mode != "Campaign set":
                _top_control_items = [campaign_ui]
                if view_mode_ui is not None:
                    _top_control_items.append(view_mode_ui)
            elif collection_set_ui is not None:
                _top_control_items = [view_mode_ui, collection_set_ui] if view_mode_ui is not None else [collection_set_ui]
            else:
                _top_control_items = [item for item in [view_mode_ui] if item is not None]
            if _top_control_items:
                _items.append(mo.hstack(_top_control_items, justify="start", align="end", wrap=True, gap=0.35))
            if active_view_mode != "Campaign set":
                _items.append(selected_campaign_brief_md)
            _reader_plot_panel = None
            if reader_evidence_plot_type_ui is not None:
                _reader_controls = [reader_evidence_plot_type_ui]
                if reader_evidence_artifact_ui is not None:
                    _reader_controls.append(reader_evidence_artifact_ui)
                if reader_evidence_time_ui is not None:
                    _reader_controls.append(reader_evidence_time_ui)
                _reader_plot_panel = mo.vstack(
                    [
                        mo.hstack(_reader_controls, justify="start", align="end", wrap=True, gap=0.35),
                        reader_evidence_visual,
                    ],
                    gap=0.35,
                )
            _plot_items = []
            if selected_visual_choice is not None or _reader_plot_panel is None:
                _plot_items.append(plot_panel)
            if _reader_plot_panel is not None:
                _plot_items.append(_reader_plot_panel)
            _items.append(mo.vstack(_plot_items, gap=0.55))
            _campaign_inventory_label = "Raw campaign inventory" if collection_visuals else "Campaigns at a glance"
            _accordion_items = {
                _campaign_inventory_label: opal_table(campaign_summary_df, page_size=12),
            }
            if active_view_mode != "Campaign set":
                _status_panel = mo.vstack(
                    [selected_overview_panel, selected_validity_md, changes_panel, evidence_panel],
                    gap=0.35,
                )
                _data_panel = mo.vstack(
                    [reader_evidence_panel, label_staging_panel, metric_definitions_panel, artifact_garden_panel],
                    gap=0.35,
                )
                _accordion_items.update(
                    {
                        "Campaign status": _status_panel,
                        "Data and evidence records": _data_panel,
                    }
                )
            _items.append(mo.accordion(_accordion_items, multiple=True))
            mo.vstack(_items)
        """
    )


def _main_cell() -> str:
    return block(
        """
        if __name__ == "__main__":
            app.run()
        """
    )


__all__ = ["render_layout_cells"]
