"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/details_cells.py

Notebook-set template builders for details cells OPAL analysis notebook set template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# ruff: noqa: E501

from __future__ import annotations

from ._support import block


def render_details_cells() -> str:
    """Render evidence, secondary detail panels, layout, and app entrypoint cells."""

    return "\n\n".join(
        (
            _evidence_cell(),
            _details_cell(),
            _reader_evidence_plot_type_cell(),
            _reader_evidence_artifact_cell(),
            _reader_evidence_visual_cell(),
            _layout_cell(),
            _main_cell(),
        )
    )


def _evidence_cell() -> str:
    return block(
        """
        @app.cell
        def _(build_notebook_evidence_rows, mo, opal_table, pl, selected_campaign_model):
            _rows = build_notebook_evidence_rows(selected_campaign_model)
            if _rows:
                evidence_panel = opal_table(pl.DataFrame(_rows), page_size=10)
            else:
                evidence_panel = mo.md("No warnings or stale artifacts reported for this campaign.")
            return evidence_panel
        """
    )


def _details_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            build_notebook_artifact_garden_rows,
            build_notebook_artifact_garden_summary_rows,
            build_notebook_change_rows,
            build_notebook_change_summary_rows,
            build_notebook_label_staging_rows,
            build_notebook_metric_definition_rows,
            mo,
            opal_table,
            pl,
            render_notebook_reader_evidence_panel,
            selected_campaign_model,
        ):
            _df, _table = pl.DataFrame, opal_table
            _metric_rows = build_notebook_metric_definition_rows(selected_campaign_model)
            metric_definitions_panel = (
                _table(_df(_metric_rows), page_size=8) if _metric_rows else mo.md("No metric definitions.")
            )

            _change_rows = build_notebook_change_rows(selected_campaign_model)
            changes_table = _table(_df(_change_rows), page_size=8) if _change_rows else mo.md("No round changes.")
            changes_panel = mo.vstack(
                [
                    _table(_df(build_notebook_change_summary_rows(selected_campaign_model)), page_size=8),
                    changes_table,
                ]
            )

            _artifact_rows = build_notebook_artifact_garden_rows(selected_campaign_model)
            _artifact_summary_rows = build_notebook_artifact_garden_summary_rows(selected_campaign_model)
            artifact_rows_panel = (
                _table(_df(_artifact_rows), page_size=12) if _artifact_rows else mo.md("No artifacts.")
            )
            artifact_garden_panel = mo.vstack(
                [
                    _table(_df(_artifact_summary_rows), page_size=8),
                    artifact_rows_panel,
                ]
            )
            _label_rows = build_notebook_label_staging_rows(selected_campaign_model)
            label_staging_panel = _table(_df(_label_rows), page_size=8) if _label_rows else mo.md("No label inputs.")

            _reader_evidence = render_notebook_reader_evidence_panel(
                selected_campaign_model,
                mo=mo,
                opal_table=opal_table,
                pl=pl,
            )
            reader_evidence_panel = _reader_evidence["panel"]
            reader_evidence_surface = _reader_evidence["surface"]
            return (
                artifact_garden_panel, changes_panel, label_staging_panel, metric_definitions_panel,
                reader_evidence_panel, reader_evidence_surface,
            )
        """
    )


def _reader_evidence_plot_type_cell() -> str:
    return block(
        """
        @app.cell
        def _(mo, reader_evidence_surface, render_notebook_reader_evidence_plot_type_control):
            reader_evidence_plot_type_ui = render_notebook_reader_evidence_plot_type_control(
                reader_evidence_surface,
                mo=mo,
            )
            return reader_evidence_plot_type_ui
        """
    )


def _reader_evidence_artifact_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            mo,
            reader_evidence_plot_type_ui,
            reader_evidence_surface,
            render_notebook_reader_evidence_artifact_control,
        ):
            selected_reader_evidence_plot_type_label = (
                str(reader_evidence_plot_type_ui.value) if reader_evidence_plot_type_ui is not None else None
            )
            reader_evidence_artifact_ui = render_notebook_reader_evidence_artifact_control(
                reader_evidence_surface,
                selected_plot_type_label=selected_reader_evidence_plot_type_label,
                mo=mo,
            )
            return reader_evidence_artifact_ui, selected_reader_evidence_plot_type_label
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
            render_notebook_reader_evidence_artifact_visual,
            selected_reader_evidence_plot_type_label,
        ):
            _selected_artifact_label = (
                None if reader_evidence_artifact_ui is None else str(reader_evidence_artifact_ui.value)
            )
            reader_evidence_visual = render_notebook_reader_evidence_artifact_visual(
                reader_evidence_surface,
                selected_plot_type_label=selected_reader_evidence_plot_type_label,
                selected_artifact_label=_selected_artifact_label,
                mo=mo,
            )
            return reader_evidence_visual
        """
    )


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
            reader_evidence_visual,
            selected_campaign_brief_md,
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
                _items.append(mo.vstack(_top_control_items, gap=0.20))
            if active_view_mode != "Campaign set": _items.append(selected_campaign_brief_md)
            _plot_items = [plot_panel]
            if reader_evidence_plot_type_ui is not None:
                _reader_controls = [reader_evidence_plot_type_ui]
                if reader_evidence_artifact_ui is not None:
                    _reader_controls.append(reader_evidence_artifact_ui)
                _plot_items.append(mo.vstack([mo.hstack(_reader_controls, justify="start", align="end", wrap=True, gap=0.35), reader_evidence_visual], gap=0.35))
            _items.append(mo.vstack(_plot_items, gap=0.55))
            _campaign_inventory_label = "Raw campaign inventory" if collection_visuals else "Campaigns at a glance"
            _accordion_items = {
                _campaign_inventory_label: opal_table(campaign_summary_df, page_size=12),
            }
            if active_view_mode != "Campaign set":
                _status_panel = mo.vstack([selected_overview_panel, selected_validity_md, changes_panel], gap=0.35)
                _data_panel = mo.vstack([label_staging_panel,metric_definitions_panel,artifact_garden_panel], gap=0.35)
                _accordion_items.update(
                    {
                        "Campaign status": _status_panel,
                        "Reader evidence records": reader_evidence_panel,
                        "Data inputs and artifacts": _data_panel,
                        "Warnings and stale artifacts": evidence_panel,
                    }
                )
            _items.append(mo.accordion(_accordion_items, multiple=True)); mo.vstack(_items)
        """
    )


def _main_cell() -> str:
    return block(
        """
        if __name__ == "__main__":
            app.run()
        """
    )


__all__ = ["render_details_cells"]
