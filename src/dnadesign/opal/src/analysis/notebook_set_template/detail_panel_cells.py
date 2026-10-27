"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/detail_panel_cells.py

Notebook-set template builders for campaign detail panel cells.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# ruff: noqa: E501

from __future__ import annotations

from ._support import block


def render_detail_panel_cells() -> str:
    """Render evidence and secondary detail cells."""

    return "\n\n".join((_evidence_cell(), _details_cell()))


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


__all__ = ["render_detail_panel_cells"]
