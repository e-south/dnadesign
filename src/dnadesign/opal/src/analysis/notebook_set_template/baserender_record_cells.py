"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/baserender_record_cells.py

Notebook-set template builders for BaseRender record cells OPAL analysis notebook set.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ._support import block


def render_baserender_record_cells() -> str:
    return "\n\n".join(
        (
            _selected_record_ids_cell(),
            _selected_record_selector_cell(),
            _selected_record_row_cell(),
        )
    )


def _selected_record_ids_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            build_notebook_selected_baserender_records,
            selected_baserender_round,
            selected_baserender_run_id,
            selected_baserender_selection_view_id,
            selected_campaign_analysis,
        ):
            selected_baserender_records, selected_baserender_status_rows = build_notebook_selected_baserender_records(
                selected_campaign_analysis,
                selection_view_id=selected_baserender_selection_view_id,
                round_value=selected_baserender_round,
                run_id=selected_baserender_run_id,
            )
            return selected_baserender_records, selected_baserender_status_rows
        """
    )


def _selected_record_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            build_notebook_baserender_review_state,
            mo,
            opal_table,
            pl,
            selected_baserender_records,
            selected_baserender_status_rows,
            selected_campaign_baserender_contract,
            selected_campaign_store,
        ):
            (
                baserender_has_renderable_records,
                baserender_record_selector,
                baserender_diagnostic_panel,
            ) = build_notebook_baserender_review_state(
                selected_campaign_store.records_path,
                selected_campaign_baserender_contract,
                selected_baserender_records,
                selected_baserender_status_rows,
                mo=mo,
                opal_table=opal_table,
                pl=pl,
            )
            return (
                baserender_diagnostic_panel,
                baserender_has_renderable_records,
                baserender_record_selector,
            )
        """
    )


def _selected_record_row_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            baserender_record_selector,
            resolve_notebook_baserender_record_selection,
            selected_baserender_records,
            selected_campaign_baserender_contract,
            selected_campaign_store,
        ):
            baserender_record_id, baserender_record_row, baserender_selection_record = (
                resolve_notebook_baserender_record_selection(
                    selected_campaign_store.records_path,
                    baserender_record_selector.value if baserender_record_selector is not None else None,
                    selected_baserender_records,
                    selected_campaign_baserender_contract,
                )
            )
            return baserender_record_id, baserender_record_row, baserender_selection_record
        """
    )


__all__ = ["render_baserender_record_cells"]
