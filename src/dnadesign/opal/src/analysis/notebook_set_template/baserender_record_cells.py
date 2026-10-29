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
            _record_memory_cell(),
            _record_evidence_bundle_cell(),
            _record_controls_cell(),
            _active_record_selector_cell(),
            _selected_record_row_cell(),
        )
    )


def _record_memory_cell() -> str:
    return block(
        """
        @app.cell
        def _(mo):
            baserender_record_memory, set_baserender_record_memory = mo.state({})
            return baserender_record_memory, set_baserender_record_memory
        """
    )


def _record_evidence_bundle_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            baserender_campaign_model,
            build_notebook_baserender_evidence_bundle,
            mo,
            opal_table,
            pl,
            selected_baserender_round,
            selected_baserender_run_id,
            selected_campaign_baserender_contract,
            selected_campaign_labels_df,
            selected_campaign_store,
        ):
            baserender_record_evidence_bundle = build_notebook_baserender_evidence_bundle(
                selected_campaign_store.records_path,
                selected_campaign_baserender_contract,
                baserender_campaign_model,
                labels_df=selected_campaign_labels_df,
                run_id=selected_baserender_run_id,
                round_value=selected_baserender_round,
                mo=mo,
                opal_table=opal_table,
                pl=pl,
            )
            return baserender_record_evidence_bundle,
        """
    )


def _record_controls_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            baserender_campaign_model,
            baserender_record_evidence_bundle,
            baserender_record_memory,
            build_notebook_baserender_record_controls,
            mo,
            selected_baserender_round,
            selected_baserender_run_id,
            set_baserender_record_memory,
        ):
            _campaign_slug = str(
                (baserender_campaign_model.get("campaign") or {}).get("slug") or ""
            ).strip()
            baserender_record_controls = build_notebook_baserender_record_controls(
                baserender_record_evidence_bundle,
                campaign_slug=_campaign_slug,
                run_id=selected_baserender_run_id,
                round_value=selected_baserender_round,
                review_group_key="handoff",
                deliverable_key="baserender",
                memory=baserender_record_memory,
                set_memory=set_baserender_record_memory,
                mo=mo,
            )
            return baserender_record_controls,
        """
    )


def _active_record_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            baserender_record_controls,
            baserender_record_evidence_bundle,
            selected_baserender_selection_view_id,
        ):
            _view_id = str(selected_baserender_selection_view_id)
            if _view_id not in baserender_record_evidence_bundle:
                raise ValueError(f"Unknown BaseRender record view {_view_id!r}.")
            _bundle = baserender_record_evidence_bundle[_view_id]
            baserender_has_candidate_records = bool(_bundle["has_candidate_records"])
            baserender_has_renderable_records = bool(_bundle["has_renderable_records"])
            baserender_record_selector = baserender_record_controls[_view_id]
            baserender_diagnostic_panel = _bundle["diagnostic_panel"]
            baserender_candidate_records = _bundle["records"]
            selected_baserender_status_rows = _bundle["status_rows"]
            return (
                baserender_diagnostic_panel,
                baserender_has_candidate_records,
                baserender_has_renderable_records,
                baserender_record_selector,
                baserender_candidate_records,
                selected_baserender_status_rows,
            )
        """
    )


def _selected_record_row_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            baserender_candidate_records,
            baserender_record_selector,
            resolve_notebook_baserender_candidate_record,
            selected_campaign_baserender_contract,
            selected_campaign_store,
        ):
            baserender_record_id, baserender_record_row, baserender_candidate_evidence = (
                resolve_notebook_baserender_candidate_record(
                    selected_campaign_store.records_path,
                    baserender_record_selector.value if baserender_record_selector is not None else None,
                    baserender_candidate_records,
                    selected_campaign_baserender_contract,
                )
            )
            return baserender_candidate_evidence, baserender_record_id, baserender_record_row
        """
    )


__all__ = ["render_baserender_record_cells"]
