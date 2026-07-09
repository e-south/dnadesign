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
            baserender_run_ui,
            build_notebook_selected_baserender_record_ids,
            selected_baserender_round,
            selected_campaign_analysis,
        ):
            _run_id = str(baserender_run_ui.value) if baserender_run_ui is not None else None
            selected_baserender_ids, selected_baserender_status_rows = (
                build_notebook_selected_baserender_record_ids(
                    selected_campaign_analysis,
                    round_value=selected_baserender_round,
                    run_id=_run_id,
                )
            )
            return selected_baserender_ids, selected_baserender_status_rows
        """
    )


def _selected_record_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            build_notebook_baserender_record_annotation_counts,
            build_notebook_baserender_record_choices_with_counts,
            build_notebook_baserender_record_options,
            mo,
            select_notebook_baserender_default_record_id,
            selected_baserender_ids,
            selected_campaign_baserender_contract,
            selected_campaign_store,
        ):
            baserender_record_options = build_notebook_baserender_record_options(
                selected_campaign_store.records_path,
                selected_campaign_baserender_contract,
                record_ids=selected_baserender_ids,
            )
            baserender_record_annotation_counts = build_notebook_baserender_record_annotation_counts(
                selected_campaign_store.records_path,
                selected_campaign_baserender_contract,
                record_ids=baserender_record_options,
            )
            _annotation_label = (
                "TFBS"
                if str(selected_campaign_baserender_contract.get("adapter_kind") or "") == "densegen_tfbs"
                else "annotations"
            )
            _choice_rows = build_notebook_baserender_record_choices_with_counts(
                baserender_record_options,
                baserender_record_annotation_counts,
                annotation_label=_annotation_label,
            )
            baserender_record_choices = {choice["label"]: choice["record_id"] for choice in _choice_rows}
            _default_record_id = select_notebook_baserender_default_record_id(
                baserender_record_options,
                baserender_record_annotation_counts,
            )
            _default_label = next(
                (
                    choice["label"]
                    for choice in _choice_rows
                    if str(choice["record_id"]) == str(_default_record_id)
                ),
                next(iter(baserender_record_choices)),
            )
            baserender_record_selector = mo.ui.dropdown(
                baserender_record_choices,
                value=_default_label,
                label="Selected sequence",
                searchable=True,
                full_width=True,
            )
            return baserender_record_annotation_counts, baserender_record_choices, baserender_record_selector
        """
    )


def _selected_record_row_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            baserender_record_selector,
            load_notebook_baserender_record_row,
            selected_campaign_baserender_contract,
            selected_campaign_store,
        ):
            baserender_record_id = str(baserender_record_selector.value)
            baserender_record_row = load_notebook_baserender_record_row(
                selected_campaign_store.records_path,
                baserender_record_id,
                selected_campaign_baserender_contract,
            )
            return baserender_record_id, baserender_record_row
        """
    )


__all__ = ["render_baserender_record_cells"]
