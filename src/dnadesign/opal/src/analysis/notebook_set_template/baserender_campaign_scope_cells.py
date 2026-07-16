"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/baserender_campaign_scope_cells.py

Notebook-set template builders for BaseRender campaign scope cells OPAL analysis notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ._support import block


def render_baserender_campaign_scope_cells() -> str:
    return "\n\n".join(
        (
            _collection_baserender_role_cell(),
            _baserender_campaign_model_cell(),
            _selected_baserender_selection_view_cell(),
            _selected_campaign_baserender_contract_cell(),
        )
    )


def _collection_baserender_role_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            active_view_mode,
            build_notebook_collection_baserender_role_control,
            campaigns,
            collection,
            mo,
            selected_collection_set_choice,
        ):
            collection_baserender_role_choices, baserender_role_ui = (
                build_notebook_collection_baserender_role_control(
                    campaigns,
                    collection,
                    selected_collection_set_choice,
                    active_view_mode=active_view_mode,
                    mo=mo,
                )
            )
            return collection_baserender_role_choices, baserender_role_ui
        """
    )


def _baserender_campaign_model_cell() -> str:
    return block(
        """
        @app.cell
        def _(active_view_mode, baserender_role_ui, build_notebook_baserender_selection_view_control, campaigns, """
        "collection_baserender_role_choices, mo, resolve_notebook_baserender_campaign_model, selected_campaign_model):"
        """
            baserender_campaign_model, selected_baserender_role_choice = (
                resolve_notebook_baserender_campaign_model(
                    active_view_mode=active_view_mode,
                    campaigns=campaigns,
                    role_choices=collection_baserender_role_choices,
                    role_selector_value=baserender_role_ui.value if baserender_role_ui is not None else None,
                    selected_campaign_model=selected_campaign_model,
                )
            )
            baserender_selection_view_options, baserender_selection_view_ui = (
                build_notebook_baserender_selection_view_control(
                    active_view_mode=active_view_mode,
                    campaign_model=baserender_campaign_model,
                    mo=mo,
                )
            )
            return (
                baserender_campaign_model,
                baserender_selection_view_options,
                baserender_selection_view_ui,
                selected_baserender_role_choice,
            )
        """
    )


def _selected_baserender_selection_view_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            active_view_mode,
            baserender_selection_view_options,
            baserender_selection_view_ui,
            resolve_notebook_baserender_selection_view_id,
            selected_selection_view_id,
        ):
            selected_baserender_selection_view_id = resolve_notebook_baserender_selection_view_id(
                active_view_mode=active_view_mode,
                selection_view_options=baserender_selection_view_options,
                selector_value=(
                    baserender_selection_view_ui.value
                    if baserender_selection_view_ui is not None else None
                ),
                campaign_selection_view_id=selected_selection_view_id,
            )
            return selected_baserender_selection_view_id,
        """
    )


def _selected_campaign_baserender_contract_cell() -> str:
    return block(
        """
        @app.cell
        def _(baserender_campaign_model, load_notebook_baserender_campaign_context):
            (
                selected_campaign_analysis,
                selected_campaign_baserender_contract,
                selected_campaign_runs_df,
                selected_campaign_store,
            ) = load_notebook_baserender_campaign_context(baserender_campaign_model)
            return (
                selected_campaign_analysis, selected_campaign_baserender_contract,
                selected_campaign_runs_df, selected_campaign_store,
            )
        """
    )


__all__ = ["render_baserender_campaign_scope_cells"]
