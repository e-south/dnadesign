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
            _selected_campaign_baserender_contract_cell(),
        )
    )


def _collection_baserender_role_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            active_view_mode,
            build_notebook_collection_baserender_role_choices,
            campaigns,
            collection,
            mo,
            selected_collection_set_choice,
        ):
            collection_baserender_role_choices = (
                build_notebook_collection_baserender_role_choices(
                    campaigns,
                    collection,
                    selected_collection_set_choice,
                )
                if active_view_mode == "Campaign set"
                else []
            )
            if collection_baserender_role_choices:
                _labels = [choice["label"] for choice in collection_baserender_role_choices]
                baserender_role_ui = mo.ui.dropdown(_labels, value=_labels[0], label="Label source")
            else:
                baserender_role_ui = None
            return collection_baserender_role_choices, baserender_role_ui
        """
    )


def _baserender_campaign_model_cell() -> str:
    return block(
        """
        @app.cell
        def _(active_view_mode, baserender_role_ui, campaigns, """
        "collection_baserender_role_choices, selected_campaign_model):"
        """
            selected_baserender_role_choice = None
            baserender_campaign_model = selected_campaign_model
            if active_view_mode == "Campaign set" and baserender_role_ui is not None:
                _selected = str(baserender_role_ui.value)
                selected_baserender_role_choice = next(
                    choice for choice in collection_baserender_role_choices if choice["label"] == _selected
                )
                _slug = str(selected_baserender_role_choice["campaign_slug"])
                baserender_campaign_model = next(
                    campaign for campaign in campaigns if str(campaign["campaign"]["slug"]) == _slug
                )
            return baserender_campaign_model, selected_baserender_role_choice
        """
    )


def _selected_campaign_baserender_contract_cell() -> str:
    return block(
        """
        @app.cell
        def _(CampaignAnalysis, Path, baserender_campaign_model, build_notebook_baserender_contract, pl):
            selected_campaign_analysis = CampaignAnalysis.from_config_path(
                Path(baserender_campaign_model["campaign"]["config_path"]),
                allow_dir=True,
            )
            selected_campaign_store = selected_campaign_analysis.records_store()
            _metadata = baserender_campaign_model["campaign"].get("metadata") or {}
            _metadata_records_path = str(_metadata.get("baserender_metadata_records_path") or "").strip() or None
            _metadata_schema_columns = []
            if _metadata_records_path:
                try:
                    _metadata_schema_columns = list(pl.scan_parquet(_metadata_records_path).collect_schema().names())
                except Exception:
                    _metadata_schema_columns = []
            selected_campaign_baserender_contract = build_notebook_baserender_contract(
                selected_campaign_store.schema_columns(),
                records_path=str(selected_campaign_store.records_path),
                metadata_records_path=_metadata_records_path,
                metadata_schema_columns=_metadata_schema_columns,
            )
            try:
                selected_campaign_labels_df = selected_campaign_analysis.read_labels()
            except Exception:
                selected_campaign_labels_df = pl.DataFrame()
            try:
                selected_campaign_runs_df = selected_campaign_analysis.read_runs()
            except Exception:
                selected_campaign_runs_df = pl.DataFrame()
            return (
                selected_campaign_analysis,
                selected_campaign_baserender_contract,
                selected_campaign_labels_df,
                selected_campaign_runs_df,
                selected_campaign_store,
            )
        """
    )


__all__ = ["render_baserender_campaign_scope_cells"]
