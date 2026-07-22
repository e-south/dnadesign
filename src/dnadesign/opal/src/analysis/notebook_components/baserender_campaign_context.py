"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender_campaign_context.py

Role-scoped campaign context for BaseRender notebook evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..campaign import CampaignAnalysis
from .baserender import build_notebook_baserender_contract


def load_notebook_baserender_campaign_context(
    campaign_model: Mapping[str, Any],
) -> tuple[CampaignAnalysis, dict[str, Any], Any, Any]:
    """Load the record, adapter, and run context for one exact campaign."""

    import polars as pl

    campaign = campaign_model.get("campaign") or {}
    analysis = CampaignAnalysis.from_config_path(Path(campaign["config_path"]), allow_dir=True)
    store = analysis.records_store()
    metadata = campaign.get("metadata") or {}
    metadata_path = str(metadata.get("baserender_metadata_records_path") or "").strip() or None
    metadata_columns: list[str] = []
    if metadata_path:
        try:
            metadata_columns = list(pl.scan_parquet(metadata_path).collect_schema().names())
        except Exception:
            metadata_columns = []
    contract = build_notebook_baserender_contract(
        store.schema_columns(),
        records_path=str(store.records_path),
        metadata_records_path=metadata_path,
        metadata_schema_columns=metadata_columns,
    )
    try:
        runs_df = analysis.read_runs()
    except Exception:
        runs_df = pl.DataFrame()
    return analysis, contract, runs_df, store


__all__ = ["load_notebook_baserender_campaign_context"]
