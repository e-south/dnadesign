"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/notebooks/api/generated.py

Re-exports helpers used by generated OPAL notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ...src.analysis import notebook_components as _notebook_components
from ...src.analysis.campaign import CampaignAnalysis
from ...src.analysis.campaign_progress import (
    assess_records_contract_for_schema,
    build_ledger_status_table,
    build_records_preview,
    cli_handoff_lines,
    read_optional_table,
    records_status_lines,
    table_status_lines,
    unavailable_table,
    x_provenance_status_lines,
)
from ...src.analysis.ledger import available_rounds, latest_round, latest_run_id, require_columns
from ...src.plots.config import list_configured_plot_specs, load_plot_config, parse_enabled, parse_tags
from ...src.reporting.campaign_collection import load_campaign_collection_manifest
from ...src.reporting.notebook import build_notebook_view_model
from ...src.reporting.notebook_set import build_campaign_set_notebook_view_model, build_campaign_set_round_options

_NOTEBOOK_COMPONENT_EXPORTS = tuple(_notebook_components.__all__)
globals().update({name: getattr(_notebook_components, name) for name in _NOTEBOOK_COMPONENT_EXPORTS})

__all__ = [
    "CampaignAnalysis",
    "assess_records_contract_for_schema",
    "available_rounds",
    "build_campaign_set_notebook_view_model",
    "build_campaign_set_round_options",
    "build_ledger_status_table",
    "build_notebook_view_model",
    "build_records_preview",
    "cli_handoff_lines",
    "latest_round",
    "latest_run_id",
    "load_campaign_collection_manifest",
    "load_plot_config",
    "list_configured_plot_specs",
    "parse_enabled",
    "parse_tags",
    "read_optional_table",
    "records_status_lines",
    "require_columns",
    "table_status_lines",
    "unavailable_table",
    "x_provenance_status_lines",
] + list(_NOTEBOOK_COMPONENT_EXPORTS)
