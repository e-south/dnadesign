"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/__init__.py

Public OPAL package API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .src.analysis.campaign_progress import (
    assess_records_contract,
    assess_records_contract_for_values,
    build_ledger_status_table,
    build_records_preview,
    cli_handoff_lines,
    read_optional_table,
    records_status_lines,
    table_status_lines,
    unavailable_table,
    x_provenance_status_lines,
)
from .src.analysis.facade import (
    CampaignAnalysis,
    available_rounds,
    latest_round,
    latest_run_id,
    require_columns,
)
from .src.analysis.notebook_set_template import render_campaign_set_notebook
from .src.analysis.notebook_template import render_campaign_notebook
from .src.config.loader import load_config
from .src.plots.api import run_campaign_plots
from .src.plots.config import load_plot_config, parse_enabled, parse_tags
from .src.plots.manifests import load_plot_artifact_manifest, load_plot_manifest_index
from .src.registries.plots import describe_plot_kind, list_plot_kinds
from .src.reporting.artifact_garden import build_artifact_garden_audit, prune_stale_artifacts
from .src.reporting.notebook import build_notebook_view_model, smoke_check_notebook
from .src.reporting.notebook_set import build_campaign_set_notebook_view_model
from .src.reporting.predictions import read_campaign_predictions
from .src.reporting.progress import build_campaign_progress, render_campaign_progress_text
from .src.reporting.review import build_campaign_review, load_review_manifest
from .src.runtime.memory_guard import enforce_x_matrix_memory_budget, estimate_x_matrix_memory
from .src.storage.x_contracts import validate_x_parquet_column

__all__ = [
    "CampaignAnalysis",
    "assess_records_contract",
    "assess_records_contract_for_values",
    "available_rounds",
    "build_ledger_status_table",
    "build_artifact_garden_audit",
    "build_campaign_progress",
    "build_campaign_review",
    "build_notebook_view_model",
    "build_campaign_set_notebook_view_model",
    "build_records_preview",
    "cli_handoff_lines",
    "describe_plot_kind",
    "enforce_x_matrix_memory_budget",
    "estimate_x_matrix_memory",
    "latest_round",
    "latest_run_id",
    "list_plot_kinds",
    "load_config",
    "load_plot_artifact_manifest",
    "load_plot_config",
    "load_plot_manifest_index",
    "load_review_manifest",
    "parse_enabled",
    "parse_tags",
    "prune_stale_artifacts",
    "read_optional_table",
    "read_campaign_predictions",
    "records_status_lines",
    "render_campaign_notebook",
    "render_campaign_set_notebook",
    "render_campaign_progress_text",
    "require_columns",
    "run_campaign_plots",
    "smoke_check_notebook",
    "table_status_lines",
    "unavailable_table",
    "validate_x_parquet_column",
    "x_provenance_status_lines",
]
