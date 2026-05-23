"""Public helper surface for the checked-in OPAL campaign progress notebook."""

from __future__ import annotations

from ...src.analysis.campaign_progress import (
    active_record_rows,
    assess_records_contract,
    build_ledger_status_table,
    build_records_preview,
    campaign_contract_rows,
    cli_handoff_lines,
    records_status_lines,
    x_provenance_status_lines,
    x_provenance_status_rows,
)
from ...src.analysis.dashboard.api import (
    campaign_label_from_path,
    diagnostics_to_lines,
    find_repo_root,
    list_campaign_paths,
    load_campaign_selection,
    load_parquet_cached,
)

__all__ = [
    "assess_records_contract",
    "active_record_rows",
    "build_ledger_status_table",
    "build_records_preview",
    "campaign_label_from_path",
    "campaign_contract_rows",
    "cli_handoff_lines",
    "diagnostics_to_lines",
    "find_repo_root",
    "list_campaign_paths",
    "load_campaign_selection",
    "load_parquet_cached",
    "records_status_lines",
    "x_provenance_status_rows",
    "x_provenance_status_lines",
]
