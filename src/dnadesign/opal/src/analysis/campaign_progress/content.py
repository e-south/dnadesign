"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/campaign_progress/content.py

Readable content rows for campaign progress notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..dashboard.datasets import CampaignInfo
from .models import RecordsContractReport
from .records import records_status_rows


def campaign_contract_rows(
    info: CampaignInfo | None,
    *,
    config_path: Path | str | None,
    records_path: Path | str | None,
    records_report: RecordsContractReport,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if info is not None:
        rows.extend(
            [
                {"field": "campaign", "value": info.slug},
                {"field": "ownership", "value": info.owner_scope},
            ]
        )
        if info.study_id is not None:
            rows.append({"field": "study", "value": info.study_id})
        rows.extend(
            [
                {"field": "X column", "value": info.x_column},
                {"field": "Y column", "value": info.y_column},
                {"field": "Y expected length", "value": info.y_expected_length},
                {"field": "model", "value": info.model_name},
                {
                    "field": "selection views",
                    "value": ", ".join(
                        f"{view.id}: {view.objective_name} -> {view.selection_name}" for view in info.selection_views
                    ),
                },
            ]
        )
    if config_path is not None:
        rows.append({"field": "config", "value": str(config_path)})
    if records_path is not None:
        rows.append({"field": "records", "value": str(records_path)})
    rows.extend(records_status_rows(records_report))
    return rows


def x_provenance_status_rows(report: RecordsContractReport) -> list[dict[str, str]]:
    if report.x_column:
        if report.x_column in report.missing_required_columns:
            x_state = "missing"
        elif report.x_values_loaded:
            x_state = "present"
        else:
            x_state = "present in records schema; values not loaded in notebook preview"
        return [
            {"field": "X column", "value": report.x_column},
            {"field": "X state", "value": x_state},
            {
                "field": "boundary",
                "value": (
                    "OPAL treats X as an explicit candidate-table contract and does not inspect producer geometry."
                ),
            },
        ]
    return [
        {"field": "X column", "value": "not configured"},
        {
            "field": "execution requirement",
            "value": "OPAL records remain inspectable, but campaign execution needs an explicit X column.",
        },
    ]


def x_provenance_status_lines(report: RecordsContractReport) -> list[str]:
    return [f"- {row['field']}: `{row['value']}`" for row in x_provenance_status_rows(report)]


def cli_handoff_lines(config_path: Path | str) -> list[str]:
    config_text = str(config_path)
    return [
        "### Canonical OPAL inspection commands",
        "",
        "Pre-run campaign viewer generation (writes notebook):",
        "",
        "```bash",
        f"uv run opal validate -c {config_text}",
        f"uv run opal notebook generate -c {config_text} --round latest --force",
        f"uv run opal notebook run -c {config_text}",
        "```",
        "",
        "Post-run ledger inspection:",
        "",
        "```bash",
        f"uv run opal status -c {config_text} --with-ledger",
        f"uv run opal runs list -c {config_text}",
        (
            f"uv run opal record-show -c {config_text} --view <selection-view-id> "
            "--selected-rank 1 --round latest --run-id latest"
        ),
        f"uv run opal verify-outputs -c {config_text} --view <selection-view-id> --round latest",
        f"uv run opal plot -c {config_text} --view <selection-view-id>",
        "```",
    ]
