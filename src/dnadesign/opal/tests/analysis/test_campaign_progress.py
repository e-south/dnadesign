"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/analysis/test_campaign_progress.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dnadesign.opal.src.analysis.campaign_progress import (
    assess_records_contract,
    build_ledger_status_table,
    build_records_preview,
    cli_handoff_lines,
    projection_status_lines,
    read_optional_table,
    table_status_lines,
)
from dnadesign.opal.src.analysis.dashboard.datasets import CampaignInfo


def _campaign_info() -> CampaignInfo:
    return CampaignInfo(
        label="demo",
        path=Path("campaign.yaml"),
        workdir=None,
        slug="demo",
        x_column="opal__view__x",
        y_column="opal__view__y",
        y_expected_length=8,
        model_name="random_forest",
        model_params={},
        objective_name="sfxi_v1",
        objective_params={},
        selection_name="top_n",
        selection_params={},
        training_policy={},
        y_ops=[],
    )


def test_records_contract_ready_without_projection_columns() -> None:
    df = pl.DataFrame(
        {
            "id": ["rec-1"],
            "bio_type": ["promoter"],
            "sequence": ["ACGT"],
            "alphabet": ["DNA"],
            "opal__view__x": [[0.1, 0.2, 0.3]],
        }
    )

    report = assess_records_contract(df, _campaign_info())

    assert report.ready
    assert not report.projection_ready
    assert report.missing_required_columns == ()
    assert "OPAL records remain inspectable" in "\n".join(projection_status_lines(report))


def test_records_contract_requires_configured_x_column() -> None:
    df = pl.DataFrame(
        {
            "id": ["rec-1"],
            "bio_type": ["promoter"],
            "sequence": ["ACGT"],
            "alphabet": ["DNA"],
        }
    )

    report = assess_records_contract(df, _campaign_info())

    assert not report.ready
    assert report.missing_required_columns == ("opal__view__x",)


def test_records_preview_hides_vector_but_marks_presence() -> None:
    df = pl.DataFrame(
        {
            "id": ["rec-1"],
            "bio_type": ["promoter"],
            "sequence": ["ACGT" * 40],
            "alphabet": ["DNA"],
            "opal__view__x": [[0.1, 0.2, 0.3]],
            "opal__demo__label_hist": [[1.0, None]],
        }
    )
    report = assess_records_contract(df, _campaign_info())

    preview = build_records_preview(df, report)

    assert "opal__view__x" not in preview.columns
    assert preview.get_column("x_present").to_list() == [True]
    assert preview.get_column("label_hist_present").to_list() == [True]
    assert preview.get_column("sequence_length").to_list() == [160]


def test_ledger_status_table_is_structured_when_workdir_missing() -> None:
    table = build_ledger_status_table(None)

    assert table.get_column("artifact").to_list() == ["state", "labels", "runs", "predictions"]
    assert set(table.get_column("status").to_list()) == {"missing workdir"}


def test_optional_table_reports_unavailable_without_raising() -> None:
    def _loader() -> pl.DataFrame:
        raise RuntimeError("missing runs sink")

    table = read_optional_table("runs", Path("outputs/ledger/runs.parquet"), _loader)

    assert not table.available
    assert table.df.is_empty()
    assert table.status == "unavailable"
    assert "missing runs sink" in table.message
    assert "runs: **unavailable**" in "\n".join(table_status_lines(table))


def test_cli_handoff_lines_keep_notebook_generation_in_canonical_path() -> None:
    text = "\n".join(cli_handoff_lines("campaign.yaml"))

    assert "Pre-run campaign viewer generation (writes notebook)" in text
    assert "Post-run ledger inspection" in text
    assert "uv run opal validate -c campaign.yaml" in text
    assert "uv run opal status -c campaign.yaml --with-ledger" in text
    assert "uv run opal runs list -c campaign.yaml" in text
    assert "uv run opal record-show -c campaign.yaml" in text
    assert "uv run opal verify-outputs -c campaign.yaml --round latest" in text
    assert "uv run opal plot -c campaign.yaml" in text
    assert "uv run opal notebook generate -c campaign.yaml --round latest --force" in text
    assert "uv run opal notebook run -c campaign.yaml" in text
