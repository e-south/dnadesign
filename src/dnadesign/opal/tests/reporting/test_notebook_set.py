"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/reporting/test_notebook_set.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.opal.src.reporting.notebook_set import build_campaign_set_notebook_view_model
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records


def test_campaign_set_notebook_view_model_collects_campaigns(tmp_path: Path) -> None:
    config_paths = []
    for slug in ["campaign_a", "campaign_b"]:
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records_path = workdir / "records.parquet"
        write_records(records_path, slug=slug)
        config_path = workdir / "campaign.yaml"
        write_campaign_yaml(
            config_path,
            workdir=workdir,
            records_path=records_path,
            slug=slug,
        )
        config_paths.append(config_path)

    payload = build_campaign_set_notebook_view_model(config_paths, round_selector="latest")

    assert payload["schema_version"] == "opal.notebook_campaign_set_view_model.v1"
    assert payload["campaign_count"] == 2
    assert [row["campaign"]["slug"] for row in payload["campaigns"]] == ["campaign_a", "campaign_b"]
    assert payload["campaigns"][0]["campaign"]["config_path"] == str(config_paths[0])
