"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/reporting/test_notebook.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.opal.src.reporting.notebook import build_notebook_view_model
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records


def test_notebook_view_model_includes_artifact_garden_without_records_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workdir = tmp_path / ".var" / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records_path = workdir / "records.parquet"
    write_records(records_path)
    config_path = workdir / "campaign.yaml"
    write_campaign_yaml(config_path, workdir=workdir, records_path=records_path)
    stale_plot = workdir / "outputs" / "plots" / "old.png"
    stale_plot.parent.mkdir(parents=True, exist_ok=True)
    stale_plot.write_bytes(b"stale")

    from dnadesign.opal.src.storage import records_io

    def fail_read_parquet_df(*args, **kwargs):
        raise AssertionError("notebook view model must not read records.parquet")

    monkeypatch.setattr(records_io, "read_parquet_df", fail_read_parquet_df)

    payload = build_notebook_view_model(config_path, round_selector="latest")

    audit = payload["artifact_garden"]
    assert audit["schema_version"] == "opal.artifact_garden.v1"
    assert audit["local_only"] is True
    assert audit["prune_plan"]["requires_apply"] is True
    assert any(row["path"] == str(stale_plot) for row in audit["stale_artifacts"])
