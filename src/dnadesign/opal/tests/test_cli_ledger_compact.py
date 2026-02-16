# ABOUTME: Exercises ledger compaction for run_meta datasets.
# ABOUTME: Ensures duplicate run_id rows are deduplicated.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_ledger_compact.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal.src import LEDGER_SCHEMA_VERSION
from dnadesign.opal.src import __version__ as OPAL_VERSION
from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.storage.parquet_io import write_parquet_df

from ._cli_helpers import write_campaign_yaml, write_records


def test_ledger_compact_runs_dedupes(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=False)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )

    runs_dir = workdir / "outputs" / "ledger" / "runs.parquet"
    runs_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "event": "run_meta",
        "run_id": "run-1",
        "as_of_round": 0,
        "model__name": "random_forest",
        "model__params": {"n_estimators": 1},
        "objective__name": "sfxi_v1",
        "selection__name": "top_n",
        "selection__params": {"top_k": 1},
        "schema__version": LEDGER_SCHEMA_VERSION,
        "opal__version": OPAL_VERSION,
    }
    df1 = pd.DataFrame([row])
    df2 = pd.DataFrame([{**row, "model__name": "random_forest_v2"}])

    write_parquet_df(runs_dir / "part-000.parquet", df1)
    write_parquet_df(runs_dir / "part-001.parquet", df2)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "ledger-compact",
            "-c",
            str(campaign),
            "--runs",
            "--apply",
        ],
    )
    assert res.exit_code == 0, res.stdout

    out_df = pd.read_parquet(runs_dir)
    assert out_df["run_id"].nunique() == 1
    assert len(out_df) == 1
